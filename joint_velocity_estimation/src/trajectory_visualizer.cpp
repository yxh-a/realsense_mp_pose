#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <mutex>
#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>


class ArmTrajFromDQ : public rclcpp::Node
{
public:
  ArmTrajFromDQ() : Node("arm_traj_from_dq")
  {
    // ---- Parameters ----
    // JointState with positions
    joint_state_topic_ = this->declare_parameter<std::string>("joint_state_topic", "/updated_arm/joint_states");
    // Publish rate (integration step)
    publish_rate_hz_ = this->declare_parameter<double>("publish_rate_hz", 0.5);
    // motion frequency 
    motion_freq_hz_ = this->declare_parameter<double>("motion_freq_hz", 30.0);
    // How long the displayed trajectory lasts
    horizon_seconds_ = this->declare_parameter<double>("horizon_seconds", 3.0);
    // Scaling (optional) if dq is noisy or too large
    dq_scale_ = this->declare_parameter<double>("dq_scale", 1.0);
    // MoveIt SRDF robot name (can be left empty)
    model_id_ = this->declare_parameter<std::string>("model_id", "upper_arm"); // e.g., "iiwa" or your SRDF name

    // ---- Publishers ----
    disp_pub_ = this->create_publisher<moveit_msgs::msg::DisplayTrajectory>("/display_planned_path", 10);

    // ---- Subscribers ----
    js_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      joint_state_topic_, 10,
      std::bind(&ArmTrajFromDQ::onJointState, this, std::placeholders::_1));

   
    // ---- Timer for integration/publish ----
    const auto dt = std::chrono::duration<double>(1.0 / publish_rate_hz_);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(dt),
      std::bind(&ArmTrajFromDQ::onTimer, this));
    

    // load static transform if provided
    std::string kf_config_path = ament_index_cpp::get_package_share_directory("joint_velocity_estimation") + "/config/joint_velocity_config.yaml";
    YAML::Node kf_config = YAML::LoadFile(kf_config_path);
    T_ = geometry_msgs::msg::Transform();
    if (kf_config["parameters"]["visualizer"]["static_transformation"]) {
      const auto& trans = kf_config["parameters"]["visualizer"]["static_transformation"];
      if (trans.size() == 7) {
        T_.translation.x = trans[0].as<double>();
        T_.translation.y = trans[1].as<double>();
        T_.translation.z = trans[2].as<double>();
        T_.rotation.x = trans[3].as<double>();
        T_.rotation.y = trans[4].as<double>();
        T_.rotation.z = trans[5].as<double>();
        T_.rotation.w = trans[6].as<double>();
      }
    }

    RCLCPP_INFO(get_logger(), "Initialized ArmTrajFromDQ node");
  }

private:
  // ---- Callbacks ----
  void onJointState(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->name.empty() || msg->position.empty()) return;
    std::lock_guard<std::mutex> lock(mtx_);
    js_raw_ = *msg;
    
    have_js_ = true;
    // Store positions
  }
  // ---- Integration and publish ----
  void onTimer()
  {
    if (!have_js_) return;
    
    sensor_msgs::msg::JointState js;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      if (!have_js_) return; // need both
      js = js_raw_;
      have_js_ = false; // wait for next update
    }
    
    q_.resize(joint_names_.size(), 0.0);
    q_ = js.position; // copy positions
    dq_.resize(joint_names_.size(), 0.0);
    dq_ = js.velocity; // copy velocities

    if (dq_.size() != q_.size()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "dq size (%zu) != positions size (%zu); waiting...",
                           dq_.size(), q_.size());
      return;
    }

    // Build a short trajectory starting from current state
    moveit_msgs::msg::DisplayTrajectory disp;
    disp.model_id = model_id_;

    // trajectory_start = current measured state (so RViz knows where we begin)
    disp.trajectory_start.joint_state = makeCurrentJointStateMsg();
    disp.trajectory_start.joint_state.header.frame_id = "camera_depth_optical_frame"; // optional

    moveit_msgs::msg::RobotTrajectory rob_traj;
    trajectory_msgs::msg::JointTrajectory jt;
    jt.joint_names = joint_names_;
    jt.header.frame_id = "camera_depth_optical_frame"; // optional

    // Make N points over horizon_seconds_
    const int N = horizon_seconds_ * motion_freq_hz_; // 30 Hz display rate
    std::vector<double> q = q_; // copy current positions
    const double dt = 1.0 / motion_freq_hz_; // 30 Hz display rate

    for (int k = 0; k < N; ++k) {
      // Euler integration: q_{k+1} = q_k + dq * dt
      // Optional scale if you need to tone down the motion
      for (size_t j = 0; j < q.size(); ++j) {
        q[j] += dq_[j] * dt;
      }

      trajectory_msgs::msg::JointTrajectoryPoint p;
      p.positions = q;
    //   p.velocities = dq_; // just for visualization; not used by RViz
      // time_from_start is in seconds; k+1 because first point is after one dt
      rclcpp::Duration t_from_start = rclcpp::Duration::from_seconds((k + 1) * dt);
      p.time_from_start = t_from_start;

      jt.points.push_back(std::move(p));
    }

    rob_traj.joint_trajectory = std::move(jt);
    disp.trajectory.push_back(std::move(rob_traj));

    auto &md = disp.trajectory_start.multi_dof_joint_state;
    md.joint_names = {"world_to_camera"};
    md.transforms = {T_};

    disp_pub_->publish(disp);
    RCLCPP_INFO(get_logger(), "Published trajectory with %zu points", N);
  }

  sensor_msgs::msg::JointState makeCurrentJointStateMsg() const
  {
    sensor_msgs::msg::JointState js;
    js.name = joint_names_;
    js.position = q_;
    // You could also fill velocities_ if you store them; not required
    return js;
  }

private:
  // Params
  std::string joint_state_topic_;
  std::string model_id_;
  double publish_rate_hz_{0.5};
  double motion_freq_hz_{30.0};
  double horizon_seconds_{3.0};
  double dq_scale_{1.0};

  // Pub/Sub
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr disp_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // State
  std::vector<std::string> joint_names_ {
        "jRightShoulder_rotx","jRightShoulder_rotz","jRightShoulder_roty",
        "jRightElbow_rotz","jRightElbow_roty","jRightWrist_rotx","jRightWrist_rotz"
  }; // default names
  std::unordered_map<std::string, size_t> name_to_index_;
  std::vector<double> dq_, q_;
  geometry_msgs::msg::Transform T_;

  // Mutex for thread safety if needed
  std::mutex mtx_;
  sensor_msgs::msg::JointState js_raw_;
  bool have_js_{false};


};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmTrajFromDQ>());
  rclcpp::shutdown();
  return 0;
}
