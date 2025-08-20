#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "std_msgs/msg/float32_multi_array.hpp"


#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <chrono>

using namespace std::chrono_literals;


class EndEffectorVelocityNode : public rclcpp::Node
{
public:
  EndEffectorVelocityNode() : Node("ee_velocity_node")
  {
    // --------------- Robot Model Initialization ---------------
    RCLCPP_INFO(this->get_logger(), "Initializing robot model...");
    // Load URDF and build the model
    std::string urdf_path = ament_index_cpp::get_package_share_directory("lbr_description") + "/urdf/iiwa7/iiwa7.urdf";
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);

    q_ = Eigen::VectorXd::Zero(model_.nq);
    dq_ = Eigen::VectorXd::Zero(model_.nv);
    pinocchio::framesForwardKinematics(model_, data_, q_);

    RCLCPP_INFO(this->get_logger(), "Model loaded: %s", model_.name.c_str());

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/lbr/joint_states", 10,
      std::bind(&EndEffectorVelocityNode::jointCallback, this, std::placeholders::_1));

    twist_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/lbr/ee_velocity", 10);

    // set the frame name of end effector
    ee_frame_name_ = "lbr_link_ee";  // change to your robot's actual end effector frame
    ee_frame_id_ = model_.getFrameId(ee_frame_name_);

    // ---------------- Human Arm Model Initialization ----------------
    RCLCPP_INFO(this->get_logger(), "Initializing human arm model...");
    std::string human_arm_urdf_path = ament_index_cpp::get_package_share_directory("image_pose_tracking") + "/config/right_arm.urdf";
    pinocchio::urdf::buildModel(human_arm_urdf_path, arm_model_);
    arm_data_ = pinocchio::Data(arm_model_);

    RCLCPP_INFO(this->get_logger(), "Human arm model initialized.");
    hand_frame_name_ = "RightHand";  // change to your robot's actual hand frame
    hand_frame_id_ = arm_model_.getFrameId(hand_frame_name_);
    RCLCPP_INFO(this->get_logger(), "Human arm model loaded: %s", arm_model_.name.c_str());
    
    // subscribe to the human arm joint states
    q_arm_ = Eigen::VectorXd::Zero(arm_model_.nq);
    dq_arm_ = Eigen::VectorXd::Zero(arm_model_.nv);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_arm_);

    arm_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/optimized_arm/joint_states", 10,
      std::bind(&EndEffectorVelocityNode::jointCallback_arm, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to human arm joint states.");
    updated_arm_joint_states_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/updated_arm/joint_states", 10);

    // Initialize static transformations
    // TF Hand-to-EE transform
    // read the hand-to-EE transform from the YAML file
    RCLCPP_INFO(this->get_logger(), "Loading hand-to-EE transform from YAML");
    std::string config_path = ament_index_cpp::get_package_share_directory("pose_optimization") + "/config/parameters.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    if (!config["ee2hand"])
    {
        RCLCPP_ERROR(this->get_logger(), "ee2hand configuration not found in %s", config_path);
        return;
    }
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    if (config["ee2hand"]["translation"])
    {
        translation = Eigen::Vector3d(
            config["ee2hand"]["translation"][0].as<double>(),
            config["ee2hand"]["translation"][1].as<double>(),
            config["ee2hand"]["translation"][2].as<double>()
        );
    }
    Eigen::Vector3d rotation = Eigen::Vector3d::Zero();
    if (config["ee2hand"]["rotation"])
    {
        rotation = Eigen::Vector3d(
            config["ee2hand"]["rotation"][0].as<double>(),
            config["ee2hand"]["rotation"][1].as<double>(),
            config["ee2hand"]["rotation"][2].as<double>()
        );
    }
    T_eehand_ = pinocchio::SE3::Identity();
    Eigen::Matrix3d R = Eigen::AngleAxisd(rotation[0], Eigen::Vector3d::UnitX()).toRotationMatrix()
        * Eigen::AngleAxisd(rotation[1], Eigen::Vector3d::UnitY()).toRotationMatrix()
        * Eigen::AngleAxisd(rotation[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_eehand_.rotation() = R;
    T_eehand_.translation() = translation;

    T_handee_ = T_eehand_.inverse();

    // TF Robot to Camera transform
    if (!config["world2camera"])
    {
      RCLCPP_ERROR(this->get_logger(), "world2camera configuration not found in %s", config_path);
      return;
    }
    translation = Eigen::Vector3d::Zero();
    if (config["world2camera"]["translation"])
    {
      translation = Eigen::Vector3d(
          config["world2camera"]["translation"][0].as<double>(),
          config["world2camera"]["translation"][1].as<double>(),
          config["world2camera"]["translation"][2].as<double>()
      );
    }
    rotation = Eigen::Vector3d::Zero();
    if (config["world2camera"]["rotation"])
    {
      rotation = Eigen::Vector3d(
          config["world2camera"]["rotation"][0].as<double>(),
          config["world2camera"]["rotation"][1].as<double>(),
          config["world2camera"]["rotation"][2].as<double>()
      );
    }
    T_worldcamera_ = pinocchio::SE3::Identity();
    R = Eigen::AngleAxisd(rotation[0], Eigen::Vector3d::UnitX()).toRotationMatrix()
        * Eigen::AngleAxisd(rotation[1], Eigen::Vector3d::UnitY()).toRotationMatrix()
        * Eigen::AngleAxisd(rotation[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_worldcamera_.rotation() = R;
    T_worldcamera_.translation() = translation;
    RCLCPP_INFO(this->get_logger(), "Robot to Camera transform initialized");

    // TF Camera to Shoulder transform can be read from arm_model_
    T_camerashoulder_ = arm_data_.oMf[arm_model_.getFrameId("RightShoulder")];
    if (!T_camerashoulder_.rotation().allFinite() || !T_camerashoulder_.translation().allFinite())
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid Camera to Shoulder transform (NaNs detected)");
      return;
    }
    T_worldshoulder_ = T_worldcamera_ * T_camerashoulder_;

    // initialize Kalman filter
    RCLCPP_INFO(this->get_logger(), "Initializing Kalman filter...");
    std::string kf_config_path = ament_index_cpp::get_package_share_directory("joint_velocity_estimation") + "/config/joint_velocity_config.yaml";
    YAML::Node kf_config = YAML::LoadFile(kf_config_path);
    if (!kf_config["parameters"])
    {
      RCLCPP_ERROR(this->get_logger(), "parameters configuration not found in %s", kf_config_path);
      return;
    }
    if (kf_config["parameters"]["kf"])
    {
      double q_dq = kf_config["parameters"]["kf"]["q_dq"].as<double>(1.0e-3);
      double q_ddq = kf_config["parameters"]["kf"]["q_ddq"].as<double>(1.0e-1);
      kf_.init(model_.nv, q_dq, q_ddq);
      rate_hz_ = kf_config["parameters"]["rates"]["control_rate_hz"].as<double>(200.0);
    }
    if (kf_config["parameters"]["task_measurement"])
    {
      double sigma_min = kf_config["parameters"]["task_measurement"]["sigma_min"].as<double>(0.20);
      double sigma_max = kf_config["parameters"]["task_measurement"]["sigma_max"].as<double>(1.00);
      double v0 = kf_config["parameters"]["task_measurement"]["v0"].as<double>(0.05); 
    }
    
    if (kf_config["parameters"]["vision_measurement"])
    {
      double sigma_vis = kf_config["parameters"]["vision_measurement"]["sigma_vis"].as<double>(0.25);
      double freshness_max_sec = kf_config["parameters"]["vision_measurement"]["freshness_max_sec"].as<double>(0.20);
      RCLCPP_INFO(this->get_logger(), "Vision measurement initialized with sigma: %.2f, freshness: %.2f sec", sigma_vis, freshness_max_sec);
    }
    if (kf_config["parameters"]["output"])
    {
      print_velocity_ = kf_config["parameters"]["output"]["print_velocity"].as<bool>(true);
      print_sigmas_ = kf_config["parameters"]["output"]["print_sigmas"].as<bool>(true);
      RCLCPP_INFO(this->get_logger(), "Output configuration: print_velocity = %s", print_velocity_ ? "true" : "false");
      RCLCPP_INFO(this->get_logger(), "Output configuration: print_sigmas = %s", print_sigmas_ ? "true" : "false");
    }
    
  }

private:
  void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->position.size() != model_.nq || msg->velocity.size() != model_.nv)
    {
      RCLCPP_WARN(this->get_logger(), "Joint state size mismatch with model.");
      return;
    }

    Eigen::VectorXd q = Eigen::VectorXd::Map(msg->position.data(), model_.nq);
    Eigen::VectorXd dq = Eigen::VectorXd::Map(msg->velocity.data(), model_.nv);

    // Compute Jacobian
    pinocchio::computeJointJacobians(model_, data_, q);
    pinocchio::framesForwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    T_worldee_ = data_.oMf[ee_frame_id_];
    if (!isValidTransform(T_worldee_))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid end effector transform (NaNs detected)");
      return;
    }
    T_worldhand_ = T_worldee_ * T_eehand_;
    T_shoulderhand_ = T_worldshoulder_.inverse() * T_worldhand_;


    Eigen::Matrix<double, 6, Eigen::Dynamic> J;
    J.setZero(6, model_.nv);
    J = pinocchio::getFrameJacobian(model_, data_, ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);

    v_ee_ = J * dq;

    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header.stamp = msg->header.stamp;
    twist_msg.header.frame_id = "lbr_link_ee";  // change to your robot's actual end effector frame

    twist_msg.twist.linear.x = v_ee_[0];
    twist_msg.twist.linear.y = v_ee_[1];
    twist_msg.twist.linear.z = v_ee_[2];
    twist_msg.twist.angular.x = v_ee_[3];
    twist_msg.twist.angular.y = v_ee_[4];
    twist_msg.twist.angular.z = v_ee_[5];

    twist_pub_->publish(twist_msg);

    // transform the twist of EE in world frame to twist of hand in shoulder frame
    Ad_T = T_shoulderhand_.toActionMatrix();
    if (!isValidMatrix(Ad_T))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid action matrix (NaNs detected)");
      return;
    }
    v_hand_ = Ad_T * v_ee_;

    // get arm jacboian (7DoF)
    pinocchio::computeJointJacobians(arm_model_, arm_data_, q_arm_);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_arm_);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_arm;
    J_arm.setZero(6, arm_model_.nv);
    J_arm = pinocchio::getFrameJacobian(arm_model_, arm_data_,hand_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
    if (!isValidMatrix(J_arm))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid arm Jacobian (NaNs detected)");
      return;
    }
    // Compute pseudo-inverse of the arm Jacobian
    Eigen::Matrix<double, Eigen::Dynamic, 6> J_arm_pinv(arm_model_.nv, 6);
    J_arm_pinv = J_arm.completeOrthogonalDecomposition().pseudoInverse();
    if (!isValidMatrix(J_arm_pinv))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid arm Jacobian pseudo-inverse (NaNs detected)");
      return;
    }
    // dq_arm_ = J_arm_pinv * v_hand_;
    // ===== Multi-rate fusion =====
    // Predict every robot tick
    kf_.predict();
    // (A) Task-space dq from J# v_hand (fast). Trust less when EE is slow.
    Eigen::VectorXd dq_task = J_arm_pinv * v_hand_;
    double vnorm = v_hand_.head<3>().norm() + v_hand_.tail<3>().norm(); // linear + angular speed
    // at very low end effector speed, the robot provides limited information about the arm joint velocities
    // therefore we scale the sigma of the task-space measurement to be larger
    double sigma_task = task_sigma_min_;
    if (vnorm < v0_)
    {
      sigma_task = task_sigma_max_;
    }
    
    if (dq_task.allFinite()) kf_.update_dq(dq_task, sigma_task);

    // (B) Vision dq (slow). Weight by confidence and recency.
    if (last_vis_stamp_.nanoseconds() > 0){
      double age = (this->now() - last_vis_stamp_).seconds();

      if (age < freshness_max_sec_ && dq_vis_.size()==arm_model_.nv){
        if (dq_vis_.allFinite()) kf_.update_dq(dq_vis_, sigma_vis_);
      }
    }
    // RCLCPP_INFO(this->get_logger(), "KF state: %d, dq size: %d, dtc: %.3f", kf_.initialized, kf_.dq().size(), kf_.dtc);

    dq_arm_ = kf_.dq();
    if (print_velocity_)
    {
      RCLCPP_INFO(this->get_logger(), "Hand velocity: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
                  v_hand_[0], v_hand_[1], v_hand_[2], v_hand_[3], v_hand_[4], v_hand_[5]);
      RCLCPP_INFO(this->get_logger(), "Estimated arm joint velocities (deg): %f, %f, %f, %f, %f, %f, %f",
                dq_arm_[0]*180.0/M_PI, dq_arm_[1]*180.0/M_PI, dq_arm_[2]*180.0/M_PI,
                dq_arm_[3]*180.0/M_PI, dq_arm_[4]*180.0/M_PI, dq_arm_[5]*180.0/M_PI, dq_arm_[6]*180.0/M_PI);
    }
    if (print_sigmas_)
    {
      RCLCPP_INFO(this->get_logger(), "Task-space sigma: %.2f, Vision sigma: %.2f", sigma_task, sigma_vis_);
    }
    // double dt = (this->now() - rclcpp::Time(msg->header.stamp)).seconds();
    // q_arm_ += dq_arm_ * dt;

    sensor_msgs::msg::JointState updated_arm_joint_state;
    updated_arm_joint_state.header.stamp = this->now();
    updated_arm_joint_state.name = arm_model_.names;
    Eigen::VectorXd q_arm_updated = q_arm_ + dq_arm_ * (1); //
    updated_arm_joint_state.position = std::vector<double>(q_arm_.data(), q_arm_.data() + q_arm_.size());
    updated_arm_joint_state.velocity = std::vector<double>(dq_arm_.data(), dq_arm_.data() + dq_arm_.size());
    
    updated_arm_joint_states_pub_->publish(updated_arm_joint_state);

  }

  void jointCallback_arm(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->name.size() != arm_model_.nq || msg->position.size() != arm_model_.nq)
    {
      RCLCPP_WARN(this->get_logger(), "Arm joint state size mismatch with model.");
      return;
    }

    q_arm_ = Eigen::VectorXd::Map(msg->position.data(), arm_model_.nq);
    dq_vis_ = Eigen::VectorXd::Map(msg->velocity.data(), arm_model_.nv);
    last_vis_stamp_ = this->now();

    pinocchio::forwardKinematics(arm_model_, arm_data_, q_arm_);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);
  }

  bool isValidTransform(const pinocchio::SE3 &T)
  {
    return T.rotation().allFinite() && T.translation().allFinite();
  }

  bool isValidMatrix(const Eigen::MatrixXd &M)
  {
    return M.allFinite();
  }


  pinocchio::Model model_, arm_model_;
  pinocchio::Data data_, arm_data_;
  std::string ee_frame_name_, hand_frame_name_;
  pinocchio::FrameIndex ee_frame_id_, hand_frame_id_;
  pinocchio::SE3 T_eehand_, T_handee_, T_worldcamera_,T_camerashoulder_,T_worldee_,T_worldhand_,T_shoulderhand_,T_worldshoulder_;
  Eigen::VectorXd q_arm_, dq_arm_, q_, dq_;
  Eigen::VectorXd v_hand_, v_ee_;
  Eigen::Matrix<double, 6, 6> Ad_T;

  // Kalman filter parameters
  double q_dq_, q_ddq_; // process noise for dq and ddq
  double rate_hz_ = 200.0; // control rate in Hz
  double task_sigma_min_ = 0.20; // minimum sigma for task-space measurement
  double task_sigma_max_ = 1.00; // maximum sigma for task-space measurement
  double v0_ = 0.05; // scale for task-space measurement sigma

  double freshness_max_sec_ = 0.20; // max age of vision measurement to consider it valid
  bool use_confidence_ = false; // whether to use confidence in vision measurement
  double sigma_vis_ = 0.25; // constant vision noise (rad/s)
  double sigma_vis_base_ = 0.50; // base sigma if using confidence

  bool print_velocity_ = true; // whether to print joint velocities to console
  bool print_sigmas_ = true; // whether to print measurement and task sigmas to console




  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_; 
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_joint_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr updated_arm_joint_states_pub_;

  struct DQKalman{
    int n{}; double dtc{};
    // State: x = [dq; ddq] (2n)
    Eigen::VectorXd x;              // (2n)
    Eigen::MatrixXd F, Q, P, I;     // (2n x 2n)
    bool initialized{false};
    void init(int n_, double dtc_, double q_dq=1e-3, double q_ddq=1e-1){
      n=n_; dtc=dtc_;
      x = Eigen::VectorXd::Zero(2*n);
      F = Eigen::MatrixXd::Zero(2*n,2*n);
      Q = Eigen::MatrixXd::Zero(2*n,2*n);
      P = 1e-2 * Eigen::MatrixXd::Identity(2*n,2*n);
      I = Eigen::MatrixXd::Identity(2*n,2*n);
      F.block(0,0,n,n).setIdentity();
      F.block(0,n,n,n) = dtc * Eigen::MatrixXd::Identity(n,n);
      F.block(n,n,n,n).setIdentity();
      Q.block(0,0,n,n) = q_dq  * Eigen::MatrixXd::Identity(n,n);
      Q.block(n,n,n,n) = q_ddq * Eigen::MatrixXd::Identity(n,n);
      initialized = true;
    }
    void predict(){ 
      if(!initialized) return; x = F*x; P = F*P*F.transpose() + Q; 
    }
    void update_dq(const Eigen::VectorXd& z, double sigma){
      if(!initialized) return;
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n,2*n);
      H.block(0,0,n,n).setIdentity();
      Eigen::MatrixXd R = (sigma*sigma) * Eigen::MatrixXd::Identity(n,n);
      Eigen::VectorXd y = z - H*x;
      Eigen::MatrixXd S = H*P*H.transpose() + R;
      Eigen::MatrixXd K = P*H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(n,n));
      x = x + K*y;
      P = (I - K*H) * P;
    }
    Eigen::VectorXd dq() const { return x.head(n); }
  } kf_;

  Eigen::VectorXd dq_vis_;
  double conf_vis_{0.0};
  rclcpp::Time last_vis_stamp_;

};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EndEffectorVelocityNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

