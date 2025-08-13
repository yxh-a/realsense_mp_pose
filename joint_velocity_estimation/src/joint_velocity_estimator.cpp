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
    joint_velocity_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/arm/joint_velocities", 10);

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
    Eigen::Matrix<double, Eigen::Dynamic, 6> J_arm_pinv;
    J_arm_pinv.setZero(arm_model_.nv, 6);
    J_arm_pinv = J_arm.completeOrthogonalDecomposition().pseudoInverse();
    if (!isValidMatrix(J_arm_pinv))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid arm Jacobian pseudo-inverse (NaNs detected)");
      return;
    }
    dq_arm_ = J_arm_pinv * v_hand_;
    // double dt = (this->now() - rclcpp::Time(msg->header.stamp)).seconds();
    // q_arm_ += dq_arm_ * dt;
    std_msgs::msg::Float32MultiArray joint_velocities_msg;
    joint_velocities_msg.data.resize(arm_model_.nv);
    for (size_t i = 0; i < arm_model_.nv; ++i)
    {
      joint_velocities_msg.data[i] = dq_arm_[i]/M_PI*180.0;  // Convert to degrees
    }
    joint_velocity_pub_->publish(joint_velocities_msg);
  }

  void jointCallback_arm(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->name.size() != arm_model_.nq || msg->position.size() != arm_model_.nq)
    {
      RCLCPP_WARN(this->get_logger(), "Arm joint state size mismatch with model.");
      return;
    }

    q_arm_ = Eigen::VectorXd::Map(msg->position.data(), arm_model_.nq);
    pinocchio::forwardKinematics(arm_model_, arm_data_, q_arm_);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);

    // // safe joint names once
    // if (updated_arm_joint_states_pub_->get_subscription_count() > 0)
    // {
    //   sensor_msgs::msg::JointState updated_joint_states;
    //   updated_joint_states.header = msg->header;
    //   updated_joint_states.name = msg->name;  // Assuming the names are the same
    //   updated_joint_states.position.resize(q_arm_.size());
    //   updated_joint_states.velocity.resize(q_arm_.size());
    //   for (size_t i = 0; i < q_arm_.size(); ++i)
    //   {
    //     updated_joint_states.position[i] = q_arm_[i];
    //     updated_joint_states.velocity[i] = dq_arm_[i];
    //   }
    //   updated_arm_joint_states_pub_->publish(updated_joint_states);
    //   // RCLCPP_INFO(this->get_logger(), "Published updated arm joint states.");
    // }
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

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_joint_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr updated_arm_joint_states_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr joint_velocity_pub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EndEffectorVelocityNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

