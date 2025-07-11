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

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>


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

    // Initialize static transformations

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

    Eigen::Matrix<double, 6, Eigen::Dynamic> J;
    J.setZero(6, model_.nv);
    J = pinocchio::getFrameJacobian(model_, data_, ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);

    Eigen::VectorXd v = J * dq;

    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header.stamp = msg->header.stamp;
    twist_msg.header.frame_id = "world"; 

    twist_msg.twist.linear.x = v[0];
    twist_msg.twist.linear.y = v[1];
    twist_msg.twist.linear.z = v[2];
    twist_msg.twist.angular.x = v[3];
    twist_msg.twist.angular.y = v[4];
    twist_msg.twist.angular.z = v[5];

    twist_pub_->publish(twist_msg);
  }

  pinocchio::Model model_, arm_model_;
  pinocchio::Data data_, arm_data_;
  std::string ee_frame_name_, hand_frame_name_;
  pinocchio::FrameIndex ee_frame_id_, hand_frame_id_;
  pinocchio::SE3 ee_to_hand_transform_, robot_to_camera_transform_, camera_to_shoulder_transform_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;

};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EndEffectorVelocityNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

