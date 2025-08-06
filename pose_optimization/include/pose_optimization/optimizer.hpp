#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Dense>
#include <nlopt.h>
#include <string>
#include <vector>

class PoseOptimizer : public rclcpp::Node
{
public:
    PoseOptimizer();
    static double costFunction(unsigned n, const double* x, double* grad, void* data);

private:
    // Callbacks
    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

    // Pinocchio model and data
    pinocchio::Model model_;
    pinocchio::Data data_;
    std::vector<std::string> joint_names_;
    Eigen::VectorXd q_init_, q;
    // Transforms
    Eigen::Isometry3d hand_to_ee_, ee_to_hand_;
    pinocchio::SE3 T_shoulder_hand, T_shoulder_hand_ref, T_base_hand, T_base_shoulder;
    Eigen::Matrix3d R_diff;
    pinocchio::FrameIndex hand_idx, sh_idx;

    // ROS 2 interfaces
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    geometry_msgs::msg::TransformStamped tf_shoulder2ee;

    // optimization parameters
    nlopt_opt opt_;

    // Operational parameters
    std::string method_;
    bool print_error_before_loop = false;
    bool print_error_after_loop = false;
    bool print_error_in_loop = false;
    bool print_joint_angles = false;
    
    // Nlopt optimization parameters
    std::string algorithm = "LN_COBYLA"; // Default algorithm
    int max_iterations = 100;
    double tolerance = 1e-4;
    double pos_weight = 10.0; // Weight for position error in cost function
    double rot_weight = 1.0; // Weight for rotation error in cost function
    std::vector<double> joint_weights = {10, 10, 10, 10, 1.0, 1.0, 1.0}; // Weights for joint angles
    double joint_penalty_weight = 1; // Penalty weight for joint angles

    // SVD optimization parameters
    double convergence_threshold = 1e-2; // Threshold for convergence
    double damping_factor = 1e-2; // Damping factor for SVD optimization
    bool print_jacobian = false; // Print Jacobian matrix during optimization
};
