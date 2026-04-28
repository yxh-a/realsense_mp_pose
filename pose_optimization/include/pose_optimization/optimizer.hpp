#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

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
    void joint_state_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    // Pinocchio model and data
    pinocchio::Model model_;
    pinocchio::Data data_;
    std::vector<std::string> joint_names_;
    std::vector<std::string> opt_joint_names_;
    Eigen::VectorXd q_init_, q, q_prev_, q_prev2_;
    bool have_prev_, have_prev2_;
    double w_vel_, w_acc_;

    // Transforms
    Eigen::Isometry3d hand_to_ee_, ee_to_hand_;
    pinocchio::SE3 T_shoulder_hand, T_shoulder_hand_ref, T_base_hand, T_base_shoulder;
    Eigen::Matrix3d R_diff;
    pinocchio::FrameIndex hand_idx, sh_idx;
    std::string shoulder_frame_name_, hand_frame_name_;

    // ROS 2 interfaces
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    geometry_msgs::msg::TransformStamped tf_shoulder2ee;

    // time keeper
    rclcpp::Time last_update_time_;

    // optimization parameters
    nlopt_opt opt_;

    // Operational parameters
    std::string method_;
    bool print_error_before_loop = false;
    bool print_error_after_loop = false;
    bool print_error_in_loop = false;
    bool print_joint_angles = false;
    bool print_critical_transforms = true;
    
    // Nlopt optimization parameters
    std::string algorithm = "LN_COBYLA"; // Default algorithm
    int max_iterations = 100;
    double tolerance = 1e-4;
    double pos_weight = 10.0; // Weight for position error in cost function
    double rot_weight = 1.0; // Weight for rotation error in cost function
    std::vector<double> joint_weights = {10, 10, 10, 10, 1.0, 1.0, 1.0}; // Weights for joint angles
    double joint_penalty_weight = 1; // Penalty weight for joint angles

    // velocity smoothing
    std::vector<std::vector<double>> dq_window_;


    //sensitivity analysis parameters
    double sigma;
    bool apply_sensitivity;
    Eigen::Vector3d noise;
    Eigen::Matrix3d R_noise;

};
