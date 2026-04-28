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


class VelocityEstimator : public rclcpp::Node
{
public:
    VelocityEstimator();

private:
    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void jointCallback_arm(const sensor_msgs::msg::JointState::SharedPtr msg);

    bool isValidTransform(const pinocchio::SE3 &T);
    bool isValidMatrix(const Eigen::MatrixXd &M);

    // -------------------- Kalman filter --------------------
    struct DQKalman
    {
        int n{0};
        double dtc{0.0};

        // State: x = [q; dq], size 2*n
        Eigen::VectorXd x;
        Eigen::MatrixXd P, I;
        bool initialized{false};

        void init(int n_, double dtc_, double Pq0 = 1e-2, double Pdq0 = 1e-2);
        void set_state(const Eigen::VectorXd& q0, const Eigen::VectorXd& dq0);
        void predict(double qa, double velocity_decay, double acc_cap, double dt = -1.0);

        void correct_task_velocity(const Eigen::Matrix<double,6,Eigen::Dynamic>& J,
                                   const Eigen::Matrix<double,6,1>& v_meas,
                                   double r_sigma_lin, double r_sigma_rot);
        void correct_joint_measurement(const Eigen::VectorXd& q_meas,
                                       const Eigen::VectorXd& dq_meas,
                                       double sigma_q, double sigma_dq);

        Eigen::VectorXd q()  const;
        Eigen::VectorXd dq() const;
    };
    
    // Kalman filter parameters
    double q_q_, q_dq_, q_a_;
    double rate_hz_ = 250.0; // control rate in Hz
    double velocity_decay_ = 0.0; // exponential decay toward zero velocity
    double acc_cap_ = 10.0; // cap on acceleration implied by prediction
    double r_sigma_lin_ = 0.02; // measurement noise
    double r_sigma_rot_ = 0.05; // measurement noise
    double sigma_q_ = 0.01; // vision joint position measurement noise
    double sigma_dq_ = 0.1; // vision joint velocity measurement noise

    bool print_velocity_ = true; // whether to print joint velocities to console
    bool print_sigmas_ = true; // whether to print measurement and task sigmas to console
    bool first_dq_ = false;
    bool use_kalman_filter_ = true;
    bool fallback_to_latest_tf_ = true;
    double tf_lookup_timeout_sec_ = 0.05;

    std::vector<std::string> joint_names_ = {
        "upt_jRightShoulder_rotx",
        "upt_jRightShoulder_rotz",
        "upt_jRightShoulder_roty",
        "upt_jRightElbow_rotz",
        "upt_jRightElbow_roty",
        "upt_jRightWrist_rotx",
        "upt_jRightWrist_rotz"
    };

    double t_vis_ = 0.0, now_ = 0.0;

    // sensitivity analysis
    double sigma_sh_x = 0.0;
    double sigma_sh_y = 0.0;
    double sigma_sh_z = 0.0;
    bool apply_sensitivity = false;

    // -------------------- ROS interfaces --------------------
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_joint_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr updated_arm_joint_states_pub_;

    // math utilities
    pinocchio::Model model_, arm_model_;
    pinocchio::Data data_, arm_data_;
    std::string ee_frame_name_, hand_frame_name_, shoulder_frame_name_;
    pinocchio::FrameIndex ee_frame_id_, hand_frame_id_;
    pinocchio::SE3 T_eehand_, T_handee_, T_worldee_,T_worldhand_,T_shoulderhand_,T_worldshoulder_;
    Eigen::VectorXd q_arm_, dq_arm_, q_, dq_, dq_arm_prev_;
    Eigen::VectorXd v_ee_in_world_, v_hand_in_world_, v_hand_in_shoulder_;
    Eigen::Matrix<double, 6, 6> Ad_T;

    // Kalman filter instance and vision dq
    DQKalman kf_;
    Eigen::VectorXd dq_vis_;
    rclcpp::Time last_vis_stamp_;
    rclcpp::Time last_robot_stamp_;
    bool first_vis_{false};
    bool first_robot_{false};
    bool new_vis_measurement_{false};

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    geometry_msgs::msg::TransformStamped tf_world2shoulder;
};
