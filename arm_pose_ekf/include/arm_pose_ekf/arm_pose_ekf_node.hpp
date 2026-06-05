#pragma once

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Dense>

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

class ArmPoseEkfNode : public rclcpp::Node
{
public:
    ArmPoseEkfNode();

private:
    static constexpr int kDof = 7;
    static constexpr int kStateSize = 14;

    struct Ekf
    {
        Eigen::Matrix<double, kStateSize, 1> x = Eigen::Matrix<double, kStateSize, 1>::Zero();
        Eigen::Matrix<double, kStateSize, kStateSize> P = Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
        Eigen::Matrix<double, kStateSize, kStateSize> I = Eigen::Matrix<double, kStateSize, kStateSize>::Identity();

        void init(const Eigen::Matrix<double, kDof, 1>& q0,
                  const Eigen::Matrix<double, kDof, 1>& dq0,
                  double q_variance,
                  double dq_variance);
        void predict(double dt, double acceleration_variance, double velocity_decay);
        void correct(const Eigen::VectorXd& residual,
                     const Eigen::MatrixXd& H,
                     const Eigen::MatrixXd& R);
        double normalizedInnovationSquared(const Eigen::VectorXd& residual,
                                           const Eigen::MatrixXd& H,
                                           const Eigen::MatrixXd& R) const;
        Eigen::Matrix<double, kDof, 1> q() const;
        Eigen::Matrix<double, kDof, 1> dq() const;
    };

    struct PointMeasurement
    {
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
        rclcpp::Time stamp;
        bool available = false;
    };

    void robotJointCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void keypointCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg,
                          const std::string& keypoint_name);

    void predictTo(const rclcpp::Time& stamp);
    void correctHandPoseAndTwist(const sensor_msgs::msg::JointState& robot_msg);
    void correctKeypointMeasurements();
    bool keypointSegmentsTooClose(const rclcpp::Time& stamp) const;
    void publishState(const rclcpp::Time& stamp);
    void publishKeypointDebugMarkers(const rclcpp::Time& stamp);
    void loadStateConstraintsFromArmModel();
    void applyStateConstraints(const char* source);

    Eigen::VectorXd modelConfigurationFromOpt(const double* values) const;
    Eigen::VectorXd modelConfigurationFromOpt(const Eigen::Matrix<double, kDof, 1>& values) const;
    Eigen::Matrix<double, 6, Eigen::Dynamic> independentJacobian(
        const Eigen::Matrix<double, 6, Eigen::Dynamic>& full_jacobian) const;
    Eigen::Matrix<double, 3, kDof> frameLinearJacobian(pinocchio::FrameIndex frame_id);
    Eigen::Matrix<double, 6, kDof> handJacobian();

    bool transformPointMeasurementToShoulder(
        const geometry_msgs::msg::PoseWithCovarianceStamped& msg,
        PointMeasurement& measurement) const;

    pinocchio::SE3 robotEndEffectorPose(const sensor_msgs::msg::JointState& msg,
                                        Eigen::Matrix<double, 6, 1>& ee_twist_world);
    pinocchio::SE3 lookupShoulderToEndEffector(rclcpp::Time& tf_stamp) const;
    pinocchio::SE3 lookupWorldToShoulder(const rclcpp::Time& stamp) const;
    void publishGroundTruthHandTransform(const pinocchio::SE3& T_shoulder_hand,
                                         const rclcpp::Time& stamp) const;
    void updateArmKinematics(const Eigen::Matrix<double, kDof, 1>& q);
    static Eigen::Matrix3d covarianceFromMessage(const geometry_msgs::msg::PoseWithCovarianceStamped& msg);
    static bool isFinite(const Eigen::MatrixXd& matrix);

    std::string robot_prefix_;
    std::string robot_color_;
    std::string world_frame_;
    std::string shoulder_frame_name_;
    std::string hand_frame_name_;
    std::string ee_frame_name_;

    pinocchio::Model robot_model_;
    pinocchio::Data robot_data_;
    pinocchio::FrameIndex robot_ee_frame_id_;

    pinocchio::Model arm_model_;
    pinocchio::Data arm_data_;
    std::array<pinocchio::JointIndex, kDof> arm_joint_ids_;
    std::array<double, kDof> arm_joint_multipliers_;
    Eigen::Matrix<double, kDof, 1> q_lower_limits_ =
        Eigen::Matrix<double, kDof, 1>::Constant(-std::numeric_limits<double>::infinity());
    Eigen::Matrix<double, kDof, 1> q_upper_limits_ =
        Eigen::Matrix<double, kDof, 1>::Constant(std::numeric_limits<double>::infinity());
    Eigen::Matrix<double, kDof, 1> dq_velocity_limits_ =
        Eigen::Matrix<double, kDof, 1>::Constant(std::numeric_limits<double>::infinity());
    pinocchio::JointIndex mimic_joint_id_;
    double mimic_joint_multiplier_ = -1.0;
    pinocchio::FrameIndex shoulder_frame_id_;
    pinocchio::FrameIndex elbow_frame_id_;
    pinocchio::FrameIndex wrist_frame_id_;
    pinocchio::FrameIndex hand_frame_id_;
    std::vector<std::string> joint_names_;

    pinocchio::SE3 T_ee_hand_ = pinocchio::SE3::Identity();

    Ekf ekf_;
    bool initialized_ = false;
    rclcpp::Time last_predict_stamp_;
    rclcpp::Time last_processed_robot_stamp_;

    std::map<std::string, PointMeasurement> keypoint_measurements_;

    double initial_q_variance_ = 1.0;
    double initial_dq_variance_ = 1.0;
    double process_acceleration_variance_ = 1.0e-3;
    double velocity_decay_ = 1.0;
    double hand_position_sigma_ = 0.01;
    double hand_rotation_sigma_ = 0.05;
    double hand_linear_velocity_sigma_ = 0.03;
    double hand_angular_velocity_sigma_ = 0.1;
    double default_keypoint_sigma_ = 0.05;
    double keypoint_direction_sigma_ = 0.05;
    double max_keypoint_age_sec_ = 0.2;
    double min_keypoint_segment_length_ = 0.15;
    double keypoint_nis_threshold_ = 11.34;
    bool use_hand_pose_ = true;
    bool use_hand_pose_position_ = true;
    bool use_hand_pose_rotation_ = false;
    bool use_hand_twist_ = true;
    bool use_keypoints_ = true;
    bool use_keypoint_gating_ = true;
    bool reject_keypoints_when_segments_too_close_ = true;
    bool enforce_joint_limits_ = true;
    bool zero_velocity_at_limits_ = true;
    bool use_latest_shoulder_tf_ = true;
    bool fallback_to_latest_tf_ = true;
    double tf_lookup_timeout_sec_ = 0.05;
    double max_update_rate_hz_ = 100.0;
    bool publish_keypoint_debug_markers_ = false;
    double keypoint_debug_marker_scale_ = 2.0;
    double keypoint_debug_marker_min_radius_ = 0.01;
    double keypoint_debug_marker_max_radius_ = 0.30;
    int robot_joint_queue_depth_ = 1;
    int keypoint_queue_depth_ = 1;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr robot_joint_sub_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr> keypoint_subs_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr keypoint_debug_marker_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
