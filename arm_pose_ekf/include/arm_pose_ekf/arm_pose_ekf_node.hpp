#pragma once

// =============================================================================
// arm_pose_ekf
// -----------------------------------------------------------------------------
// Extended Kalman filter that estimates the 7-DOF right-arm joint state of a
// human subject by fusing:
//   1. The robot end-effector pose/twist (the robot rigidly holds the hand, so
//      its measured EE pose is a high-rate observation of the hand pose).
//   2. Camera keypoint distributions (shoulder/elbow/wrist 3D Gaussians) used
//      as bone-DIRECTION measurements to resolve the arm's redundant DOFs.
//
// State (see Ekf):  x = [q (7), dq (7)]  -> kStateSize = 14
//   q = [elv_angle, shoulder_elv, shoulder_rot,
//        elbow_flexion, forearm_prosup, wrist_x, wrist_z]
//
// PIPELINE (single-threaded executor: all callbacks are serialized)
//   robotJointCallback is the MAIN TRIGGER and drives one full cycle per robot
//   joint-state message. The keypoint callbacks are passive: they only transform
//   and cache the latest vision measurement, consumed during the main trigger.
//
//     robotJointCallback (MAIN TRIGGER)            keypointCallback (passive)
//       -> predictTo                                 -> transform to shoulder
//       -> correctHandPoseAndTwist                   -> cache PointMeasurement
//       -> correctKeypointMeasurements (uses cache)
//       -> publishState
//
// The .cpp is organized into clearly banner-separated sections:
//   PARAMETER LOADING / SETUP / DATA ACQUISITION / DATA PROCESSING /
//   OUTPUT & VISUALIZATION.
//
// MATH / RESIDUALS (see the .cpp correction methods for per-measurement detail)
//   predict : pure constant-velocity model + white-noise acceleration.
//             q_{k+1} = q_k + dt*dq_k,  dq_{k+1} = dq_k,  P = F P F^T + Q.
//   correct : standard EKF innovation update, x += K r, Joseph-form P update.
//             Measurements applied sequentially (FK re-linearized between them):
//     - hand position : r = p_measured - p_model(q),       H = [J_v, 0]
//     - hand rotation : r = log3(R_measured R_model^T),     H = [J_w, 0]
//     - hand twist    : r = v_measured - J_hand(q) dq,      H = [0, J_hand]
//     - bone direction: r = u_measured - u_model(q),        H = [J_u, 0]
//                       where u = (p_end - p_start)/||.|| (unit bone vector)
//   constraints : URDF box limits enforced by Gaussian PDF truncation
//                 (Simon & Simon 2010), updating both the mean and covariance.
// =============================================================================

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

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

    // ------------------------------------------------------------------------
    // EKF core (state-only; no ROS, no kinematics). State x = [q (7), dq (7)].
    // ------------------------------------------------------------------------
    struct Ekf
    {
        Eigen::Matrix<double, kStateSize, 1> x = Eigen::Matrix<double, kStateSize, 1>::Zero();
        Eigen::Matrix<double, kStateSize, kStateSize> P = Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
        Eigen::Matrix<double, kStateSize, kStateSize> I = Eigen::Matrix<double, kStateSize, kStateSize>::Identity();

        void init(const Eigen::Matrix<double, kDof, 1>& q0,
                  const Eigen::Matrix<double, kDof, 1>& dq0,
                  double q_variance,
                  double dq_variance);
        // Constant-velocity prediction with white-noise-acceleration process
        // covariance scaled by acceleration_variance.
        void predict(double dt, double acceleration_variance);
        void correct(const Eigen::VectorXd& residual,
                     const Eigen::MatrixXd& H,
                     const Eigen::MatrixXd& R);
        Eigen::Matrix<double, kDof, 1> q() const;
        Eigen::Matrix<double, kDof, 1> dq() const;
    };

    // Cached camera keypoint Gaussian, expressed in the shoulder frame.
    struct PointMeasurement
    {
        Eigen::Vector3d mean = Eigen::Vector3d::Zero();
        Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
        rclcpp::Time stamp;
        bool available = false;
    };

    // --- Parameter loading & setup ------------------------------------------
    void declareParameters();
    void loadParameters();

    // --- Data acquisition: ROS subscription callbacks -----------------------
    // robotJointCallback is the MAIN TRIGGER for the whole filter cycle.
    void robotJointCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void keypointCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg,
                          const std::string& keypoint_name);
    // Acquisition helpers: turn raw messages / TF lookups into shoulder-frame
    // quantities the corrections consume.
    bool transformPointMeasurementToShoulder(
        const geometry_msgs::msg::PoseWithCovarianceStamped& msg,
        PointMeasurement& measurement) const;
    pinocchio::SE3 robotEndEffectorPose(const sensor_msgs::msg::JointState& msg,
                                        Eigen::Matrix<double, 6, 1>& ee_twist_world);
    pinocchio::SE3 lookupShoulderToEndEffector(rclcpp::Time& tf_stamp) const;
    pinocchio::SE3 lookupWorldToShoulder(const rclcpp::Time& stamp) const;
    static Eigen::Matrix3d covarianceFromMessage(const geometry_msgs::msg::PoseWithCovarianceStamped& msg);

    // --- Data processing: EKF predict / correct / constrain -----------------
    void predictTo(const rclcpp::Time& stamp);
    void correctHandPoseAndTwist(const sensor_msgs::msg::JointState& robot_msg);
    void correctKeypointMeasurements();
    // Soft pseudo-measurement that holds the vision-observed proximal joints
    // (q0-q3) at their last vision-confident value while the keypoint update is
    // unavailable (e.g. the elbow is occluded), so the 6-DOF hand pose cannot
    // drag the proximal null space. No-op until a valid anchor has been cached.
    void applyProximalAnchor();
    bool keypointSegmentsTooClose(const rclcpp::Time& stamp) const;
    void loadStateConstraintsFromArmModel();
    void applyStateConstraints(const char* source);
    // Covariance-aware projection of one state component onto [lower, upper] via
    // Gaussian PDF truncation (updates both ekf_.x and ekf_.P).
    void truncateStateToInterval(int index, double lower, double upper);

    // --- Data processing: kinematics / Jacobian helpers ---------------------
    // Map the 7 EKF (independent) coordinates into the full Pinocchio model q,
    // including the mimic shoulder joint, then reduce model Jacobians back to
    // the 7 independent columns.
    Eigen::VectorXd modelConfigurationFromOpt(const double* values) const;
    Eigen::VectorXd modelConfigurationFromOpt(const Eigen::Matrix<double, kDof, 1>& values) const;
    Eigen::Matrix<double, 6, Eigen::Dynamic> independentJacobian(
        const Eigen::Matrix<double, 6, Eigen::Dynamic>& full_jacobian) const;
    Eigen::Matrix<double, 3, kDof> frameLinearJacobian(pinocchio::FrameIndex frame_id);
    Eigen::Matrix<double, 6, kDof> handJacobian();
    void updateArmKinematics(const Eigen::Matrix<double, kDof, 1>& q);

    // --- Output & visualization ---------------------------------------------
    void publishState(const rclcpp::Time& stamp);
    void publishGroundTruthHandTransform(const pinocchio::SE3& T_shoulder_hand,
                                         const rclcpp::Time& stamp) const;

    // --- Frames / model ------------------------------------------------------
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

    // --- Filter state --------------------------------------------------------
    Ekf ekf_;
    bool initialized_ = false;
    double last_predict_dt_ = 1.0 / 250.0;
    rclcpp::Time last_predict_stamp_;
    rclcpp::Time last_processed_robot_stamp_;
    std::map<std::string, PointMeasurement> keypoint_measurements_;

    // --- Parameters: setup-only (read during construction) ------------------
    std::string arm_xacro_file_;
    std::string subject_arm_lengths_file_;
    int subject_id_ = 1;
    double fallback_upper_arm_length_ = 0.35;
    double fallback_forearm_length_ = 0.26;
    double fallback_hand_length_ = 0.0918;
    std::vector<double> initial_q_param_;
    std::vector<double> initial_dq_param_;
    std::string robot_joint_topic_;
    std::string output_topic_;

    // --- Parameters: state / process ----------------------------------------
    double initial_q_variance_ = 1.0;
    double initial_dq_variance_ = 1.0;
    double process_acceleration_variance_ = 1.0e-3;

    // --- Parameters: hand-pose measurement ----------------------------------
    bool use_hand_pose_ = true;
    bool use_hand_pose_position_ = true;
    bool use_hand_pose_rotation_ = false;
    double hand_position_sigma_ = 0.01;
    double hand_rotation_sigma_ = 0.05;
    // Per-axis (x, y, z in HAND-frame axes) multipliers on hand_rotation_sigma_.
    // Models grip compliance: a large ratio inflates that axis' measurement
    // noise so EE rotation about it barely corrects the state. [1,1,1] = isotropic.
    Eigen::Vector3d hand_rotation_axis_ratio_ = Eigen::Vector3d::Ones();

    // --- Parameters: hand-twist measurement ---------------------------------
    bool use_hand_twist_ = true;
    double hand_linear_velocity_sigma_ = 0.03;
    double hand_angular_velocity_sigma_ = 0.1;

    // --- Parameters: keypoint (vision) measurement --------------------------
    bool use_keypoints_ = true;
    double keypoint_direction_sigma_ = 0.05;
    double max_keypoint_age_sec_ = 0.2;
    bool reject_keypoints_when_segments_too_close_ = true;
    double min_keypoint_segment_length_ = 0.15;
    // Occlusion-gated proximal anchor: when no keypoint update runs, hold q0-q3
    // at the last vision-confident value via a tight pseudo-measurement.
    bool use_proximal_anchor_ = true;
    double proximal_anchor_sigma_ = 0.02;
    Eigen::Matrix<double, 4, 1> proximal_anchor_ = Eigen::Matrix<double, 4, 1>::Zero();
    bool proximal_anchor_valid_ = false;

    // --- Parameters: constraints --------------------------------------------
    bool enforce_joint_limits_ = true;
    bool zero_velocity_at_limits_ = true;

    // --- Parameters: timing / processing ------------------------------------
    bool use_latest_shoulder_tf_ = true;
    bool fallback_to_latest_tf_ = true;
    double tf_lookup_timeout_sec_ = 0.05;
    double max_update_rate_hz_ = 100.0;
    int robot_joint_queue_depth_ = 1;
    int keypoint_queue_depth_ = 1;

    // --- ROS interfaces ------------------------------------------------------
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr robot_joint_sub_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr> keypoint_subs_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
