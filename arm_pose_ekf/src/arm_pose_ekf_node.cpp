#include "arm_pose_ekf/arm_pose_ekf_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <yaml-cpp/yaml.h>

#include <tf2/exceptions.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

// =============================================================================
// FILE-SCOPE HELPERS (no node state)
// Path resolution, subject arm-length loading, xacro rendering, parameter->Eigen
// converters, and the scalar core of the PDF-truncation constraint.
// =============================================================================
namespace {

struct ArmLengths
{
    double upper_arm_length;
    double forearm_length;
    double hand_length;
    // Per-subject grip-compliance rotation [x, y, z] in RADIANS (converted from
    // the degrees stored in the YAML). Applied to the ee->hand rotation only,
    // not to the rendered xacro. Defaults to no rotation.
    Eigen::Vector3d grip_offset_rad = Eigen::Vector3d::Zero();
};

std::string resolve_config_path(const std::string& file)
{
    if (!file.empty() && file.front() == '/') {
        return file;
    }
    return ament_index_cpp::get_package_share_directory("image_pose_tracking") +
        "/config/" + file;
}

ArmLengths load_subject_arm_lengths(
    const std::string& arm_lengths_file,
    int subject_id,
    const ArmLengths& fallback)
{
    const std::string yaml_path = resolve_config_path(arm_lengths_file);
    const YAML::Node root = YAML::LoadFile(yaml_path);
    const std::string subject_key = "subject_" + std::to_string(subject_id);
    const YAML::Node subject = root["subjects"][subject_key];
    if (!subject) {
        throw std::runtime_error(
            "Subject arm lengths not found for " + subject_key + " in " + yaml_path);
    }

    ArmLengths lengths = fallback;
    if (subject["upper_arm_length"]) {
        lengths.upper_arm_length = subject["upper_arm_length"].as<double>();
    }
    if (subject["forearm_length"]) {
        lengths.forearm_length = subject["forearm_length"].as<double>();
    }
    if (subject["hand_length"]) {
        lengths.hand_length = subject["hand_length"].as<double>();
    }
    if (subject["grip_offset"]) {
        const auto offset = subject["grip_offset"].as<std::vector<double>>();
        if (offset.size() != 3) {
            throw std::runtime_error(
                "grip_offset must have 3 entries [x, y, z] for " + subject_key + " in " + yaml_path);
        }
        constexpr double kDegToRad = M_PI / 180.0;
        lengths.grip_offset_rad =
            Eigen::Vector3d(offset[0], offset[1], offset[2]) * kDegToRad;
    }
    if (!(lengths.upper_arm_length > 0.0) || !(lengths.forearm_length > 0.0) ||
        !(lengths.hand_length > 0.0)) {
        throw std::runtime_error(
            "Subject arm lengths must be positive for " + subject_key + " in " + yaml_path);
    }
    return lengths;
}

std::string render_arm_xacro(
    const std::string& robot_prefix,
    const std::string& robot_color,
    const ArmLengths& lengths,
    const std::string& arm_xacro_file)
{
    const std::string xacro_path = resolve_config_path(arm_xacro_file);

    char temp_template[] = "/tmp/arm_pose_ekf_osim_XXXXXX.urdf";
    const int fd = mkstemps(temp_template, 5);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary arm URDF path");
    }
    std::fclose(fdopen(fd, "w"));

    const std::string command =
        "xacro " + xacro_path +
        " robot_prefix:=" + robot_prefix +
        " robot_color:=" + robot_color +
        " upper_arm_length:=" + std::to_string(lengths.upper_arm_length) +
        " forearm_length:=" + std::to_string(lengths.forearm_length) +
        " hand_length:=" + std::to_string(lengths.hand_length) +
        " > " + temp_template;

    if (std::system(command.c_str()) != 0) {
        std::remove(temp_template);
        throw std::runtime_error("Failed to render arm xacro: " + xacro_path);
    }

    return temp_template;
}

Eigen::Vector3d parameterVector3(
    const rclcpp::Node& node,
    const std::string& name,
    const std::vector<double>& fallback)
{
    const auto values = node.get_parameter(name).as_double_array();
    const auto& source = values.size() == 3 ? values : fallback;
    return Eigen::Vector3d(source[0], source[1], source[2]);
}

Eigen::Vector4d parameterVector4(
    const rclcpp::Node& node,
    const std::string& name,
    const std::vector<double>& fallback)
{
    const auto values = node.get_parameter(name).as_double_array();
    const auto& source = values.size() == 4 ? values : fallback;
    return Eigen::Vector4d(source[0], source[1], source[2], source[3]);
}

struct TruncatedMoments
{
    double mean;      // E[z]   of a standard normal restricted to [a, b]
    double variance;  // Var[z] of a standard normal restricted to [a, b]
};

// Mean and variance of a STANDARD normal N(0,1) truncated to the interval
// [a, b]; a may be -inf and b may be +inf. Scalar core of the Simon & Simon
// (2010) constrained-Kalman PDF-truncation method. When the feasible interval
// lies deep in the tail (numerically zero mass), it degrades gracefully to
// projecting the mean onto the nearest bound with zero residual variance.
TruncatedMoments standardTruncatedMoments(double a, double b)
{
    constexpr double kInvSqrt2Pi = 0.3989422804014327;
    constexpr double kInvSqrt2 = 0.7071067811865476;
    const auto pdf = [](double z) {
        return std::isfinite(z) ? kInvSqrt2Pi * std::exp(-0.5 * z * z) : 0.0;
    };
    const auto cdf = [&](double z) {
        if (z == std::numeric_limits<double>::infinity()) return 1.0;
        if (z == -std::numeric_limits<double>::infinity()) return 0.0;
        return 0.5 * std::erfc(-z * kInvSqrt2);
    };

    const double mass = cdf(b) - cdf(a);
    if (!(mass > 1.0e-12)) {
        return {std::clamp(0.0, a, b), 0.0};
    }

    const double pa = pdf(a);
    const double pb = pdf(b);
    const double mean = (pa - pb) / mass;
    const double term_a = std::isfinite(a) ? a * pa : 0.0;
    const double term_b = std::isfinite(b) ? b * pb : 0.0;
    double variance = 1.0 + (term_a - term_b) / mass - mean * mean;
    variance = std::clamp(variance, 0.0, 1.0);
    return {mean, variance};
}

}  // namespace

// =============================================================================
// EKF CORE MATH (state-only; no ROS, no kinematics)
// -----------------------------------------------------------------------------
// State x = [q (7), dq (7)], covariance P (14x14).
//
// predict(): pure constant-velocity model.
//   q_{k+1}  = q_k + dt * dq_k
//   dq_{k+1} = dq_k
//   F = d(x_{k+1})/d(x_k); P = F P F^T + Q, with Q the discretized white-noise
//   acceleration covariance ([dt^3/3, dt^2/2; dt^2/2, dt] * accel_variance).
//
// correct(): standard linearized EKF innovation update for residual r and
//   measurement Jacobian H (built by the caller per measurement):
//     S = H P H^T + R          (innovation covariance)
//     K = P H^T S^{-1}         (Kalman gain, solved via LDLT)
//     x = x + K r
//     P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form, stays PSD)
//   Measurements are applied SEQUENTIALLY (FK re-linearized between steps),
//   equivalent to one stacked update for independent R blocks but letting later
//   corrections see the updated q.
// =============================================================================

void ArmPoseEkfNode::Ekf::init(
    const Eigen::Matrix<double, kDof, 1>& q0,
    const Eigen::Matrix<double, kDof, 1>& dq0,
    double q_variance,
    double dq_variance)
{
    x.head<kDof>() = q0;
    x.tail<kDof>() = dq0;
    P.setZero();
    P.block<kDof, kDof>(0, 0).setIdentity();
    P.block<kDof, kDof>(0, 0) *= q_variance;
    P.block<kDof, kDof>(kDof, kDof).setIdentity();
    P.block<kDof, kDof>(kDof, kDof) *= dq_variance;
}

void ArmPoseEkfNode::Ekf::predict(double dt, double acceleration_variance)
{
    const double step = std::max(dt, 1.0e-4);

    // Constant-velocity transition: q += dt * dq, dq unchanged.
    Eigen::Matrix<double, kStateSize, kStateSize> F =
        Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
    F.block<kDof, kDof>(0, kDof) = step * Eigen::Matrix<double, kDof, kDof>::Identity();

    x.head<kDof>() += step * x.tail<kDof>();

    // Discretized white-noise-acceleration process covariance.
    const double dt2 = step * step;
    const double dt3 = dt2 * step;
    Eigen::Matrix<double, kStateSize, kStateSize> Q =
        Eigen::Matrix<double, kStateSize, kStateSize>::Zero();
    Q.block<kDof, kDof>(0, 0).setIdentity();
    Q.block<kDof, kDof>(0, 0) *= (dt3 / 3.0) * acceleration_variance;
    Q.block<kDof, kDof>(0, kDof).setIdentity();
    Q.block<kDof, kDof>(0, kDof) *= (dt2 / 2.0) * acceleration_variance;
    Q.block<kDof, kDof>(kDof, 0).setIdentity();
    Q.block<kDof, kDof>(kDof, 0) *= (dt2 / 2.0) * acceleration_variance;
    Q.block<kDof, kDof>(kDof, kDof).setIdentity();
    Q.block<kDof, kDof>(kDof, kDof) *= step * acceleration_variance;

    P = F * P * F.transpose() + Q;
    P = 0.5 * (P + P.transpose());
}

void ArmPoseEkfNode::Ekf::correct(
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& R)
{
    if (residual.size() == 0) {
        return;
    }

    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(
        Eigen::MatrixXd::Identity(S.rows(), S.cols()));
    x += K * residual;
    const Eigen::MatrixXd IKH = I - K * H;
    P = IKH * P * IKH.transpose() + K * R * K.transpose();
    P = 0.5 * (P + P.transpose());
}

Eigen::Matrix<double, ArmPoseEkfNode::kDof, 1> ArmPoseEkfNode::Ekf::q() const
{
    return x.head<kDof>();
}

Eigen::Matrix<double, ArmPoseEkfNode::kDof, 1> ArmPoseEkfNode::Ekf::dq() const
{
    return x.tail<kDof>();
}

// =============================================================================
// PARAMETER LOADING
// =============================================================================
void ArmPoseEkfNode::declareParameters()
{
    // Frames / model
    this->declare_parameter<std::string>("robot_prefix", "upt_");
    this->declare_parameter<std::string>("robot_color", "blue");
    this->declare_parameter<std::string>("world_frame", "lbr_link_0");
    this->declare_parameter<std::string>("shoulder_frame", "upt_RightShoulder");
    this->declare_parameter<std::string>("hand_frame", "upt_RightHandCOM");
    this->declare_parameter<std::string>("robot_ee_frame", "lbr_link_ee");
    this->declare_parameter<std::string>("arm_xacro_file", "right_arm_osim_shoulder_mesh.urdf.xacro");
    this->declare_parameter<std::string>("subject_arm_lengths_file", "subject_arm_lengths.yaml");
    this->declare_parameter<int>("subject_id", 1);
    this->declare_parameter<double>("upper_arm_length", 0.35);
    this->declare_parameter<double>("forearm_length", 0.26);
    this->declare_parameter<double>("hand_length", 0.0918);
    this->declare_parameter<std::vector<double>>("ee_to_hand.translation", {0.0, -0.03, 0.08});
    this->declare_parameter<std::vector<double>>(
        "ee_to_hand.rotation", {0.2988362387, 0.6408563821, -0.2988362387, 0.6408563821});

    // State / process
    this->declare_parameter<std::vector<double>>("state.initial_q", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    this->declare_parameter<std::vector<double>>("state.initial_dq", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    this->declare_parameter<double>("state.initial_q_variance", 1.0);
    this->declare_parameter<double>("state.initial_dq_variance", 1.0);
    this->declare_parameter<double>("process_noise.acceleration_variance", 1.0e-3);

    // Processing
    this->declare_parameter<double>("processing.max_update_rate_hz", 100.0);
    this->declare_parameter<int>("processing.robot_joint_queue_depth", 1);
    this->declare_parameter<int>("processing.keypoint_queue_depth", 1);

    // Constraints
    this->declare_parameter<bool>("constraints.enforce_joint_limits", true);
    this->declare_parameter<bool>("constraints.zero_velocity_at_limits", true);

    // Hand-pose measurement
    this->declare_parameter<bool>("measurements.hand_pose.enabled", true);
    this->declare_parameter<bool>("measurements.hand_pose.use_position", true);
    this->declare_parameter<bool>("measurements.hand_pose.use_rotation", false);
    this->declare_parameter<double>("measurements.hand_pose.sigma_position", 0.01);
    this->declare_parameter<double>("measurements.hand_pose.sigma_rotation", 0.05);
    this->declare_parameter<std::vector<double>>(
        "measurements.hand_pose.rotation_axis_ratio", {1.0, 1.0, 1.0});

    // Hand-twist measurement
    this->declare_parameter<bool>("measurements.hand_twist.enabled", true);
    this->declare_parameter<double>("measurements.hand_twist.sigma_linear_velocity", 0.03);
    this->declare_parameter<double>("measurements.hand_twist.sigma_angular_velocity", 0.1);

    // Keypoint (vision) measurement
    this->declare_parameter<bool>("measurements.keypoint_distributions.enabled", true);
    this->declare_parameter<double>("measurements.keypoint_distributions.direction_sigma", 0.05);
    this->declare_parameter<double>("measurements.keypoint_distributions.max_age_sec", 0.2);
    this->declare_parameter<bool>("measurements.keypoint_distributions.reject_when_segments_too_close", true);
    this->declare_parameter<double>("measurements.keypoint_distributions.min_segment_length", 0.15);
    this->declare_parameter<bool>("measurements.keypoint_distributions.proximal_anchor_enabled", true);
    this->declare_parameter<double>("measurements.keypoint_distributions.proximal_anchor_sigma", 0.02);

    // Timing
    this->declare_parameter<double>("timing.tf_lookup_timeout_sec", 0.05);
    this->declare_parameter<bool>("timing.use_latest_shoulder_tf", true);
    this->declare_parameter<bool>("timing.fallback_to_latest_tf", true);

    // Topics
    this->declare_parameter<std::string>("topics.robot_joint_states", "/lbr/joint_states");
    this->declare_parameter<std::string>("topics.output_joint_states", "/arm_pose_ekf/joint_states");
}

void ArmPoseEkfNode::loadParameters()
{
    robot_prefix_ = this->get_parameter("robot_prefix").as_string();
    robot_color_ = this->get_parameter("robot_color").as_string();
    world_frame_ = this->get_parameter("world_frame").as_string();
    shoulder_frame_name_ = this->get_parameter("shoulder_frame").as_string();
    hand_frame_name_ = this->get_parameter("hand_frame").as_string();
    ee_frame_name_ = this->get_parameter("robot_ee_frame").as_string();
    arm_xacro_file_ = this->get_parameter("arm_xacro_file").as_string();
    subject_arm_lengths_file_ = this->get_parameter("subject_arm_lengths_file").as_string();
    subject_id_ = std::max<int64_t>(1, this->get_parameter("subject_id").as_int());
    fallback_upper_arm_length_ = this->get_parameter("upper_arm_length").as_double();
    fallback_forearm_length_ = this->get_parameter("forearm_length").as_double();
    fallback_hand_length_ = this->get_parameter("hand_length").as_double();

    initial_q_param_ = this->get_parameter("state.initial_q").as_double_array();
    initial_dq_param_ = this->get_parameter("state.initial_dq").as_double_array();
    initial_q_variance_ = this->get_parameter("state.initial_q_variance").as_double();
    initial_dq_variance_ = this->get_parameter("state.initial_dq_variance").as_double();
    process_acceleration_variance_ = this->get_parameter("process_noise.acceleration_variance").as_double();

    max_update_rate_hz_ = this->get_parameter("processing.max_update_rate_hz").as_double();
    robot_joint_queue_depth_ =
        std::max(1, static_cast<int>(this->get_parameter("processing.robot_joint_queue_depth").as_int()));
    keypoint_queue_depth_ =
        std::max(1, static_cast<int>(this->get_parameter("processing.keypoint_queue_depth").as_int()));

    enforce_joint_limits_ = this->get_parameter("constraints.enforce_joint_limits").as_bool();
    zero_velocity_at_limits_ = this->get_parameter("constraints.zero_velocity_at_limits").as_bool();

    use_hand_pose_ = this->get_parameter("measurements.hand_pose.enabled").as_bool();
    use_hand_pose_position_ = this->get_parameter("measurements.hand_pose.use_position").as_bool();
    use_hand_pose_rotation_ = this->get_parameter("measurements.hand_pose.use_rotation").as_bool();
    hand_position_sigma_ = this->get_parameter("measurements.hand_pose.sigma_position").as_double();
    hand_rotation_sigma_ = this->get_parameter("measurements.hand_pose.sigma_rotation").as_double();
    hand_rotation_axis_ratio_ = parameterVector3(
        *this, "measurements.hand_pose.rotation_axis_ratio", {1.0, 1.0, 1.0});

    use_hand_twist_ = this->get_parameter("measurements.hand_twist.enabled").as_bool();
    hand_linear_velocity_sigma_ = this->get_parameter("measurements.hand_twist.sigma_linear_velocity").as_double();
    hand_angular_velocity_sigma_ = this->get_parameter("measurements.hand_twist.sigma_angular_velocity").as_double();

    use_keypoints_ = this->get_parameter("measurements.keypoint_distributions.enabled").as_bool();
    keypoint_direction_sigma_ = this->get_parameter("measurements.keypoint_distributions.direction_sigma").as_double();
    max_keypoint_age_sec_ = this->get_parameter("measurements.keypoint_distributions.max_age_sec").as_double();
    reject_keypoints_when_segments_too_close_ =
        this->get_parameter("measurements.keypoint_distributions.reject_when_segments_too_close").as_bool();
    min_keypoint_segment_length_ =
        this->get_parameter("measurements.keypoint_distributions.min_segment_length").as_double();
    use_proximal_anchor_ =
        this->get_parameter("measurements.keypoint_distributions.proximal_anchor_enabled").as_bool();
    proximal_anchor_sigma_ =
        this->get_parameter("measurements.keypoint_distributions.proximal_anchor_sigma").as_double();

    tf_lookup_timeout_sec_ = this->get_parameter("timing.tf_lookup_timeout_sec").as_double();
    use_latest_shoulder_tf_ = this->get_parameter("timing.use_latest_shoulder_tf").as_bool();
    fallback_to_latest_tf_ = this->get_parameter("timing.fallback_to_latest_tf").as_bool();

    robot_joint_topic_ = this->get_parameter("topics.robot_joint_states").as_string();
    output_topic_ = this->get_parameter("topics.output_joint_states").as_string();

    // ee->hand fixed transform (quaternion stored x, y, z, w).
    const Eigen::Vector3d ee_hand_translation = parameterVector3(
        *this, "ee_to_hand.translation", {0.0, -0.03, 0.08});
    const Eigen::Vector4d ee_hand_rotation = parameterVector4(
        *this, "ee_to_hand.rotation", {0.2988362387, 0.6408563821, -0.2988362387, 0.6408563821});
    const Eigen::Quaterniond q_ee_hand(
        ee_hand_rotation[3], ee_hand_rotation[0], ee_hand_rotation[1], ee_hand_rotation[2]);
    T_ee_hand_.translation() = ee_hand_translation;
    T_ee_hand_.rotation() = q_ee_hand.normalized().toRotationMatrix();
}

// =============================================================================
// SETUP
// Loads parameters, the robot + arm Pinocchio models, resolves joint/frame ids,
// initializes the EKF state, and wires up ROS pub/sub/TF.
// =============================================================================
ArmPoseEkfNode::ArmPoseEkfNode()
    : rclcpp::Node("arm_pose_ekf")
{
    declareParameters();
    loadParameters();

    // Robot (iiwa) model for the end-effector forward kinematics.
    const std::string robot_urdf =
        ament_index_cpp::get_package_share_directory("lbr_description") + "/urdf/iiwa7/iiwa7.urdf";
    pinocchio::urdf::buildModel(robot_urdf, robot_model_);
    robot_data_ = pinocchio::Data(robot_model_);
    robot_ee_frame_id_ = robot_model_.getFrameId(ee_frame_name_);

    // Subject-scaled human arm model, rendered from the xacro.
    const ArmLengths arm_lengths = load_subject_arm_lengths(
        subject_arm_lengths_file_,
        subject_id_,
        {fallback_upper_arm_length_, fallback_forearm_length_, fallback_hand_length_});
    const std::string arm_urdf =
        render_arm_xacro(robot_prefix_, robot_color_, arm_lengths, arm_xacro_file_);
    pinocchio::urdf::buildModel(arm_urdf, arm_model_);
    std::remove(arm_urdf.c_str());
    arm_data_ = pinocchio::Data(arm_model_);

    // Apply the per-subject grip-compliance offset to the nominal ee->hand
    // rotation set by loadParameters(). The offset is intrinsic X->Y->Z Euler
    // angles about the HAND-frame axes, so it right-multiplies the nominal
    // rotation: R_ee_hand <- R_ee_hand * Rx * Ry * Rz. Translation is unchanged;
    // a [0, 0, 0] offset is a no-op.
    const Eigen::Vector3d& grip = arm_lengths.grip_offset_rad;
    const Eigen::Matrix3d R_grip_offset =
        (Eigen::AngleAxisd(grip.x(), Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(grip.y(), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(grip.z(), Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    T_ee_hand_.rotation() = T_ee_hand_.rotation() * R_grip_offset;
    if (!grip.isZero()) {
        RCLCPP_INFO(
            this->get_logger(),
            "Applied grip_offset (deg) [%.3f, %.3f, %.3f] to ee_to_hand rotation.",
            grip.x() * 180.0 / M_PI, grip.y() * 180.0 / M_PI, grip.z() * 180.0 / M_PI);
    }

    joint_names_ = {
        robot_prefix_ + "jRightShoulder_elv_angle",
        robot_prefix_ + "jRightShoulder_shoulder_elv",
        robot_prefix_ + "jRightShoulder_shoulder_rot",
        robot_prefix_ + "jRightElbow_rotz",
        robot_prefix_ + "jRightElbow_roty",
        robot_prefix_ + "jRightWrist_rotx",
        robot_prefix_ + "jRightWrist_rotz",
    };
    for (int i = 0; i < kDof; ++i) {
        arm_joint_ids_[i] = arm_model_.getJointId(joint_names_[i]);
        arm_joint_multipliers_[i] = 1.0;
    }
    loadStateConstraintsFromArmModel();
    mimic_joint_id_ = arm_model_.getJointId(robot_prefix_ + "jRightShoulder_shoulder1_r2");

    shoulder_frame_id_ = arm_model_.getFrameId(robot_prefix_ + "RightShoulder");
    elbow_frame_id_ = arm_model_.getFrameId(robot_prefix_ + "RightForeArm");
    wrist_frame_id_ = arm_model_.getFrameId(robot_prefix_ + "RightForeArm_f1");
    hand_frame_id_ = arm_model_.getFrameId(hand_frame_name_);

    // Initial EKF state.
    Eigen::Matrix<double, kDof, 1> initial_q = Eigen::Matrix<double, kDof, 1>::Zero();
    Eigen::Matrix<double, kDof, 1> initial_dq = Eigen::Matrix<double, kDof, 1>::Zero();
    for (int i = 0; i < kDof && i < static_cast<int>(initial_q_param_.size()); ++i) {
        initial_q[i] = initial_q_param_[i];
    }
    for (int i = 0; i < kDof && i < static_cast<int>(initial_dq_param_.size()); ++i) {
        initial_dq[i] = initial_dq_param_[i];
    }
    ekf_.init(initial_q, initial_dq, initial_q_variance_, initial_dq_variance_);
    applyStateConstraints("initialization");
    updateArmKinematics(ekf_.q());

    // ROS interfaces.
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    robot_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        robot_joint_topic_,
        static_cast<std::size_t>(robot_joint_queue_depth_),
        std::bind(&ArmPoseEkfNode::robotJointCallback, this, std::placeholders::_1));
    state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(output_topic_, 10);

    for (const auto& keypoint_name : {"right_wrist", "right_elbow", "right_shoulder", "left_shoulder"}) {
        keypoint_measurements_[keypoint_name] = PointMeasurement();
        keypoint_subs_.push_back(
            this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
                std::string("/keypoint_distributions/") + keypoint_name,
                static_cast<std::size_t>(keypoint_queue_depth_),
                [this, keypoint_name](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                    this->keypointCallback(msg, keypoint_name);
                }));
    }

    RCLCPP_INFO(this->get_logger(), "arm_pose_ekf initialized.");
}

// =============================================================================
// DATA ACQUISITION: ROS subscription callbacks
// =============================================================================

// ***** MAIN TRIGGER *****
// Every robot joint-state message drives one full EKF cycle:
//   predict -> hand pose/twist correction -> keypoint correction -> publish.
// The keypoint callbacks below only cache data; they do NOT advance the filter.
void ArmPoseEkfNode::robotJointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->position.size() != static_cast<std::size_t>(robot_model_.nq) ||
        msg->velocity.size() != static_cast<std::size_t>(robot_model_.nv)) {
        RCLCPP_WARN(this->get_logger(), "Robot joint state size mismatch with model.");
        return;
    }

    const rclcpp::Time stamp(msg->header.stamp);
    if (initialized_ && max_update_rate_hz_ > 0.0) {
        const double min_period = 1.0 / max_update_rate_hz_;
        if ((stamp - last_processed_robot_stamp_).seconds() < min_period) {
            return;
        }
    }

    if (!initialized_) {
        initialized_ = true;
        last_predict_stamp_ = stamp;
    }
    predictTo(stamp);
    last_processed_robot_stamp_ = stamp;

    correctHandPoseAndTwist(*msg);
    correctKeypointMeasurements();
    publishState(stamp);
}

// Passive: transform the camera-frame Gaussian into the shoulder frame and
// cache it. Consumed later by correctKeypointMeasurements() during the main
// trigger. Does not advance the EKF.
void ArmPoseEkfNode::keypointCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg,
    const std::string& keypoint_name)
{
    if (!use_keypoints_) {
        return;
    }
    PointMeasurement measurement;
    if (!transformPointMeasurementToShoulder(*msg, measurement)) {
        return;
    }
    keypoint_measurements_[keypoint_name] = measurement;
}

bool ArmPoseEkfNode::transformPointMeasurementToShoulder(
    const geometry_msgs::msg::PoseWithCovarianceStamped& msg,
    PointMeasurement& measurement) const
{
    // keypoint_distribution_extraction publishes each Gaussian in the camera
    // optical frame. The arm EKF model is rooted at RightShoulder, so
    // lookupTransform(target=shoulder, source=message_frame) maps points from
    // the message frame into the shoulder frame (rotation + shoulder-origin
    // translation).
    pinocchio::SE3 T_shoulder_source = pinocchio::SE3::Identity();
    try {
        auto tf = tf_buffer_->lookupTransform(
            shoulder_frame_name_,
            msg.header.frame_id,
            msg.header.stamp,
            rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));
        T_shoulder_source.translation() = Eigen::Vector3d(
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z);
        Eigen::Quaterniond q(
            tf.transform.rotation.w,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z);
        T_shoulder_source.rotation() = q.normalized().toRotationMatrix();
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Failed to transform keypoint distribution to shoulder frame: %s", ex.what());
        return false;
    }

    const Eigen::Vector3d p_source(
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z);
    measurement.mean = T_shoulder_source.act(p_source);
    measurement.covariance =
        T_shoulder_source.rotation() * covarianceFromMessage(msg) * T_shoulder_source.rotation().transpose();
    measurement.stamp = msg.header.stamp;
    measurement.available = true;
    return measurement.mean.allFinite() && measurement.covariance.allFinite();
}

pinocchio::SE3 ArmPoseEkfNode::robotEndEffectorPose(
    const sensor_msgs::msg::JointState& msg,
    Eigen::Matrix<double, 6, 1>& ee_twist_world)
{
    Eigen::VectorXd q = Eigen::VectorXd::Map(msg.position.data(), robot_model_.nq);
    Eigen::VectorXd dq = Eigen::VectorXd::Map(msg.velocity.data(), robot_model_.nv);
    pinocchio::computeJointJacobians(robot_model_, robot_data_, q);
    pinocchio::framesForwardKinematics(robot_model_, robot_data_, q);
    pinocchio::updateFramePlacements(robot_model_, robot_data_);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, robot_model_.nv);
    J = pinocchio::getFrameJacobian(robot_model_, robot_data_, robot_ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
    ee_twist_world = J * dq;
    return robot_data_.oMf[robot_ee_frame_id_];
}

pinocchio::SE3 ArmPoseEkfNode::lookupShoulderToEndEffector(rclcpp::Time& tf_stamp) const
{
    const auto tf = tf_buffer_->lookupTransform(
        shoulder_frame_name_,
        ee_frame_name_,
        tf2::TimePointZero);

    pinocchio::SE3 T = pinocchio::SE3::Identity();
    T.translation() = Eigen::Vector3d(
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z);
    Eigen::Quaterniond q(
        tf.transform.rotation.w,
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z);
    T.rotation() = q.normalized().toRotationMatrix();
    tf_stamp = tf.header.stamp;
    return T;
}

pinocchio::SE3 ArmPoseEkfNode::lookupWorldToShoulder(const rclcpp::Time& stamp) const
{
    geometry_msgs::msg::TransformStamped tf;
    if (use_latest_shoulder_tf_) {
        tf = tf_buffer_->lookupTransform(world_frame_, shoulder_frame_name_, tf2::TimePointZero);
    } else {
        try {
            tf = tf_buffer_->lookupTransform(
                world_frame_,
                shoulder_frame_name_,
                stamp,
                rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));
        } catch (const tf2::TransformException& ex) {
            if (!fallback_to_latest_tf_) {
                throw;
            }
            tf = tf_buffer_->lookupTransform(world_frame_, shoulder_frame_name_, tf2::TimePointZero);
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Timestamped shoulder TF unavailable, using latest: %s", ex.what());
        }
    }

    pinocchio::SE3 T = pinocchio::SE3::Identity();
    T.translation() = Eigen::Vector3d(
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z);
    Eigen::Quaterniond q(
        tf.transform.rotation.w,
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z);
    T.rotation() = q.normalized().toRotationMatrix();
    return T;
}

Eigen::Matrix3d ArmPoseEkfNode::covarianceFromMessage(
    const geometry_msgs::msg::PoseWithCovarianceStamped& msg)
{
    // Only the translational XYZ covariance is used for keypoint updates.
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            covariance(row, col) = msg.pose.covariance[row * 6 + col];
        }
    }
    return covariance;
}

// =============================================================================
// DATA PROCESSING: prediction & state constraints
// =============================================================================
void ArmPoseEkfNode::predictTo(const rclcpp::Time& stamp)
{
    double dt = (stamp - last_predict_stamp_).seconds();
    if (dt <= 0.0 || dt > 1.0) {
        dt = 1.0 / 250.0;
    }
    last_predict_dt_ = dt;
    ekf_.predict(dt, process_acceleration_variance_);
    last_predict_stamp_ = stamp;
    applyStateConstraints("predict");
    updateArmKinematics(ekf_.q());
}

void ArmPoseEkfNode::loadStateConstraintsFromArmModel()
{
    // The EKF estimates only the seven independent coordinates listed in
    // joint_names_. Pinocchio stores URDF limits in the rendered xacro model
    // vectors using each joint's model q/v indices, so this copies only those
    // seven entries into compact EKF-space arrays.
    for (int i = 0; i < kDof; ++i) {
        const auto joint_id = arm_joint_ids_[i];
        const auto q_index = arm_model_.idx_qs[joint_id];
        const auto v_index = arm_model_.idx_vs[joint_id];

        q_lower_limits_[i] = arm_model_.lowerPositionLimit[q_index];
        q_upper_limits_[i] = arm_model_.upperPositionLimit[q_index];
        dq_velocity_limits_[i] = arm_model_.velocityLimit[v_index];

        RCLCPP_INFO(
            this->get_logger(),
            "EKF joint limit from URDF: %s q=[%.6f, %.6f] velocity=%.6f",
            joint_names_[i].c_str(),
            q_lower_limits_[i],
            q_upper_limits_[i],
            dq_velocity_limits_[i]);
    }
}

void ArmPoseEkfNode::truncateStateToInterval(int index, double lower, double upper)
{
    // Covariance-aware projection of state component `index` onto [lower, upper]
    // via Gaussian PDF truncation (Simon & Simon 2010). For an axis-aligned
    // (single-component) box constraint the general algorithm reduces to a
    // closed-form rank-1 update. With phi = e_index,
    //   s^2 = phi^T P phi = P(index,index),   v = P phi = P.col(index)
    // standardize the bounds to a = (lower-x)/s, b = (upper-x)/s, take the
    // truncated standard-normal moments (mean m, variance w) on [a, b], then
    //   x <- x + (m / s) * v
    //   P <- P + ((w - 1) / s^2) * v v^T
    // This shifts the mean toward the feasible region AND shrinks P along the
    // constrained direction, propagating to all correlated states (notably the
    // q<->dq cross-covariance). It is a no-op when the marginal already lies
    // several sigma inside the bounds (m~0, w~1).
    const double s2 = ekf_.P(index, index);
    if (!(s2 > 1.0e-12)) {
        return;  // component already (numerically) certain; nothing to truncate
    }
    const double s = std::sqrt(s2);
    const double x_i = ekf_.x[index];
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const double a = std::isfinite(lower) ? (lower - x_i) / s : -kInf;
    const double b = std::isfinite(upper) ? (upper - x_i) / s : kInf;
    if (a > b) {
        return;  // contradictory bounds; leave the state untouched
    }

    const TruncatedMoments moments = standardTruncatedMoments(a, b);
    const Eigen::Matrix<double, kStateSize, 1> v = ekf_.P.col(index);
    ekf_.x += (moments.mean / s) * v;
    ekf_.P += ((moments.variance - 1.0) / s2) * (v * v.transpose());
    ekf_.P = 0.5 * (ekf_.P + ekf_.P.transpose());
}

void ArmPoseEkfNode::applyStateConstraints(const char* source)
{
    if (!enforce_joint_limits_) {
        return;
    }

    constexpr double kInf = std::numeric_limits<double>::infinity();

    // Truncate each box-constrained position to its URDF limits.
    for (int i = 0; i < kDof; ++i) {
        const double before = ekf_.x[i];
        truncateStateToInterval(i, q_lower_limits_[i], q_upper_limits_[i]);
        const double lower = q_lower_limits_[i];
        const double upper = q_upper_limits_[i];
        const bool was_violating =
            (std::isfinite(lower) && before < lower) ||
            (std::isfinite(upper) && before > upper);
        if (was_violating) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Truncated %s from %.6f into [%.6f, %.6f] during %s (now %.6f).",
                joint_names_[i].c_str(), before, lower, upper, source, ekf_.x[i]);
        }
    }

    // Truncate each velocity to its URDF velocity limit.
    for (int i = 0; i < kDof; ++i) {
        const double velocity_limit = dq_velocity_limits_[i];
        if (std::isfinite(velocity_limit) && velocity_limit > 0.0) {
            truncateStateToInterval(kDof + i, -velocity_limit, velocity_limit);
        }
    }

    if (zero_velocity_at_limits_) {
        // When a joint sits on a position bound, forbid velocity that would push
        // it farther outside, as a covariance-aware one-sided velocity
        // truncation (q at lower -> dq >= 0; q at upper -> dq <= 0).
        for (int i = 0; i < kDof; ++i) {
            const double lower = q_lower_limits_[i];
            const double upper = q_upper_limits_[i];
            const bool at_lower = std::isfinite(lower) && ekf_.x[i] <= lower + 1.0e-6;
            const bool at_upper = std::isfinite(upper) && ekf_.x[i] >= upper - 1.0e-6;
            if (at_lower) {
                truncateStateToInterval(kDof + i, 0.0, kInf);
            } else if (at_upper) {
                truncateStateToInterval(kDof + i, -kInf, 0.0);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Measurement corrections.
// correctHandPoseAndTwist fuses the robot-derived hand observation. The robot
// rigidly holds the hand, so T_world_ee (from robot FK) maps to a measured hand
// pose/twist in the shoulder frame. Up to three sequential corrections run:
//   (1) hand POSITION : r = p_meas - p_model(q),  H = [J_v(q), 0]      (3D)
//   (2) hand ROTATION : r = log3(R_meas R_model^T), H = [J_w(q), 0]    (3D)
//   (3) hand TWIST    : r = v_meas - J_hand(q) dq, H = [0, J_hand(q)]  (6D)
// Jacobians are LOCAL_WORLD_ALIGNED, so the rotation residual is left/world-
// aligned (R_meas R_model^T) to match the angular Jacobian convention.
// -----------------------------------------------------------------------------
void ArmPoseEkfNode::correctHandPoseAndTwist(const sensor_msgs::msg::JointState& robot_msg)
{
    Eigen::Matrix<double, 6, 1> ee_twist_world;
    const pinocchio::SE3 T_world_ee = robotEndEffectorPose(robot_msg, ee_twist_world);

    rclcpp::Time shoulder_ee_tf_stamp(robot_msg.header.stamp);
    pinocchio::SE3 T_shoulder_ee = pinocchio::SE3::Identity();
    pinocchio::SE3 T_shoulder_hand_measured = pinocchio::SE3::Identity();
    try {
        T_shoulder_ee = lookupShoulderToEndEffector(shoulder_ee_tf_stamp);
        T_shoulder_hand_measured = T_shoulder_ee * T_ee_hand_;
        publishGroundTruthHandTransform(T_shoulder_hand_measured, shoulder_ee_tf_stamp);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Skipping hand pose/twist update because shoulder-to-EE TF is unavailable: %s", ex.what());
        return;
    }

    pinocchio::SE3 T_world_shoulder = pinocchio::SE3::Identity();
    try {
        T_world_shoulder = lookupWorldToShoulder(robot_msg.header.stamp);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Skipping hand pose/twist update because shoulder TF is unavailable: %s", ex.what());
        return;
    }

    // Transport the EE twist to the hand point, then express it in the shoulder
    // frame (the EKF's working frame).
    const Eigen::Vector3d ee_to_hand_world = T_world_ee.rotation() * T_ee_hand_.translation();
    Eigen::Matrix<double, 6, 1> hand_twist_world;
    hand_twist_world.head<3>() = ee_twist_world.head<3>() + ee_twist_world.tail<3>().cross(ee_to_hand_world);
    hand_twist_world.tail<3>() = ee_twist_world.tail<3>();

    Eigen::Matrix<double, 6, 1> hand_twist_shoulder;
    const Eigen::Matrix3d R_shoulder_world = T_world_shoulder.rotation().transpose();
    hand_twist_shoulder.head<3>() = R_shoulder_world * hand_twist_world.head<3>();
    hand_twist_shoulder.tail<3>() = R_shoulder_world * hand_twist_world.tail<3>();

    updateArmKinematics(ekf_.q());
    const pinocchio::SE3 T_model = arm_data_.oMf[hand_frame_id_];
    const Eigen::Matrix<double, 6, kDof> J_hand = handJacobian();

    if (use_hand_pose_ && use_hand_pose_position_) {
        Eigen::Vector3d residual = T_shoulder_hand_measured.translation() - T_model.translation();
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, kStateSize);
        H.block(0, 0, 3, kDof) = J_hand.topRows<3>();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * hand_position_sigma_ * hand_position_sigma_;
        ekf_.correct(residual, H, R);
        applyStateConstraints("hand position correction");
        updateArmKinematics(ekf_.q());
    }

    if (use_hand_pose_ && use_hand_pose_rotation_) {
        const pinocchio::SE3 T_updated = arm_data_.oMf[hand_frame_id_];
        const Eigen::Matrix<double, 6, kDof> J_updated = handJacobian();

        Eigen::Vector3d residual =
            pinocchio::log3(T_shoulder_hand_measured.rotation() * T_updated.rotation().transpose());
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, kStateSize);
        H.block(0, 0, 3, kDof) = J_updated.bottomRows<3>();

        // Grip compliance: when the robot EE rotates, the tool re-seats in the
        // hand, so EE rotation about the grip-compliant HAND axes never reaches
        // the wrist/forearm joints. Encode this as an anisotropic measurement
        // covariance specified in the hand frame: per-axis sigma =
        // hand_rotation_sigma_ * rotation_axis_ratio[x,y,z] (larger ratio ->
        // trusted less). Rotate it into the shoulder-aligned residual frame with
        // the current model hand orientation.
        const Eigen::Vector3d axis_sigma = hand_rotation_sigma_ * hand_rotation_axis_ratio_;
        const Eigen::Matrix3d R_hand_frame = axis_sigma.cwiseProduct(axis_sigma).asDiagonal();
        const Eigen::Matrix3d R_shoulder_aligned =
            T_updated.rotation() * R_hand_frame * T_updated.rotation().transpose();
        Eigen::Matrix3d R = 0.5 * (R_shoulder_aligned + R_shoulder_aligned.transpose());
        ekf_.correct(residual, H, R);
        applyStateConstraints("hand pose correction");
        updateArmKinematics(ekf_.q());
    }

    if (use_hand_twist_) {
        const Eigen::Matrix<double, 6, kDof> J_updated = handJacobian();
        Eigen::Matrix<double, 6, 1> residual = hand_twist_shoulder - J_updated * ekf_.dq();
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6, kStateSize);
        H.block(0, kDof, 6, kDof) = J_updated;
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(6, 6);
        R.block(0, 0, 3, 3).setIdentity();
        R.block(0, 0, 3, 3) *= hand_linear_velocity_sigma_ * hand_linear_velocity_sigma_;
        R.block(3, 3, 3, 3).setIdentity();
        R.block(3, 3, 3, 3) *= hand_angular_velocity_sigma_ * hand_angular_velocity_sigma_;
        ekf_.correct(residual, H, R);
        applyStateConstraints("hand twist correction");
        updateArmKinematics(ekf_.q());
    }
}

// Fuses camera keypoints as bone-DIRECTION measurements. Two sequential updates,
// each correcting only the DOFs that physically place that bone:
//   upper_arm (shoulder->elbow), active DOFs {0,1}: shoulder elv_angle/elv
//   forearm   (elbow->wrist),    active DOFs {2,3}: shoulder_rot, elbow flexion
// Residual is on the UNIT bone vector u = (p_end - p_start)/||.||:
//   r = u_meas - u_model(q),  H = [ (I - u u^T)/||s|| * (J_end - J_start), 0 ]
void ArmPoseEkfNode::correctKeypointMeasurements()
{
    if (!use_keypoints_) {
        return;
    }
    if (keypointSegmentsTooClose(last_predict_stamp_)) {
        // Elbow occluded / landmarks collapsed: vision can't constrain the arm,
        // so hold the proximal joints instead of letting the hand pose drift them.
        applyProximalAnchor();
        return;
    }

    updateArmKinematics(ekf_.q());

    const auto fresh_measurement = [this](const std::string& name) -> const PointMeasurement* {
        const auto it = keypoint_measurements_.find(name);
        if (it == keypoint_measurements_.end() || !it->second.available) {
            return nullptr;
        }
        if ((last_predict_stamp_ - it->second.stamp).seconds() > max_keypoint_age_sec_) {
            return nullptr;
        }
        return &it->second;
    };

    const auto* wrist_measurement = fresh_measurement("right_wrist");
    const auto* elbow_measurement = fresh_measurement("right_elbow");
    const auto* shoulder_measurement = fresh_measurement("right_shoulder");
    if (wrist_measurement == nullptr || elbow_measurement == nullptr || shoulder_measurement == nullptr) {
        // No fresh vision this cycle: hold the proximal joints.
        applyProximalAnchor();
        return;
    }

    struct SegmentDirectionMeasurement
    {
        std::string name;
        const PointMeasurement* start_measurement;
        const PointMeasurement* end_measurement;
        pinocchio::FrameIndex start_frame_id;
        pinocchio::FrameIndex end_frame_id;
        std::vector<int> active_dofs;
    };

    const std::array<SegmentDirectionMeasurement, 2> segment_measurements = {{
        // Upper-arm direction: the measured shoulder->elbow bone vector corrects
        // the two shoulder coordinates that place the elbow. Shoulder axial
        // rotation is excluded (it rotates the humerus about its long axis and
        // should not be inferred from elbow position).
        {"upper_arm", shoulder_measurement, elbow_measurement, shoulder_frame_id_, elbow_frame_id_, {0, 1}},
        // Forearm direction (after the upper-arm correction): the elbow->wrist
        // direction corrects shoulder axial rotation and elbow flexion. Wrist
        // coordinates are excluded.
        {"forearm", elbow_measurement, wrist_measurement, elbow_frame_id_, wrist_frame_id_, {2, 3}},
    }};

    bool vision_applied = false;
    for (const auto& measurement_spec : segment_measurements) {
        const double min_segment_length = std::max(0.0, min_keypoint_segment_length_);
        const Eigen::Vector3d measured_segment =
            measurement_spec.end_measurement->mean - measurement_spec.start_measurement->mean;
        const double measured_length = measured_segment.norm();
        if (measured_length < min_segment_length) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Skipping %s direction update because measured segment is too short: %.3f < %.3f.",
                measurement_spec.name.c_str(), measured_length, min_segment_length);
            continue;
        }
        const Eigen::Vector3d measured_direction = measured_segment / measured_length;

        updateArmKinematics(ekf_.q());
        const Eigen::Vector3d model_segment =
            arm_data_.oMf[measurement_spec.end_frame_id].translation() -
            arm_data_.oMf[measurement_spec.start_frame_id].translation();
        const double model_length = model_segment.norm();
        if (model_length < 1.0e-9) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Skipping %s direction update because model segment is too short.",
                measurement_spec.name.c_str());
            continue;
        }
        const Eigen::Vector3d model_direction = model_segment / model_length;

        // Direction residual: r = measured_unit_bone_vector - model_unit_bone_vector(q)
        const Eigen::Vector3d direction_residual = measured_direction - model_direction;

        // Jacobian of a normalized segment u = s/||s||:
        //   du/dq = (I - u u^T) / ||s|| * d(p_end - p_start)/dq
        const Eigen::Matrix3d model_normalization_jacobian =
            (Eigen::Matrix3d::Identity() - model_direction * model_direction.transpose()) / model_length;

        Eigen::Matrix<double, 3, kDof> J_segment =
            frameLinearJacobian(measurement_spec.end_frame_id) -
            frameLinearJacobian(measurement_spec.start_frame_id);
        Eigen::Matrix<double, 3, kDof> J_direction = model_normalization_jacobian * J_segment;
        Eigen::Matrix<double, 3, kDof> J_masked = Eigen::Matrix<double, 3, kDof>::Zero();
        for (const int active_dof : measurement_spec.active_dofs) {
            if (active_dof >= 0 && active_dof < kDof) {
                J_masked.col(active_dof) = J_direction.col(active_dof);
            }
        }

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, kStateSize);
        H.block(0, 0, 3, kDof) = J_masked;

        // Propagate the endpoint position covariances through the same
        // normalization used for the measured bone vector. The residual is
        // dimensionless, so direction_sigma is also dimensionless and behaves
        // roughly like a small-angle/radian noise floor.
        const Eigen::Matrix3d measured_normalization_jacobian =
            (Eigen::Matrix3d::Identity() - measured_direction * measured_direction.transpose()) / measured_length;
        Eigen::Matrix3d R =
            measured_normalization_jacobian *
            (measurement_spec.start_measurement->covariance + measurement_spec.end_measurement->covariance) *
            measured_normalization_jacobian.transpose();
        R = 0.5 * (R + R.transpose());
        if (!R.allFinite()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Skipping %s direction update because covariance is not finite.",
                measurement_spec.name.c_str());
            continue;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> covariance_solver(R);
        if (covariance_solver.info() != Eigen::Success) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Skipping %s direction update because covariance eigendecomposition failed.",
                measurement_spec.name.c_str());
            continue;
        }
        R = covariance_solver.eigenvectors() *
            covariance_solver.eigenvalues().cwiseMax(0.0).asDiagonal() *
            covariance_solver.eigenvectors().transpose();
        const double direction_sigma = std::max(1.0e-6, keypoint_direction_sigma_);
        R.diagonal().array() += direction_sigma * direction_sigma;

        ekf_.correct(direction_residual, H, R);
        applyStateConstraints("keypoint direction correction");
        updateArmKinematics(ekf_.q());
        vision_applied = true;
    }

    if (vision_applied) {
        // Cache the vision-confident proximal pose (q0-q3) to fall back on while
        // the elbow is later occluded. See applyProximalAnchor().
        proximal_anchor_ = ekf_.q().head<4>();
        proximal_anchor_valid_ = true;
    } else {
        applyProximalAnchor();
    }
}

// Holds the four vision-observed proximal joints (q0-q3: shoulder elv_angle,
// shoulder_elv, shoulder_rot, elbow_flexion) near their last vision-confident
// value via a soft equality pseudo-measurement, used while no keypoint update
// is available. Rationale: a 6-DOF hand pose cannot resolve the proximal
// null space (shoulder axial rotation in particular), so when vision drops out
// the hand-pose correction drifts those joints. Anchoring is information, not a
// hand-set covariance: it collapses P on q0-q3 the same way vision did, so the
// hand pose can no longer move them, and it pulls back drift that already
// started.  r = q_anchor - q,  H = [I4 on q0-q3, 0],  R = sigma^2 I.
//
// The anchor target COASTS rather than freezing: each occluded cycle it is
// advanced by dt * dq using the hand-twist-corrected velocity (the twist update
// is not vision-gated, so dq stays honest through the gap). This makes it a
// constant-velocity hold consistent with the predict step, so it cancels only
// the hand-pose null-space leak while letting twist + dynamics carry the
// proximal joints through a 1-2 s occlusion instead of pinning them.
void ArmPoseEkfNode::applyProximalAnchor()
{
    if (!use_proximal_anchor_ || !proximal_anchor_valid_) {
        return;
    }
    constexpr int kAnchorDof = 4;
    proximal_anchor_ += last_predict_dt_ * ekf_.dq().head<kAnchorDof>();
    Eigen::VectorXd residual = proximal_anchor_ - ekf_.q().head<kAnchorDof>();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(kAnchorDof, kStateSize);
    H.block(0, 0, kAnchorDof, kAnchorDof).setIdentity();
    const double sigma = std::max(1.0e-6, proximal_anchor_sigma_);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(kAnchorDof, kAnchorDof) * (sigma * sigma);
    ekf_.correct(residual, H, R);
    applyStateConstraints("proximal anchor");
    updateArmKinematics(ekf_.q());
}

bool ArmPoseEkfNode::keypointSegmentsTooClose(const rclcpp::Time& stamp) const
{
    if (!reject_keypoints_when_segments_too_close_) {
        return false;
    }

    const auto fresh_measurement = [this, &stamp](const std::string& name) -> const PointMeasurement* {
        const auto it = keypoint_measurements_.find(name);
        if (it == keypoint_measurements_.end() || !it->second.available) {
            return nullptr;
        }
        if ((stamp - it->second.stamp).seconds() > max_keypoint_age_sec_) {
            return nullptr;
        }
        return &it->second;
    };

    const auto* wrist = fresh_measurement("right_wrist");
    const auto* elbow = fresh_measurement("right_elbow");
    const auto* right_shoulder = fresh_measurement("right_shoulder");
    const auto* left_shoulder = fresh_measurement("left_shoulder");
    if (wrist == nullptr || elbow == nullptr || right_shoulder == nullptr || left_shoulder == nullptr) {
        return false;
    }

    const double upper_arm_length = (elbow->mean - right_shoulder->mean).norm();
    const double forearm_length = (wrist->mean - elbow->mean).norm();
    const double shoulder_line_length = (left_shoulder->mean - right_shoulder->mean).norm();
    const double min_length = std::max(0.0, min_keypoint_segment_length_);

    if (upper_arm_length >= min_length &&
        forearm_length >= min_length &&
        shoulder_line_length >= min_length) {
        return false;
    }

    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "Skipping keypoint EKF corrections because landmarks are too close: "
        "upper_arm=%.3f forearm=%.3f shoulder_line=%.3f min=%.3f",
        upper_arm_length, forearm_length, shoulder_line_length, min_length);
    return true;
}

// =============================================================================
// DATA PROCESSING: kinematics / Jacobian helpers
// Bridge between the 7 independent EKF coordinates and the full Pinocchio arm
// model (which includes a mimic shoulder joint coupled to coordinate 0).
// =============================================================================
Eigen::VectorXd ArmPoseEkfNode::modelConfigurationFromOpt(const double* values) const
{
    Eigen::VectorXd q_model = Eigen::VectorXd::Zero(arm_model_.nq);
    for (int i = 0; i < kDof; ++i) {
        q_model[arm_model_.idx_qs[arm_joint_ids_[i]]] = arm_joint_multipliers_[i] * values[i];
    }
    q_model[arm_model_.idx_qs[mimic_joint_id_]] = mimic_joint_multiplier_ * values[0];
    return q_model;
}

Eigen::VectorXd ArmPoseEkfNode::modelConfigurationFromOpt(
    const Eigen::Matrix<double, kDof, 1>& values) const
{
    return modelConfigurationFromOpt(values.data());
}

Eigen::Matrix<double, 6, Eigen::Dynamic> ArmPoseEkfNode::independentJacobian(
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& full_jacobian) const
{
    Eigen::Matrix<double, 6, Eigen::Dynamic> reduced(6, kDof);
    reduced.setZero();
    for (int i = 0; i < kDof; ++i) {
        reduced.col(i) =
            arm_joint_multipliers_[i] * full_jacobian.col(arm_model_.idx_vs[arm_joint_ids_[i]]);
    }
    reduced.col(0) += mimic_joint_multiplier_ * full_jacobian.col(arm_model_.idx_vs[mimic_joint_id_]);
    return reduced;
}

Eigen::Matrix<double, 3, ArmPoseEkfNode::kDof> ArmPoseEkfNode::frameLinearJacobian(
    pinocchio::FrameIndex frame_id)
{
    Eigen::Matrix<double, 6, Eigen::Dynamic> J_full(6, arm_model_.nv);
    J_full.setZero();
    J_full = pinocchio::getFrameJacobian(arm_model_, arm_data_, frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    return independentJacobian(J_full).topRows<3>();
}

Eigen::Matrix<double, 6, ArmPoseEkfNode::kDof> ArmPoseEkfNode::handJacobian()
{
    Eigen::Matrix<double, 6, Eigen::Dynamic> J_full(6, arm_model_.nv);
    J_full.setZero();
    J_full = pinocchio::getFrameJacobian(arm_model_, arm_data_, hand_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
    return independentJacobian(J_full);
}

void ArmPoseEkfNode::updateArmKinematics(const Eigen::Matrix<double, kDof, 1>& q)
{
    const Eigen::VectorXd q_model = modelConfigurationFromOpt(q);
    pinocchio::computeJointJacobians(arm_model_, arm_data_, q_model);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_model);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);
}

// =============================================================================
// OUTPUT & VISUALIZATION
// =============================================================================
void ArmPoseEkfNode::publishState(const rclcpp::Time& stamp)
{
    // Final guardrail for the public output topic: the state is already
    // constrained after each predict/correction, but truncating here guarantees
    // robot_state_publisher never receives an out-of-limit joint angle.
    applyStateConstraints("publish");

    sensor_msgs::msg::JointState msg;
    msg.header.stamp = stamp;
    msg.name = joint_names_;
    const auto q = ekf_.q();
    const auto dq = ekf_.dq();
    msg.position.assign(q.data(), q.data() + kDof);
    msg.velocity.assign(dq.data(), dq.data() + kDof);
    state_pub_->publish(msg);
}

void ArmPoseEkfNode::publishGroundTruthHandTransform(
    const pinocchio::SE3& T_shoulder_hand,
    const rclcpp::Time& stamp) const
{
    // Visualization only: broadcast the robot-measured hand pose in the shoulder
    // frame so it can be compared against the estimated hand in RViz.
    geometry_msgs::msg::TransformStamped gt_transform;
    gt_transform.header.stamp = stamp;
    gt_transform.header.frame_id = shoulder_frame_name_;
    gt_transform.child_frame_id = "RightHand (Ground Truth)";
    gt_transform.transform.translation.x = T_shoulder_hand.translation().x();
    gt_transform.transform.translation.y = T_shoulder_hand.translation().y();
    gt_transform.transform.translation.z = T_shoulder_hand.translation().z();
    Eigen::Quaterniond q_gt(T_shoulder_hand.rotation());
    gt_transform.transform.rotation.w = q_gt.w();
    gt_transform.transform.rotation.x = q_gt.x();
    gt_transform.transform.rotation.y = q_gt.y();
    gt_transform.transform.rotation.z = q_gt.z();
    tf_broadcaster_->sendTransform(gt_transform);
}

// =============================================================================
// ENTRY POINT
// =============================================================================
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<ArmPoseEkfNode>();
        rclcpp::spin(node);
    } catch (const std::exception& ex) {
        RCLCPP_FATAL(rclcpp::get_logger("arm_pose_ekf"), "Failed to start arm_pose_ekf: %s", ex.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
