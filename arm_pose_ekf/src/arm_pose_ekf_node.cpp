#include "arm_pose_ekf/arm_pose_ekf_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>

#include <tf2/exceptions.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace {

std::string resolve_arm_xacro_path(const std::string& arm_xacro_file)
{
    if (!arm_xacro_file.empty() && arm_xacro_file.front() == '/') {
        return arm_xacro_file;
    }

    return ament_index_cpp::get_package_share_directory("image_pose_tracking") +
        "/config/" + arm_xacro_file;
}

std::string render_arm_xacro(
    const std::string &robot_prefix,
    const std::string &robot_color,
    double upper_arm_length,
    double forearm_length,
    const std::string& arm_xacro_file)
{
    const std::string xacro_path = resolve_arm_xacro_path(arm_xacro_file);

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
        " upper_arm_length:=" + std::to_string(upper_arm_length) +
        " forearm_length:=" + std::to_string(forearm_length) +
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

}  // namespace

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

void ArmPoseEkfNode::Ekf::predict(double dt, double acceleration_variance, double velocity_decay)
{
    const double step = std::max(dt, 1.0e-4);
    const double alpha = std::exp(-std::max(0.0, velocity_decay) * step);

    Eigen::Matrix<double, kStateSize, kStateSize> F =
        Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
    F.block<kDof, kDof>(0, kDof) = step * alpha * Eigen::Matrix<double, kDof, kDof>::Identity();
    F.block<kDof, kDof>(kDof, kDof) = alpha * Eigen::Matrix<double, kDof, kDof>::Identity();

    x.head<kDof>() += step * alpha * x.tail<kDof>();
    x.tail<kDof>() *= alpha;

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

double ArmPoseEkfNode::Ekf::normalizedInnovationSquared(
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& R) const
{
    if (residual.size() == 0) {
        return 0.0;
    }

    const Eigen::MatrixXd S = H * P * H.transpose() + R;
    const Eigen::VectorXd solved = S.ldlt().solve(residual);
    if (!solved.allFinite()) {
        return std::numeric_limits<double>::infinity();
    }
    return residual.dot(solved);
}

Eigen::Matrix<double, ArmPoseEkfNode::kDof, 1> ArmPoseEkfNode::Ekf::q() const
{
    return x.head<kDof>();
}

Eigen::Matrix<double, ArmPoseEkfNode::kDof, 1> ArmPoseEkfNode::Ekf::dq() const
{
    return x.tail<kDof>();
}

ArmPoseEkfNode::ArmPoseEkfNode()
    : rclcpp::Node("arm_pose_ekf")
{
    this->declare_parameter<std::string>("robot_prefix", "upt_");
    this->declare_parameter<std::string>("robot_color", "blue");
    this->declare_parameter<std::string>("world_frame", "lbr_link_0");
    this->declare_parameter<std::string>("shoulder_frame", "upt_RightShoulder");
    this->declare_parameter<std::string>("hand_frame", "upt_RightHandCOM");
    this->declare_parameter<std::string>("robot_ee_frame", "lbr_link_ee");
    this->declare_parameter<std::string>("arm_xacro_file", "right_arm_osim_shoulder_mesh.urdf.xacro");
    this->declare_parameter<double>("upper_arm_length", 0.35);
    this->declare_parameter<double>("forearm_length", 0.26);
    this->declare_parameter<std::vector<double>>("ee_to_hand.translation", {0.0, -0.03, 0.08});
    this->declare_parameter<std::vector<double>>(
        "ee_to_hand.rotation", {0.2988362387, 0.6408563821, -0.2988362387, 0.6408563821});
    this->declare_parameter<std::vector<double>>("state.initial_q", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    this->declare_parameter<std::vector<double>>("state.initial_dq", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    this->declare_parameter<double>("state.initial_q_variance", 1.0);
    this->declare_parameter<double>("state.initial_dq_variance", 1.0);
    this->declare_parameter<double>("process_noise.acceleration_variance", 1.0e-3);
    this->declare_parameter<double>("process_noise.velocity_decay", 1.0);
    this->declare_parameter<double>("processing.max_update_rate_hz", 100.0);
    this->declare_parameter<int>("processing.robot_joint_queue_depth", 1);
    this->declare_parameter<int>("processing.keypoint_queue_depth", 1);
    this->declare_parameter<bool>("debug.keypoint_markers.enabled", false);
    this->declare_parameter<std::string>("debug.keypoint_markers.topic", "/arm_pose_ekf/keypoint_markers");
    this->declare_parameter<double>("debug.keypoint_markers.sigma_scale", 2.0);
    this->declare_parameter<double>("debug.keypoint_markers.min_radius", 0.01);
    this->declare_parameter<double>("debug.keypoint_markers.max_radius", 0.30);
    this->declare_parameter<bool>("constraints.enforce_joint_limits", true);
    this->declare_parameter<bool>("constraints.zero_velocity_at_limits", true);
    this->declare_parameter<bool>("measurements.hand_pose.enabled", true);
    this->declare_parameter<bool>("measurements.hand_pose.use_position", true);
    this->declare_parameter<bool>("measurements.hand_pose.use_rotation", false);
    this->declare_parameter<double>("measurements.hand_pose.sigma_position", 0.01);
    this->declare_parameter<double>("measurements.hand_pose.sigma_rotation", 0.05);
    this->declare_parameter<bool>("measurements.hand_twist.enabled", true);
    this->declare_parameter<double>("measurements.hand_twist.sigma_linear_velocity", 0.03);
    this->declare_parameter<double>("measurements.hand_twist.sigma_angular_velocity", 0.1);
    this->declare_parameter<bool>("measurements.keypoint_distributions.enabled", true);
    this->declare_parameter<double>("measurements.keypoint_distributions.default_sigma", 0.05);
    this->declare_parameter<double>("measurements.keypoint_distributions.direction_sigma", 0.05);
    this->declare_parameter<double>("measurements.keypoint_distributions.max_age_sec", 0.2);
    this->declare_parameter<bool>("measurements.keypoint_distributions.reject_when_segments_too_close", true);
    this->declare_parameter<double>("measurements.keypoint_distributions.min_segment_length", 0.15);
    this->declare_parameter<bool>("measurements.keypoint_distributions.gating.enabled", true);
    this->declare_parameter<double>("measurements.keypoint_distributions.gating.nis_threshold", 11.34);
    this->declare_parameter<double>("timing.tf_lookup_timeout_sec", 0.05);
    this->declare_parameter<bool>("timing.use_latest_shoulder_tf", true);
    this->declare_parameter<bool>("timing.fallback_to_latest_tf", true);
    this->declare_parameter<std::string>("topics.robot_joint_states", "/lbr/joint_states");
    this->declare_parameter<std::string>("topics.output_joint_states", "/arm_pose_ekf/joint_states");

    robot_prefix_ = this->get_parameter("robot_prefix").as_string();
    robot_color_ = this->get_parameter("robot_color").as_string();
    world_frame_ = this->get_parameter("world_frame").as_string();
    shoulder_frame_name_ = this->get_parameter("shoulder_frame").as_string();
    hand_frame_name_ = this->get_parameter("hand_frame").as_string();
    ee_frame_name_ = this->get_parameter("robot_ee_frame").as_string();
    const std::string arm_xacro_file = this->get_parameter("arm_xacro_file").as_string();
    const double upper_arm_length = this->get_parameter("upper_arm_length").as_double();
    const double forearm_length = this->get_parameter("forearm_length").as_double();

    initial_q_variance_ = this->get_parameter("state.initial_q_variance").as_double();
    initial_dq_variance_ = this->get_parameter("state.initial_dq_variance").as_double();
    process_acceleration_variance_ = this->get_parameter("process_noise.acceleration_variance").as_double();
    velocity_decay_ = this->get_parameter("process_noise.velocity_decay").as_double();
    max_update_rate_hz_ = this->get_parameter("processing.max_update_rate_hz").as_double();
    robot_joint_queue_depth_ =
        std::max(1, static_cast<int>(this->get_parameter("processing.robot_joint_queue_depth").as_int()));
    keypoint_queue_depth_ =
        std::max(1, static_cast<int>(this->get_parameter("processing.keypoint_queue_depth").as_int()));
    publish_keypoint_debug_markers_ = this->get_parameter("debug.keypoint_markers.enabled").as_bool();
    const std::string keypoint_debug_marker_topic =
        this->get_parameter("debug.keypoint_markers.topic").as_string();
    keypoint_debug_marker_scale_ = this->get_parameter("debug.keypoint_markers.sigma_scale").as_double();
    keypoint_debug_marker_min_radius_ = this->get_parameter("debug.keypoint_markers.min_radius").as_double();
    keypoint_debug_marker_max_radius_ = this->get_parameter("debug.keypoint_markers.max_radius").as_double();
    enforce_joint_limits_ = this->get_parameter("constraints.enforce_joint_limits").as_bool();
    zero_velocity_at_limits_ = this->get_parameter("constraints.zero_velocity_at_limits").as_bool();
    use_hand_pose_ = this->get_parameter("measurements.hand_pose.enabled").as_bool();
    use_hand_pose_position_ = this->get_parameter("measurements.hand_pose.use_position").as_bool();
    use_hand_pose_rotation_ = this->get_parameter("measurements.hand_pose.use_rotation").as_bool();
    hand_position_sigma_ = this->get_parameter("measurements.hand_pose.sigma_position").as_double();
    hand_rotation_sigma_ = this->get_parameter("measurements.hand_pose.sigma_rotation").as_double();
    use_hand_twist_ = this->get_parameter("measurements.hand_twist.enabled").as_bool();
    hand_linear_velocity_sigma_ = this->get_parameter("measurements.hand_twist.sigma_linear_velocity").as_double();
    hand_angular_velocity_sigma_ = this->get_parameter("measurements.hand_twist.sigma_angular_velocity").as_double();
    use_keypoints_ = this->get_parameter("measurements.keypoint_distributions.enabled").as_bool();
    default_keypoint_sigma_ = this->get_parameter("measurements.keypoint_distributions.default_sigma").as_double();
    keypoint_direction_sigma_ = this->get_parameter("measurements.keypoint_distributions.direction_sigma").as_double();
    max_keypoint_age_sec_ = this->get_parameter("measurements.keypoint_distributions.max_age_sec").as_double();
    reject_keypoints_when_segments_too_close_ =
        this->get_parameter("measurements.keypoint_distributions.reject_when_segments_too_close").as_bool();
    min_keypoint_segment_length_ =
        this->get_parameter("measurements.keypoint_distributions.min_segment_length").as_double();
    use_keypoint_gating_ = this->get_parameter("measurements.keypoint_distributions.gating.enabled").as_bool();
    keypoint_nis_threshold_ =
        this->get_parameter("measurements.keypoint_distributions.gating.nis_threshold").as_double();
    tf_lookup_timeout_sec_ = this->get_parameter("timing.tf_lookup_timeout_sec").as_double();
    use_latest_shoulder_tf_ = this->get_parameter("timing.use_latest_shoulder_tf").as_bool();
    fallback_to_latest_tf_ = this->get_parameter("timing.fallback_to_latest_tf").as_bool();

    const Eigen::Vector3d ee_hand_translation = parameterVector3(
        *this, "ee_to_hand.translation", {0.0, -0.03, 0.08});
    const Eigen::Vector4d ee_hand_rotation = parameterVector4(
        *this, "ee_to_hand.rotation", {0.2988362387, 0.6408563821, -0.2988362387, 0.6408563821});
    Eigen::Quaterniond q_ee_hand(
        ee_hand_rotation[3], ee_hand_rotation[0], ee_hand_rotation[1], ee_hand_rotation[2]);
    T_ee_hand_.translation() = ee_hand_translation;
    T_ee_hand_.rotation() = q_ee_hand.normalized().toRotationMatrix();

    const std::string robot_urdf =
        ament_index_cpp::get_package_share_directory("lbr_description") + "/urdf/iiwa7/iiwa7.urdf";
    pinocchio::urdf::buildModel(robot_urdf, robot_model_);
    robot_data_ = pinocchio::Data(robot_model_);
    robot_ee_frame_id_ = robot_model_.getFrameId(ee_frame_name_);

    const std::string arm_urdf =
        render_arm_xacro(robot_prefix_, robot_color_, upper_arm_length, forearm_length, arm_xacro_file);
    pinocchio::urdf::buildModel(arm_urdf, arm_model_);
    std::remove(arm_urdf.c_str());
    arm_data_ = pinocchio::Data(arm_model_);

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

    Eigen::Matrix<double, kDof, 1> initial_q = Eigen::Matrix<double, kDof, 1>::Zero();
    Eigen::Matrix<double, kDof, 1> initial_dq = Eigen::Matrix<double, kDof, 1>::Zero();
    const auto initial_q_param = this->get_parameter("state.initial_q").as_double_array();
    const auto initial_dq_param = this->get_parameter("state.initial_dq").as_double_array();
    for (int i = 0; i < kDof && i < static_cast<int>(initial_q_param.size()); ++i) {
        initial_q[i] = initial_q_param[i];
    }
    for (int i = 0; i < kDof && i < static_cast<int>(initial_dq_param.size()); ++i) {
        initial_dq[i] = initial_dq_param[i];
    }
    ekf_.init(initial_q, initial_dq, initial_q_variance_, initial_dq_variance_);
    applyStateConstraints("initialization");
    updateArmKinematics(ekf_.q());

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    const std::string robot_joint_topic = this->get_parameter("topics.robot_joint_states").as_string();
    const std::string output_topic = this->get_parameter("topics.output_joint_states").as_string();

    robot_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        robot_joint_topic,
        static_cast<std::size_t>(robot_joint_queue_depth_),
        std::bind(&ArmPoseEkfNode::robotJointCallback, this, std::placeholders::_1));
    state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(output_topic, 10);
    if (publish_keypoint_debug_markers_) {
        keypoint_debug_marker_pub_ =
            this->create_publisher<visualization_msgs::msg::MarkerArray>(keypoint_debug_marker_topic, 10);
    }

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
    publishKeypointDebugMarkers(measurement.stamp);
}

void ArmPoseEkfNode::predictTo(const rclcpp::Time& stamp)
{
    double dt = (stamp - last_predict_stamp_).seconds();
    if (dt <= 0.0 || dt > 1.0) {
        dt = 1.0 / 250.0;
    }
    ekf_.predict(dt, process_acceleration_variance_, velocity_decay_);
    last_predict_stamp_ = stamp;
    applyStateConstraints("predict");
    updateArmKinematics(ekf_.q());
}

void ArmPoseEkfNode::loadStateConstraintsFromArmModel()
{
    // The EKF estimates only the seven independent coordinates listed in
    // joint_names_. Pinocchio stores URDF limits in the rendered xacro model
    // vectors using each joint's model q/v indices, so this method copies only
    // those seven entries into compact EKF-space arrays.
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

void ArmPoseEkfNode::applyStateConstraints(const char* source)
{
    if (!enforce_joint_limits_) {
        return;
    }

    bool constrained = false;
    for (int i = 0; i < kDof; ++i) {
        const double before = ekf_.x[i];
        const double dq_before = ekf_.x[kDof + i];
        const double lower = q_lower_limits_[i];
        const double upper = q_upper_limits_[i];
        const double velocity_limit = dq_velocity_limits_[i];

        if (std::isfinite(lower) && ekf_.x[i] < lower) {
            ekf_.x[i] = lower;
            constrained = true;
        }
        if (std::isfinite(upper) && ekf_.x[i] > upper) {
            ekf_.x[i] = upper;
            constrained = true;
        }

        double& dq = ekf_.x[kDof + i];
        if (std::isfinite(velocity_limit) && velocity_limit > 0.0) {
            dq = std::clamp(dq, -velocity_limit, velocity_limit);
            constrained = constrained || dq != dq_before;
        }

        if (!zero_velocity_at_limits_) {
            if (before != ekf_.x[i]) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Projected %s from %.6f to %.6f during %s. Limit=[%.6f, %.6f]",
                    joint_names_[i].c_str(), before, ekf_.x[i], source, lower, upper);
            }
            if (dq_before != dq) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Projected velocity of %s from %.6f to %.6f during %s. Limit=+/-%.6f",
                    joint_names_[i].c_str(), dq_before, dq, source, velocity_limit);
            }
            continue;
        }

        // If projection placed q on a limit, remove velocity that would
        // immediately push the next prediction farther outside the feasible
        // interval. This is intentionally simple: it preserves velocity away
        // from the limits and only suppresses outward motion at active bounds.
        const bool at_lower = std::isfinite(lower) && ekf_.x[i] <= lower + 1.0e-9;
        const bool at_upper = std::isfinite(upper) && ekf_.x[i] >= upper - 1.0e-9;
        if ((at_lower && dq < 0.0) || (at_upper && dq > 0.0)) {
            dq = 0.0;
            constrained = true;
        }

        if (before != ekf_.x[i]) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Projected %s from %.6f to %.6f during %s. Limit=[%.6f, %.6f]",
                joint_names_[i].c_str(), before, ekf_.x[i], source, lower, upper);
        }
        if (dq_before != dq) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Projected velocity of %s from %.6f to %.6f during %s. Limit=+/-%.6f",
                joint_names_[i].c_str(), dq_before, dq, source, velocity_limit);
        }
    }

    if (constrained) {
        // Keep the covariance symmetric after hard projection. A full
        // constrained EKF would also reshape P; this first-stage projection
        // keeps the state feasible while leaving uncertainty mostly intact.
        ekf_.P = 0.5 * (ekf_.P + ekf_.P.transpose());
    }
}

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
        updateArmKinematics(ekf_.q());
        const pinocchio::SE3 T_updated = arm_data_.oMf[hand_frame_id_];
        const Eigen::Matrix<double, 6, kDof> J_updated = handJacobian();

        // handJacobian() is requested in LOCAL_WORLD_ALIGNED, so its angular
        // rows represent the frame angular velocity expressed in the shoulder
        // axes. Match that convention with a left/world-aligned SO(3) error:
        //   R_measured ~= exp(residual) * R_model
        // Using R_model^T * R_measured gives a body-frame residual and fights
        // this Jacobian convention, which can make the wrist spin violently.
        Eigen::Vector3d residual =
            pinocchio::log3(T_shoulder_hand_measured.rotation() * T_updated.rotation().transpose());
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, kStateSize);
        H.block(0, 0, 3, kDof) = J_updated.bottomRows<3>();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * hand_rotation_sigma_ * hand_rotation_sigma_;
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

void ArmPoseEkfNode::correctKeypointMeasurements()
{
    if (!use_keypoints_) {
        return;
    }
    if (keypointSegmentsTooClose(last_predict_stamp_)) {
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
        // Upper-arm direction mirrors the first stage of arm_solver_forward:
        // use the measured shoulder->elbow bone vector to correct only the two
        // shoulder coordinates that place the elbow in space. Shoulder axial
        // rotation is intentionally excluded because it rotates the humerus
        // about its long axis and should not be inferred from elbow position.
        {"upper_arm", shoulder_measurement, elbow_measurement, shoulder_frame_id_, elbow_frame_id_, {0, 1}},
        // Forearm direction is applied after the upper-arm correction. With
        // q0-q1 already responsible for the upper-arm vector, the remaining
        // elbow->wrist direction is used to correct shoulder axial rotation
        // and elbow flexion, similar to the geometric solve in
        // arm_solver_forward.py. Wrist-only coordinates are left out because
        // they should not be driven by elbow/wrist segment direction.
        {"forearm", elbow_measurement, wrist_measurement, elbow_frame_id_, wrist_frame_id_, {2, 3}},
    }};

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

        // Direction residual, not position residual:
        //   r = measured_unit_bone_vector - model_unit_bone_vector(q)
        const Eigen::Vector3d direction_residual = measured_direction - model_direction;
        
        // The Jacobian is the derivative of a normalized segment u=s/||s||:
        //   du/dq = (I - u u^T) / ||s|| * d(p_end - p_start)/dq
        //
        // The old position-based correction was:
        //   r = measured_link_position - model_link_position(q)
        //   H = [J_link_position 0]
        // It is intentionally replaced here by bone-vector residuals.
        
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

        const double nis = ekf_.normalizedInnovationSquared(direction_residual, H, R);
        if (!std::isfinite(nis) || nis < 0.0) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Skipping %s direction update because Mahalanobis distance is invalid.",
                measurement_spec.name.c_str());
            continue;
        }
        if (use_keypoint_gating_ && nis > keypoint_nis_threshold_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Rejecting %s direction update by NIS gate: %.3f > %.3f.",
                measurement_spec.name.c_str(), nis, keypoint_nis_threshold_);
            continue;
        }

        ekf_.correct(direction_residual, H, R);
        applyStateConstraints("keypoint direction correction");
        updateArmKinematics(ekf_.q());
    }
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

void ArmPoseEkfNode::publishState(const rclcpp::Time& stamp)
{
    // This final projection is a guardrail for the public output topic. The
    // state is already projected after each predict/correction step, but
    // clamping here guarantees robot_state_publisher never receives an
    // out-of-limit joint angle if a future code path publishes early.
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

void ArmPoseEkfNode::publishKeypointDebugMarkers(const rclcpp::Time& stamp)
{
    if (!publish_keypoint_debug_markers_ || !keypoint_debug_marker_pub_) {
        return;
    }

    visualization_msgs::msg::MarkerArray markers;

    // Clear previous markers first so unavailable or stale keypoints disappear
    // from RViz. Every marker below is already expressed in shoulder_frame_name_
    // because keypointCallback stores the transformed PointMeasurement.
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear_marker);

    const std::array<std::string, 4> names = {
        "right_wrist",
        "right_elbow",
        "right_shoulder",
        "left_shoulder",
    };

    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto it = keypoint_measurements_.find(names[i]);
        if (it == keypoint_measurements_.end() || !it->second.available) {
            continue;
        }

        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> covariance_solver(
            0.5 * (it->second.covariance + it->second.covariance.transpose()));
        double radius = keypoint_debug_marker_min_radius_;
        if (covariance_solver.info() == Eigen::Success) {
            const double max_variance = std::max(0.0, covariance_solver.eigenvalues().maxCoeff());
            radius = keypoint_debug_marker_scale_ * std::sqrt(max_variance);
            radius = std::clamp(radius, keypoint_debug_marker_min_radius_, keypoint_debug_marker_max_radius_);
        }

        visualization_msgs::msg::Marker marker;
        marker.header.stamp = stamp;
        marker.header.frame_id = shoulder_frame_name_;
        marker.ns = "arm_pose_ekf_keypoints_in_shoulder";
        marker.id = static_cast<int>(i);
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = it->second.mean.x();
        marker.pose.position.y = it->second.mean.y();
        marker.pose.position.z = it->second.mean.z();
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2.0 * radius;
        marker.scale.y = 2.0 * radius;
        marker.scale.z = 2.0 * radius;
        marker.color.a = 0.65;
        marker.color.r = names[i] == "right_wrist" ? 0.1 : 0.9;
        marker.color.g = names[i] == "right_elbow" ? 0.9 : 0.3;
        marker.color.b = names[i] == "right_shoulder" ? 0.9 : 0.1;
        markers.markers.push_back(marker);
    }

    keypoint_debug_marker_pub_->publish(markers);
}

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

bool ArmPoseEkfNode::transformPointMeasurementToShoulder(
    const geometry_msgs::msg::PoseWithCovarianceStamped& msg,
    PointMeasurement& measurement) const
{
    // keypoint_distribution_extraction publishes each Gaussian in the camera
    // optical frame. The arm EKF model, however, is rooted at RightShoulder.
    // lookupTransform(target=shoulder, source=message_frame) returns the
    // transform that maps points from the message frame into the shoulder
    // frame, including both rotation and the shoulder-origin translation.
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

void ArmPoseEkfNode::publishGroundTruthHandTransform(
    const pinocchio::SE3& T_shoulder_hand,
    const rclcpp::Time& stamp) const
{
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

void ArmPoseEkfNode::updateArmKinematics(const Eigen::Matrix<double, kDof, 1>& q)
{
    const Eigen::VectorXd q_model = modelConfigurationFromOpt(q);
    pinocchio::computeJointJacobians(arm_model_, arm_data_, q_model);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_model);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);
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

bool ArmPoseEkfNode::isFinite(const Eigen::MatrixXd& matrix)
{
    return matrix.allFinite();
}

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
