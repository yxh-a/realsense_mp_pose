#include "arm_pose_pf/arm_pose_pf_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <yaml-cpp/yaml.h>

#include <tf2/exceptions.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
// FILE-LOCAL HELPERS
// ─────────────────────────────────────────────────────────────────────────────

namespace {

struct ArmLengths
{
    double upper_arm_length;
    double forearm_length;
    double hand_length;
};

std::string resolve_arm_xacro_path(const std::string& arm_xacro_file)
{
    if (!arm_xacro_file.empty() && arm_xacro_file.front() == '/') {
        return arm_xacro_file;
    }
    return ament_index_cpp::get_package_share_directory("image_pose_tracking") +
        "/config/" + arm_xacro_file;
}

std::string resolve_subject_arm_lengths_path(const std::string& arm_lengths_file)
{
    if (!arm_lengths_file.empty() && arm_lengths_file.front() == '/') {
        return arm_lengths_file;
    }
    return ament_index_cpp::get_package_share_directory("image_pose_tracking") +
        "/config/" + arm_lengths_file;
}

ArmLengths load_subject_arm_lengths(
    const std::string& arm_lengths_file,
    int subject_id,
    const ArmLengths& fallback)
{
    const std::string yaml_path = resolve_subject_arm_lengths_path(arm_lengths_file);
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
    double upper_arm_length,
    double forearm_length,
    double hand_length,
    const std::string& arm_xacro_file)
{
    const std::string xacro_path = resolve_arm_xacro_path(arm_xacro_file);

    char temp_template[] = "/tmp/arm_pose_pf_osim_XXXXXX.urdf";
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
        " hand_length:=" + std::to_string(hand_length) +
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

Eigen::Matrix<double, 7, 1> parameterVector7(
    const rclcpp::Node& node,
    const std::string& name,
    const std::vector<double>& fallback)
{
    const auto values = node.get_parameter(name).as_double_array();
    const auto& source = values.size() == 7 ? values : fallback;
    Eigen::Matrix<double, 7, 1> result;
    for (int i = 0; i < 7; ++i) {
        result[i] = source[static_cast<std::size_t>(i)];
    }
    return result;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// PARTICLE FILTER — INTERNAL ALGORITHMS
//
// Bootstrap PF over state x = [q₀…q₆, dq₀…dq₆]ᵀ (dim 14).
// External contract:
//   init() → { predict() → correct*() → normalizeWeights() → resampleIfNeeded() }*
// ─────────────────────────────────────────────────────────────────────────────

void ArmPosePfNode::Pf::init(
    const Eigen::Matrix<double, kDof, 1>& q0,
    const Eigen::Matrix<double, kDof, 1>& dq0,
    const Eigen::Matrix<double, kDof, 1>& q_stddev,
    double dq_variance,
    int particle_count,
    unsigned int seed,
    double resample_ratio,
    double roughening_stddev_param)
{
    rng.seed(seed);
    resample_effective_ratio = std::clamp(resample_ratio, 0.0, 1.0);
    roughening_stddev = std::max(0.0, roughening_stddev_param);

    x.head<kDof>() = q0;
    x.tail<kDof>() = dq0;
    P.setZero();
    for (int i = 0; i < kDof; ++i) {
        const double sigma = std::max(0.0, q_stddev[i]);
        P(i, i) = sigma * sigma;
    }
    P.block<kDof, kDof>(kDof, kDof).setIdentity();
    P.block<kDof, kDof>(kDof, kDof) *= dq_variance;

    const int count = std::max(1, particle_count);
    particles.assign(static_cast<std::size_t>(count), Particle());
    const double uniform_log_weight = -std::log(static_cast<double>(count));
    std::normal_distribution<double> dq_noise(0.0, std::sqrt(std::max(0.0, dq_variance)));
    for (auto& particle : particles) {
        particle.x.head<kDof>() = q0;
        particle.x.tail<kDof>() = dq0;
        for (int i = 0; i < kDof; ++i) {
            std::normal_distribution<double> q_noise(0.0, std::max(0.0, q_stddev[i]));
            particle.x[i] += q_noise(rng);
            particle.x[kDof + i] += dq_noise(rng);
        }
        particle.log_weight = uniform_log_weight;
    }
    refreshEstimateFromParticles();
}

void ArmPosePfNode::Pf::predict(double dt, double acceleration_variance, double velocity_decay)
{
    const double step = std::max(dt, 1.0e-4);
    const double alpha = std::exp(-std::max(0.0, velocity_decay) * step);
    const double acceleration_stddev = std::sqrt(std::max(0.0, acceleration_variance));
    std::normal_distribution<double> acceleration_noise(0.0, acceleration_stddev);

    for (auto& particle : particles) {
        const Eigen::Matrix<double, kDof, 1> dq_before = particle.x.tail<kDof>();
        for (int i = 0; i < kDof; ++i) {
            const double acceleration = acceleration_noise(rng);
            particle.x[i] += step * alpha * dq_before[i] + 0.5 * step * step * acceleration;
            particle.x[kDof + i] = alpha * dq_before[i] + step * acceleration;
        }
    }
    refreshEstimateFromParticles();
}

void ArmPosePfNode::Pf::addLogLikelihood(std::size_t particle_index, double log_likelihood)
{
    if (particle_index >= particles.size()) {
        return;
    }
    particles[particle_index].log_weight += log_likelihood;
}

bool ArmPosePfNode::Pf::normalizeWeights()
{
    if (particles.empty()) {
        return false;
    }

    double max_log_weight = -std::numeric_limits<double>::infinity();
    for (const auto& particle : particles) {
        max_log_weight = std::max(max_log_weight, particle.log_weight);
    }
    if (!std::isfinite(max_log_weight)) {
        const double uniform_log_weight = -std::log(static_cast<double>(particles.size()));
        for (auto& particle : particles) {
            particle.log_weight = uniform_log_weight;
        }
        refreshEstimateFromParticles();
        return false;
    }

    double weight_sum = 0.0;
    for (const auto& particle : particles) {
        weight_sum += std::exp(particle.log_weight - max_log_weight);
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        const double uniform_log_weight = -std::log(static_cast<double>(particles.size()));
        for (auto& particle : particles) {
            particle.log_weight = uniform_log_weight;
        }
        refreshEstimateFromParticles();
        return false;
    }

    const double log_normalizer = max_log_weight + std::log(weight_sum);
    for (auto& particle : particles) {
        particle.log_weight -= log_normalizer;
    }
    refreshEstimateFromParticles();
    return true;
}

double ArmPosePfNode::Pf::effectiveSampleSize() const
{
    double weight_square_sum = 0.0;
    for (const auto& particle : particles) {
        const double weight = std::exp(particle.log_weight);
        weight_square_sum += weight * weight;
    }
    if (!(weight_square_sum > 0.0)) {
        return 0.0;
    }
    return 1.0 / weight_square_sum;
}

bool ArmPosePfNode::Pf::resampleIfNeeded()
{
    if (particles.empty()) {
        return false;
    }
    if (effectiveSampleSize() >= resample_effective_ratio * static_cast<double>(particles.size())) {
        return false;
    }

    std::vector<double> cumulative_weights(particles.size(), 0.0);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        cumulative += std::exp(particles[i].log_weight);
        cumulative_weights[i] = cumulative;
    }
    if (!(cumulative > 0.0) || !std::isfinite(cumulative)) {
        return false;
    }
    for (auto& weight : cumulative_weights) {
        weight /= cumulative;
    }

    std::vector<Particle> resampled;
    resampled.reserve(particles.size());
    std::uniform_real_distribution<double> offset_distribution(
        0.0, 1.0 / static_cast<double>(particles.size()));
    const double offset = offset_distribution(rng);
    std::size_t source_index = 0;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        const double sample =
            offset + static_cast<double>(i) / static_cast<double>(particles.size());
        while (source_index + 1 < cumulative_weights.size() &&
               cumulative_weights[source_index] < sample) {
            ++source_index;
        }
        resampled.push_back(particles[source_index]);
    }

    std::normal_distribution<double> roughening(0.0, roughening_stddev);
    const double uniform_log_weight = -std::log(static_cast<double>(resampled.size()));
    for (auto& particle : resampled) {
        for (int i = 0; i < kStateSize; ++i) {
            particle.x[i] += roughening(rng);
        }
        particle.log_weight = uniform_log_weight;
    }
    particles = std::move(resampled);
    refreshEstimateFromParticles();
    return true;
}

void ArmPosePfNode::Pf::refreshEstimateFromParticles()
{
    x.setZero();
    P.setZero();
    if (particles.empty()) {
        return;
    }

    double weight_sum = 0.0;
    for (const auto& particle : particles) {
        const double weight = std::exp(particle.log_weight);
        x += weight * particle.x;
        weight_sum += weight;
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        return;
    }
    x /= weight_sum;

    for (const auto& particle : particles) {
        const double weight = std::exp(particle.log_weight) / weight_sum;
        const Eigen::Matrix<double, kStateSize, 1> error = particle.x - x;
        P += weight * error * error.transpose();
    }
    P = 0.5 * (P + P.transpose());
}

Eigen::Matrix<double, ArmPosePfNode::kDof, 1> ArmPosePfNode::Pf::q() const
{
    return x.head<kDof>();
}

Eigen::Matrix<double, ArmPosePfNode::kDof, 1> ArmPosePfNode::Pf::dq() const
{
    return x.tail<kDof>();
}

// ─────────────────────────────────────────────────────────────────────────────
// NODE INITIALIZATION
// ─────────────────────────────────────────────────────────────────────────────

ArmPosePfNode::ArmPosePfNode()
    : rclcpp::Node("arm_pose_pf")
{
    // ── Parameter declarations ──────────────────────────────────────────────

    // Model and frame naming
    this->declare_parameter<std::string>("robot_prefix", "upt_");
    this->declare_parameter<std::string>("robot_color", "blue");
    this->declare_parameter<std::string>("world_frame", "lbr_link_0");
    this->declare_parameter<std::string>("shoulder_frame", "upt_RightShoulder");
    this->declare_parameter<std::string>("hand_frame", "upt_RightHandCOM");
    this->declare_parameter<std::string>("robot_ee_frame", "lbr_link_ee");
    this->declare_parameter<std::string>("robot_urdf_package", "lbr_description");
    this->declare_parameter<std::string>("robot_urdf_path", "urdf/iiwa7/iiwa7.urdf");
    this->declare_parameter<std::string>("arm_xacro_file",
        "right_arm_osim_shoulder_mesh.urdf.xacro");
    this->declare_parameter<std::string>("subject_arm_lengths_file", "subject_arm_lengths.yaml");
    this->declare_parameter<int>("subject_id", 1);
    this->declare_parameter<double>("upper_arm_length", 0.35);
    this->declare_parameter<double>("forearm_length", 0.26);
    this->declare_parameter<double>("hand_length", 0.0918);

    // Joint names (suffixes; robot_prefix is prepended at init)
    this->declare_parameter<std::vector<std::string>>("arm_joint_names", {
        "jRightShoulder_elv_angle",
        "jRightShoulder_shoulder_elv",
        "jRightShoulder_shoulder_rot",
        "jRightElbow_rotz",
        "jRightElbow_roty",
        "jRightWrist_rotx",
        "jRightWrist_rotz",
    });
    this->declare_parameter<std::string>("mimic_joint_name", "jRightShoulder_shoulder1_r2");
    this->declare_parameter<double>("mimic_joint_multiplier", -1.0);

    // EE-to-hand rigid transform [x y z w] quaternion convention
    this->declare_parameter<std::vector<double>>("ee_to_hand.translation", {0.0, -0.03, 0.08});
    this->declare_parameter<std::vector<double>>("ee_to_hand.rotation",
        {0.2988362387, 0.6408563821, -0.2988362387, 0.6408563821});

    // Initial state
    this->declare_parameter<std::vector<double>>("state.initial_q",
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    this->declare_parameter<std::vector<double>>("state.initial_dq",
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    this->declare_parameter<double>("state.initial_q_variance", 1.0);
    this->declare_parameter<double>("state.initial_dq_variance", 1.0);

    // Particle filter
    this->declare_parameter<int>("particle_filter.particle_count", 500);
    this->declare_parameter<int>("particle_filter.seed", 1);
    this->declare_parameter<double>(
        "particle_filter.joint_limit_initial_q_stddev_fraction", 0.2);
    this->declare_parameter<double>(
        "particle_filter.joint_limit_proposal_q_stddev_fraction", 0.2);
    this->declare_parameter<bool>("particle_filter.initialize_from_forward_solver", true);
    this->declare_parameter<double>("particle_filter.forward_solver_initial_q_variance", 0.05);
    this->declare_parameter<double>("particle_filter.forward_solver_initial_dq_variance", 0.01);
    this->declare_parameter<bool>("particle_filter.forward_solver_proposal.enabled", true);
    this->declare_parameter<double>(
        "particle_filter.forward_solver_proposal.resampling_fraction", 0.1);
    this->declare_parameter<std::vector<double>>(
        "particle_filter.forward_solver_proposal.q_stddev",
        {0.10, 0.10, 0.15, 0.12, 0.30, 0.40, 0.40});
    this->declare_parameter<std::vector<double>>(
        "particle_filter.forward_solver_proposal.dq_stddev",
        {0.30, 0.30, 0.40, 0.40, 0.50, 0.60, 0.60});
    this->declare_parameter<double>("particle_filter.resample_effective_ratio", 0.5);
    this->declare_parameter<double>("particle_filter.roughening_stddev", 1.0e-4);

    // Process noise and predict step
    this->declare_parameter<double>("process_noise.acceleration_variance", 1.0e-3);
    this->declare_parameter<double>("process_noise.velocity_decay", 1.0);
    this->declare_parameter<double>("process_noise.max_predict_dt_sec", 1.0);
    this->declare_parameter<double>("process_noise.fallback_predict_dt_sec", 1.0 / 250.0);

    // Processing rate and queue depths
    this->declare_parameter<double>("processing.max_update_rate_hz", 100.0);
    this->declare_parameter<int>("processing.robot_joint_queue_depth", 1);
    this->declare_parameter<int>("processing.forward_solver_queue_depth", 1);
    this->declare_parameter<int>("processing.output_queue_depth", 10);

    // Debug
    this->declare_parameter<std::string>("debug.ground_truth_hand_frame",
        "RightHand (Ground Truth)");

    // Constraint enforcement
    this->declare_parameter<bool>("constraints.enforce_joint_limits", true);
    this->declare_parameter<bool>("constraints.zero_velocity_at_limits", true);

    // Measurements
    this->declare_parameter<bool>("measurements.hand_pose.enabled", true);
    this->declare_parameter<bool>("measurements.hand_pose.use_position", true);
    this->declare_parameter<bool>("measurements.hand_pose.use_rotation", false);
    this->declare_parameter<double>("measurements.hand_pose.sigma_position", 0.01);
    this->declare_parameter<double>("measurements.hand_pose.sigma_rotation", 0.05);
    // Hand FK mode for the per-particle correction: exact per-particle FK
    // (parallelized) vs. a single linearization about the weighted mean.
    this->declare_parameter<bool>("measurements.hand_pose.use_linearized_fk", false);
    this->declare_parameter<bool>("measurements.hand_twist.enabled", true);
    this->declare_parameter<double>("measurements.hand_twist.sigma_linear_velocity", 0.03);
    this->declare_parameter<double>("measurements.hand_twist.sigma_angular_velocity", 0.1);
    // Vision (forward-solver q) as a robust per-particle likelihood, in addition
    // to its use as a resampling proposal. Heavy-tailed (Student-t) and gated so
    // noisy/outlier MediaPipe+pointcloud frames are down-weighted, not trusted.
    this->declare_parameter<bool>("measurements.vision.enabled", false);
    this->declare_parameter<std::vector<double>>("measurements.vision.q_sigma",
        {0.15, 0.15, 0.25, 0.20, 0.40, 0.50, 0.50});
    this->declare_parameter<double>("measurements.vision.student_t_dof", 3.0);
    this->declare_parameter<double>("measurements.vision.max_age_sec", 0.3);
    this->declare_parameter<bool>("measurements.vision.gating.enabled", true);
    this->declare_parameter<double>("measurements.vision.gating.nis_threshold", 18.48);

    // Timing and TF
    this->declare_parameter<double>("timing.tf_lookup_timeout_sec", 0.05);
    this->declare_parameter<bool>("timing.use_latest_shoulder_tf", true);
    this->declare_parameter<bool>("timing.fallback_to_latest_tf", true);

    // Topic names
    this->declare_parameter<std::string>("topics.robot_joint_states", "/lbr/joint_states");
    this->declare_parameter<std::string>("topics.forward_solver_joint_states", "/arm/joint_states");
    this->declare_parameter<std::string>("topics.output_joint_states",
        "/arm_pose_pf/joint_states");

    // ── Read parameters ─────────────────────────────────────────────────────

    robot_prefix_ = this->get_parameter("robot_prefix").as_string();
    robot_color_ = this->get_parameter("robot_color").as_string();
    world_frame_ = this->get_parameter("world_frame").as_string();
    shoulder_frame_name_ = this->get_parameter("shoulder_frame").as_string();
    hand_frame_name_ = this->get_parameter("hand_frame").as_string();
    ee_frame_name_ = this->get_parameter("robot_ee_frame").as_string();

    const std::string arm_xacro_file = this->get_parameter("arm_xacro_file").as_string();
    const std::string subject_arm_lengths_file =
        this->get_parameter("subject_arm_lengths_file").as_string();
    const int subject_id =
        std::max<int64_t>(1, this->get_parameter("subject_id").as_int());
    const ArmLengths arm_lengths = load_subject_arm_lengths(
        subject_arm_lengths_file, subject_id,
        {
            this->get_parameter("upper_arm_length").as_double(),
            this->get_parameter("forearm_length").as_double(),
            this->get_parameter("hand_length").as_double(),
        });

    initial_q_variance_ = this->get_parameter("state.initial_q_variance").as_double();
    initial_dq_variance_ = this->get_parameter("state.initial_dq_variance").as_double();
    particle_count_ = std::max(
        1, static_cast<int>(this->get_parameter("particle_filter.particle_count").as_int()));
    particle_seed_ = static_cast<unsigned int>(
        std::max<int64_t>(0, this->get_parameter("particle_filter.seed").as_int()));
    joint_limit_initial_q_stddev_fraction_ =
        this->get_parameter("particle_filter.joint_limit_initial_q_stddev_fraction").as_double();
    joint_limit_proposal_q_stddev_fraction_ =
        this->get_parameter("particle_filter.joint_limit_proposal_q_stddev_fraction").as_double();
    initialize_from_forward_solver_ =
        this->get_parameter("particle_filter.initialize_from_forward_solver").as_bool();
    forward_solver_initial_q_variance_ =
        this->get_parameter("particle_filter.forward_solver_initial_q_variance").as_double();
    forward_solver_initial_dq_variance_ =
        this->get_parameter("particle_filter.forward_solver_initial_dq_variance").as_double();
    forward_solver_proposal_enabled_ =
        this->get_parameter("particle_filter.forward_solver_proposal.enabled").as_bool();
    forward_solver_proposal_resampling_fraction_ = std::clamp(
        this->get_parameter(
            "particle_filter.forward_solver_proposal.resampling_fraction").as_double(),
        0.0, 1.0);
    forward_solver_proposal_q_stddev_ = parameterVector7(
        *this, "particle_filter.forward_solver_proposal.q_stddev",
        {0.10, 0.10, 0.15, 0.12, 0.30, 0.40, 0.40});
    forward_solver_proposal_dq_stddev_ = parameterVector7(
        *this, "particle_filter.forward_solver_proposal.dq_stddev",
        {0.30, 0.30, 0.40, 0.40, 0.50, 0.60, 0.60});
    particle_resample_effective_ratio_ =
        this->get_parameter("particle_filter.resample_effective_ratio").as_double();
    particle_roughening_stddev_ =
        this->get_parameter("particle_filter.roughening_stddev").as_double();

    process_acceleration_variance_ =
        this->get_parameter("process_noise.acceleration_variance").as_double();
    velocity_decay_ = this->get_parameter("process_noise.velocity_decay").as_double();
    max_predict_dt_sec_ = this->get_parameter("process_noise.max_predict_dt_sec").as_double();
    fallback_predict_dt_sec_ =
        this->get_parameter("process_noise.fallback_predict_dt_sec").as_double();

    max_update_rate_hz_ = this->get_parameter("processing.max_update_rate_hz").as_double();
    robot_joint_queue_depth_ = std::max(
        1, static_cast<int>(this->get_parameter("processing.robot_joint_queue_depth").as_int()));
    forward_solver_queue_depth_ = std::max(
        1, static_cast<int>(
            this->get_parameter("processing.forward_solver_queue_depth").as_int()));

    ground_truth_hand_frame_ = this->get_parameter("debug.ground_truth_hand_frame").as_string();

    enforce_joint_limits_ = this->get_parameter("constraints.enforce_joint_limits").as_bool();
    zero_velocity_at_limits_ =
        this->get_parameter("constraints.zero_velocity_at_limits").as_bool();

    use_hand_pose_ = this->get_parameter("measurements.hand_pose.enabled").as_bool();
    use_hand_pose_position_ =
        this->get_parameter("measurements.hand_pose.use_position").as_bool();
    use_hand_pose_rotation_ =
        this->get_parameter("measurements.hand_pose.use_rotation").as_bool();
    hand_position_sigma_ =
        this->get_parameter("measurements.hand_pose.sigma_position").as_double();
    hand_rotation_sigma_ =
        this->get_parameter("measurements.hand_pose.sigma_rotation").as_double();
    use_linearized_fk_ = this->get_parameter("measurements.hand_pose.use_linearized_fk").as_bool();
    use_hand_twist_ = this->get_parameter("measurements.hand_twist.enabled").as_bool();
    hand_linear_velocity_sigma_ =
        this->get_parameter("measurements.hand_twist.sigma_linear_velocity").as_double();
    hand_angular_velocity_sigma_ =
        this->get_parameter("measurements.hand_twist.sigma_angular_velocity").as_double();
    use_vision_measurement_ = this->get_parameter("measurements.vision.enabled").as_bool();
    vision_q_sigma_ = parameterVector7(
        *this, "measurements.vision.q_sigma", {0.15, 0.15, 0.25, 0.20, 0.40, 0.50, 0.50});
    vision_student_t_dof_ = this->get_parameter("measurements.vision.student_t_dof").as_double();
    vision_max_age_sec_ = this->get_parameter("measurements.vision.max_age_sec").as_double();
    use_vision_gating_ = this->get_parameter("measurements.vision.gating.enabled").as_bool();
    vision_nis_threshold_ =
        this->get_parameter("measurements.vision.gating.nis_threshold").as_double();

    tf_lookup_timeout_sec_ = this->get_parameter("timing.tf_lookup_timeout_sec").as_double();
    use_latest_shoulder_tf_ = this->get_parameter("timing.use_latest_shoulder_tf").as_bool();
    fallback_to_latest_tf_ = this->get_parameter("timing.fallback_to_latest_tf").as_bool();

    // ── EE-to-hand transform ────────────────────────────────────────────────

    const Eigen::Vector3d ee_hand_translation = parameterVector3(
        *this, "ee_to_hand.translation", {0.0, -0.03, 0.08});
    const Eigen::Vector4d ee_hand_rotation = parameterVector4(
        *this, "ee_to_hand.rotation",
        {0.2988362387, 0.6408563821, -0.2988362387, 0.6408563821});
    Eigen::Quaterniond q_ee_hand(
        ee_hand_rotation[3], ee_hand_rotation[0], ee_hand_rotation[1], ee_hand_rotation[2]);
    T_ee_hand_.translation() = ee_hand_translation;
    T_ee_hand_.rotation() = q_ee_hand.normalized().toRotationMatrix();

    // ── Build robot (manipulator) Pinocchio model ────────────────────────────

    const std::string robot_urdf =
        ament_index_cpp::get_package_share_directory(
            this->get_parameter("robot_urdf_package").as_string()) +
        "/" + this->get_parameter("robot_urdf_path").as_string();
    pinocchio::urdf::buildModel(robot_urdf, robot_model_);
    robot_data_ = pinocchio::Data(robot_model_);
    robot_ee_frame_id_ = robot_model_.getFrameId(ee_frame_name_);

    // ── Build arm (musculoskeletal) Pinocchio model ──────────────────────────

    const std::string arm_urdf = render_arm_xacro(
        robot_prefix_, robot_color_,
        arm_lengths.upper_arm_length, arm_lengths.forearm_length, arm_lengths.hand_length,
        arm_xacro_file);
    pinocchio::urdf::buildModel(arm_urdf, arm_model_);
    std::remove(arm_urdf.c_str());
    arm_data_ = pinocchio::Data(arm_model_);

    // Pre-allocate one Data per thread for the parallel per-particle FK loop.
#ifdef _OPENMP
    const int fk_thread_count = std::max(1, omp_get_max_threads());
#else
    const int fk_thread_count = 1;
#endif
    arm_data_pool_.assign(static_cast<std::size_t>(fk_thread_count), pinocchio::Data(arm_model_));

    // Joint names: yaml stores name suffixes; robot_prefix_ is prepended here.
    {
        const auto suffixes = this->get_parameter("arm_joint_names").as_string_array();
        if (static_cast<int>(suffixes.size()) != kDof) {
            throw std::runtime_error(
                "arm_joint_names must have exactly " + std::to_string(kDof) +
                " entries, got " + std::to_string(suffixes.size()));
        }
        joint_names_.resize(kDof);
        for (int i = 0; i < kDof; ++i) {
            joint_names_[i] = robot_prefix_ + suffixes[i];
        }
    }
    for (int i = 0; i < kDof; ++i) {
        arm_joint_ids_[i] = arm_model_.getJointId(joint_names_[i]);
        arm_joint_multipliers_[i] = 1.0;
    }
    loadStateConstraintsFromArmModel();

    mimic_joint_multiplier_ = this->get_parameter("mimic_joint_multiplier").as_double();
    mimic_joint_id_ = arm_model_.getJointId(
        robot_prefix_ + this->get_parameter("mimic_joint_name").as_string());

    hand_frame_id_ = arm_model_.getFrameId(hand_frame_name_);

    // ── Initial PF state ────────────────────────────────────────────────────

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
    if (!initialize_from_forward_solver_) {
        const Eigen::Matrix<double, kDof, 1> fallback_q_stddev =
            Eigen::Matrix<double, kDof, 1>::Constant(
                std::sqrt(std::max(0.0, initial_q_variance_)));
        pf_.init(
            initial_q, initial_dq,
            jointLimitQStddev(joint_limit_initial_q_stddev_fraction_, fallback_q_stddev),
            initial_dq_variance_, particle_count_, particle_seed_,
            particle_resample_effective_ratio_, particle_roughening_stddev_);
        particles_initialized_from_forward_solver_ = true;
        applyStateConstraints("initialization");
        updateArmKinematics(pf_.q());
    } else {
        updateArmKinematics(initial_q);
    }

    // ── ROS 2 communication ─────────────────────────────────────────────────

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    const std::string robot_joint_topic =
        this->get_parameter("topics.robot_joint_states").as_string();
    const std::string forward_solver_joint_topic =
        this->get_parameter("topics.forward_solver_joint_states").as_string();
    const std::string output_topic =
        this->get_parameter("topics.output_joint_states").as_string();

    robot_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        robot_joint_topic,
        static_cast<std::size_t>(robot_joint_queue_depth_),
        std::bind(&ArmPosePfNode::robotJointCallback, this, std::placeholders::_1));
    forward_solver_joint_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        forward_solver_joint_topic,
        static_cast<std::size_t>(forward_solver_queue_depth_),
        std::bind(&ArmPosePfNode::forwardSolverJointCallback, this, std::placeholders::_1));

    const int output_queue_depth = std::max(
        1, static_cast<int>(this->get_parameter("processing.output_queue_depth").as_int()));
    state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        output_topic, output_queue_depth);

    RCLCPP_INFO(this->get_logger(), "arm_pose_pf initialized.");
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA ACQUISITION — ROS 2 CALLBACKS
//
// robotJointCallback() is the main entry point that drives the full PF loop.
//
//   Per-message pipeline (rate-limited to max_update_rate_hz):
//     1. predictTo(stamp)            — propagate all particles via process model
//     2. correctHandPoseAndTwist()   — weight particles from robot EE FK / twist
//     3. correctVisionQ()            — optional robust vision likelihood (toggle)
//     4. publishState(stamp)         — broadcast weighted-mean joint estimate
//
// forwardSolverJointCallback() runs independently and stores the visual solver
// output [q0..q6, …].  That q is always used for one-time particle seeding and
// post-resample proposal injection; when measurements.vision.enabled is set it is
// ALSO fused as a robust (gated Student-t) per-particle likelihood in step 3.
// ─────────────────────────────────────────────────────────────────────────────

void ArmPosePfNode::robotJointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (initialize_from_forward_solver_ && !particles_initialized_from_forward_solver_) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Waiting for first forward_solver /arm/joint_states message before starting PF.");
        return;
    }

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

    // ── PF update loop ───────────────────────────────────────────────────────
    predictTo(stamp);
    last_processed_robot_stamp_ = stamp;
    correctHandPoseAndTwist(*msg);
    correctVisionQ();
    publishState(stamp);
}

void ArmPosePfNode::forwardSolverJointCallback(
    const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    // arm_solver_forward publishes [q0..q6, …]; only the first kDof values are used.
    if (msg->data.size() < static_cast<std::size_t>(kDof)) {
        return;
    }

    for (int i = 0; i < kDof; ++i) {
        latest_forward_solver_q_[i] =
            static_cast<double>(msg->data[static_cast<std::size_t>(i)]);
    }
    has_latest_forward_solver_q_ = latest_forward_solver_q_.allFinite();
    latest_forward_solver_q_stamp_ = this->now();

    if (initialize_from_forward_solver_ && !particles_initialized_from_forward_solver_) {
        initializeParticlesFromForwardSolver(*msg);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PF PIPELINE — MODEL INITIALIZATION
//   Called once from forwardSolverJointCallback() on the first valid message.
// ─────────────────────────────────────────────────────────────────────────────

bool ArmPosePfNode::initializeParticlesFromForwardSolver(
    const std_msgs::msg::Float32MultiArray& msg)
{
    if (msg.data.size() < static_cast<std::size_t>(kDof)) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Cannot initialize PF from forward_solver: expected at least %d joint values, "
            "received %zu.", kDof, msg.data.size());
        return false;
    }

    Eigen::Matrix<double, kDof, 1> q0 = Eigen::Matrix<double, kDof, 1>::Zero();
    Eigen::Matrix<double, kDof, 1> dq0 = Eigen::Matrix<double, kDof, 1>::Zero();
    for (int i = 0; i < kDof; ++i) {
        q0[i] = static_cast<double>(msg.data[static_cast<std::size_t>(i)]);
    }
    if (!q0.allFinite()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Cannot initialize PF from forward_solver because q contains non-finite values.");
        return false;
    }

    // arm_solver_forward gives a geometric rough q.  Use it only as the first
    // proposal center; particles later evolve through predict + measurement likelihoods.
    const Eigen::Matrix<double, kDof, 1> fallback_q_stddev =
        Eigen::Matrix<double, kDof, 1>::Constant(
            std::sqrt(std::max(0.0, forward_solver_initial_q_variance_)));
    pf_.init(
        q0, dq0,
        jointLimitQStddev(joint_limit_initial_q_stddev_fraction_, fallback_q_stddev),
        std::max(0.0, forward_solver_initial_dq_variance_),
        particle_count_, particle_seed_,
        particle_resample_effective_ratio_, particle_roughening_stddev_);
    particles_initialized_from_forward_solver_ = true;
    initialized_ = false;
    applyStateConstraints("forward_solver initialization");
    updateArmKinematics(pf_.q());

    RCLCPP_INFO(this->get_logger(),
        "Initialized PF from forward_solver q=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f] rad.",
        q0[0], q0[1], q0[2], q0[3], q0[4], q0[5], q0[6]);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// PF PIPELINE — PREDICT
//   Propagates all particles forward by dt seconds using the constant-velocity
//   + random-acceleration process model.  Called once per robotJointCallback.
// ─────────────────────────────────────────────────────────────────────────────

void ArmPosePfNode::predictTo(const rclcpp::Time& stamp)
{
    double dt = (stamp - last_predict_stamp_).seconds();
    if (dt <= 0.0 || dt > max_predict_dt_sec_) {
        dt = fallback_predict_dt_sec_;
    }
    pf_.predict(dt, process_acceleration_variance_, velocity_decay_);
    last_predict_stamp_ = stamp;
    applyStateConstraints("predict");
    updateArmKinematics(pf_.q());
}

// ─────────────────────────────────────────────────────────────────────────────
// PF PIPELINE — MEASUREMENT CORRECTION
//
// Hand pose/twist is the only correction source.  The measurement model h(qᵢ) is
// evaluated with the TRUE per-particle forward kinematics — no linearization.
// A particle filter exists precisely to weight particles by the exact nonlinear
// likelihood; linearizing h about the mean would collapse the update into an
// EKF-style min-norm step and suppress the large/null-space joint motion the PF
// is meant to capture.
//
// The per-particle FK is the dominant cost (one FK, plus one Jacobian when twist
// is enabled, per particle per update).  It is parallelized with OpenMP: each
// thread owns a pinocchio::Data from arm_data_pool_, and the particles are split
// across threads.  arm_model_ is read-only and shared.
// ─────────────────────────────────────────────────────────────────────────────

void ArmPosePfNode::correctHandPoseAndTwist(const sensor_msgs::msg::JointState& robot_msg)
{
    // ── Fetch robot EE pose and (optional) twist from joint states ───────────
    const bool need_twist = use_hand_twist_;
    Eigen::Matrix<double, 6, 1> ee_twist_world;
    const pinocchio::SE3 T_world_ee = robotEndEffectorPose(robot_msg, ee_twist_world, need_twist);

    // ── Fetch shoulder-to-EE transform via TF ─────────────────────────────────
    rclcpp::Time shoulder_ee_tf_stamp(robot_msg.header.stamp);
    pinocchio::SE3 T_shoulder_ee = pinocchio::SE3::Identity();
    pinocchio::SE3 T_shoulder_hand_measured = pinocchio::SE3::Identity();
    try {
        T_shoulder_ee = lookupShoulderToEndEffector(shoulder_ee_tf_stamp);
        T_shoulder_hand_measured = T_shoulder_ee * T_ee_hand_;
        publishGroundTruthHandTransform(T_shoulder_hand_measured, shoulder_ee_tf_stamp);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Skipping hand pose/twist update because shoulder-to-EE TF is unavailable: %s",
            ex.what());
        return;
    }

    pinocchio::SE3 T_world_shoulder = pinocchio::SE3::Identity();
    try {
        T_world_shoulder = lookupWorldToShoulder(robot_msg.header.stamp);
    } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Skipping hand pose/twist update because shoulder TF is unavailable: %s", ex.what());
        return;
    }

    // ── Compute hand twist in shoulder frame ──────────────────────────────────
    Eigen::Matrix<double, 6, 1> hand_twist_shoulder;
    if (need_twist) {
        const Eigen::Vector3d ee_to_hand_world = T_world_ee.rotation() * T_ee_hand_.translation();
        Eigen::Matrix<double, 6, 1> hand_twist_world;
        hand_twist_world.head<3>() =
            ee_twist_world.head<3>() + ee_twist_world.tail<3>().cross(ee_to_hand_world);
        hand_twist_world.tail<3>() = ee_twist_world.tail<3>();

        const Eigen::Matrix3d R_shoulder_world = T_world_shoulder.rotation().transpose();
        hand_twist_shoulder.head<3>() = R_shoulder_world * hand_twist_world.head<3>();
        hand_twist_shoulder.tail<3>() = R_shoulder_world * hand_twist_world.tail<3>();
    }

    if (!(use_hand_pose_ && (use_hand_pose_position_ || use_hand_pose_rotation_)) &&
        !use_hand_twist_) {
        updateArmKinematics(pf_.q());
        return;
    }

    // ── Build measurement covariance ──────────────────────────────────────────
    const int measurement_size =
        (use_hand_pose_ && use_hand_pose_position_ ? 3 : 0) +
        (use_hand_pose_ && use_hand_pose_rotation_ ? 3 : 0) +
        (need_twist ? 6 : 0);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(measurement_size, measurement_size);
    int offset = 0;
    if (use_hand_pose_ && use_hand_pose_position_) {
        covariance.block(offset, offset, 3, 3).diagonal().setConstant(
            hand_position_sigma_ * hand_position_sigma_);
        offset += 3;
    }
    if (use_hand_pose_ && use_hand_pose_rotation_) {
        covariance.block(offset, offset, 3, 3).diagonal().setConstant(
            hand_rotation_sigma_ * hand_rotation_sigma_);
        offset += 3;
    }
    if (need_twist) {
        covariance.block(offset, offset, 3, 3).diagonal().setConstant(
            hand_linear_velocity_sigma_ * hand_linear_velocity_sigma_);
        covariance.block(offset + 3, offset + 3, 3, 3).diagonal().setConstant(
            hand_angular_velocity_sigma_ * hand_angular_velocity_sigma_);
    }

    const int particle_count = static_cast<int>(pf_.particles.size());
    std::vector<double> likelihoods(static_cast<std::size_t>(particle_count), 0.0);

    if (use_linearized_fk_) {
        // ── Linearized correction: one FK + Jacobian at the weighted mean ──────
        // Approximates h(qᵢ) ≈ h(q̄) + J·(qᵢ − q̄). Cheap (O(1) FK) but biases the
        // update toward a min-norm step and discards nonlinearity; provided as a
        // toggle for benchmarking against the exact path below.
        const Eigen::Matrix<double, kDof, 1> q_mean = pf_.q();
        updateArmKinematics(q_mean, true);
        const pinocchio::SE3 T_model_mean = arm_data_.oMf[hand_frame_id_];
        Eigen::Matrix<double, 6, Eigen::Dynamic> J_full(6, arm_model_.nv);
        J_full.setZero();
        J_full = pinocchio::getFrameJacobian(
            arm_model_, arm_data_, hand_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
        const Eigen::Matrix<double, 6, kDof> J_hand_mean = independentJacobian(J_full);
        const Eigen::Matrix<double, 3, kDof> J_position = J_hand_mean.topRows<3>();
        const Eigen::Matrix<double, 3, kDof> J_rotation = J_hand_mean.bottomRows<3>();
        const Eigen::Vector3d position_residual_mean =
            T_shoulder_hand_measured.translation() - T_model_mean.translation();
        const Eigen::Vector3d rotation_residual_mean =
            pinocchio::log3(
                T_shoulder_hand_measured.rotation() * T_model_mean.rotation().transpose());

        for (std::size_t i = 0; i < pf_.particles.size(); ++i) {
            const Eigen::Matrix<double, kDof, 1> delta_q =
                pf_.particles[i].x.head<kDof>() - q_mean;
            Eigen::VectorXd residual(measurement_size);
            int local_offset = 0;
            if (use_hand_pose_ && use_hand_pose_position_) {
                residual.segment<3>(local_offset) = position_residual_mean - J_position * delta_q;
                local_offset += 3;
            }
            if (use_hand_pose_ && use_hand_pose_rotation_) {
                residual.segment<3>(local_offset) = rotation_residual_mean - J_rotation * delta_q;
                local_offset += 3;
            }
            if (need_twist) {
                residual.segment<6>(local_offset) =
                    hand_twist_shoulder - J_hand_mean * pf_.particles[i].x.tail<kDof>();
            }
            likelihoods[i] = gaussianLikelihood(residual, covariance);
        }
    } else {
        // ── Exact per-particle FK, parallelized over particles ─────────────────
        #pragma omp parallel
        {
#ifdef _OPENMP
            pinocchio::Data& data = arm_data_pool_[static_cast<std::size_t>(omp_get_thread_num())];
#else
            pinocchio::Data& data = arm_data_pool_[0];
#endif
            #pragma omp for schedule(static)
            for (int i = 0; i < particle_count; ++i) {
                likelihoods[static_cast<std::size_t>(i)] = handMeasurementLikelihood(
                    pf_.particles[static_cast<std::size_t>(i)].x, data,
                    T_shoulder_hand_measured, hand_twist_shoulder,
                    covariance, measurement_size, need_twist);
            }
        }
    }

    updateWeightsFromLikelihoods(likelihoods, "hand pose/twist update");
    updateArmKinematics(pf_.q());
}

double ArmPosePfNode::handMeasurementLikelihood(
    const Eigen::Matrix<double, kStateSize, 1>& state,
    pinocchio::Data& data,
    const pinocchio::SE3& T_shoulder_hand_measured,
    const Eigen::Matrix<double, 6, 1>& hand_twist_shoulder,
    const Eigen::MatrixXd& covariance,
    int measurement_size,
    bool need_twist) const
{
    // Exact forward kinematics for this particle on its thread-local Data.
    const Eigen::VectorXd q_model = modelConfigurationFromOpt(state.head<kDof>());
    if (need_twist) {
        pinocchio::computeJointJacobians(arm_model_, data, q_model);
    }
    pinocchio::framesForwardKinematics(arm_model_, data, q_model);
    pinocchio::updateFramePlacements(arm_model_, data);
    const pinocchio::SE3 T_model = data.oMf[hand_frame_id_];

    Eigen::VectorXd residual(measurement_size);
    int offset = 0;
    if (use_hand_pose_ && use_hand_pose_position_) {
        residual.segment<3>(offset) =
            T_shoulder_hand_measured.translation() - T_model.translation();
        offset += 3;
    }
    if (use_hand_pose_ && use_hand_pose_rotation_) {
        // Left/world-aligned SO(3) error matching the LOCAL_WORLD_ALIGNED Jacobian
        // convention:  R_measured ≈ exp(residual) · R_model.
        residual.segment<3>(offset) =
            pinocchio::log3(T_shoulder_hand_measured.rotation() * T_model.rotation().transpose());
        offset += 3;
    }
    if (need_twist) {
        Eigen::Matrix<double, 6, Eigen::Dynamic> J_full(6, arm_model_.nv);
        J_full.setZero();
        J_full = pinocchio::getFrameJacobian(
            arm_model_, data, hand_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
        const Eigen::Matrix<double, 6, kDof> J_hand = independentJacobian(J_full);
        residual.segment<6>(offset) = hand_twist_shoulder - J_hand * state.tail<kDof>();
    }
    return gaussianLikelihood(residual, covariance);
}

void ArmPosePfNode::updateWeightsFromLikelihoods(
    const std::vector<double>& likelihoods,
    const char* source)
{
    // Convert linear likelihoods to log and delegate (shared normalize/resample).
    std::vector<double> log_likelihoods(likelihoods.size());
    for (std::size_t i = 0; i < likelihoods.size(); ++i) {
        const double l = likelihoods[i];
        log_likelihoods[i] = (std::isfinite(l) && l > 0.0)
            ? std::log(l)
            : -std::numeric_limits<double>::infinity();
    }
    updateWeightsFromLogLikelihoods(log_likelihoods, source);
}

void ArmPosePfNode::updateWeightsFromLogLikelihoods(
    const std::vector<double>& log_likelihoods,
    const char* source)
{
    if (log_likelihoods.size() != pf_.particles.size()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Skipping %s: likelihood count does not match particle count.", source);
        return;
    }

    bool any_finite = false;
    for (const double ll : log_likelihoods) {
        if (std::isfinite(ll)) { any_finite = true; break; }
    }
    if (!any_finite) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Skipping %s: all particle likelihoods are invalid.", source);
        return;
    }

    for (std::size_t i = 0; i < log_likelihoods.size(); ++i) {
        pf_.addLogLikelihood(i, log_likelihoods[i]);
    }
    if (!pf_.normalizeWeights()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Particle weights collapsed during %s; reset to uniform.", source);
    }
    if (pf_.resampleIfNeeded()) {
        applyStateConstraints("resample");
        injectForwardSolverProposalParticles();
        applyStateConstraints("forward_solver proposal");
    }
    updateArmKinematics(pf_.q());
}

// ─────────────────────────────────────────────────────────────────────────────
// PF PIPELINE — VISION (FORWARD-SOLVER q) AS A ROBUST LIKELIHOOD
//
// Fuses the visual solver q as a per-particle measurement on joint angles. The
// residual qᵢ − q_vis is scored with an independent per-joint Student-t (heavy
// tails) so that high-variance MediaPipe/pointcloud jitter is down-weighted
// rather than trusted, and an optional Mahalanobis gate drops whole frames where
// vision grossly disagrees with the current estimate (outlier rejection).
//
// This is the same signal used for proposal injection; enabling both lets vision
// re-seed hypotheses (proposal) AND denoise the redundant joints over time
// (likelihood). Toggle with measurements.vision.enabled.
// ─────────────────────────────────────────────────────────────────────────────

void ArmPosePfNode::correctVisionQ()
{
    if (!use_vision_measurement_ || !has_latest_forward_solver_q_ || pf_.particles.empty()) {
        return;
    }

    // Reject stale vision (no fresh forward-solver frame within the age window).
    if (latest_forward_solver_q_stamp_.nanoseconds() <= 0) {
        return;
    }
    const double age_sec = (this->now() - latest_forward_solver_q_stamp_).seconds();
    if (age_sec > std::max(0.0, vision_max_age_sec_)) {
        return;
    }

    const Eigen::Matrix<double, kDof, 1> q_vis = latest_forward_solver_q_;

    // Gross-outlier gate on the mean residual (Gaussian-equivalent Mahalanobis).
    if (use_vision_gating_) {
        double mahalanobis = 0.0;
        const Eigen::Matrix<double, kDof, 1> mean_residual = pf_.q() - q_vis;
        for (int d = 0; d < kDof; ++d) {
            const double sigma = std::max(1.0e-6, vision_q_sigma_[d]);
            const double z = mean_residual[d] / sigma;
            mahalanobis += z * z;
        }
        if (!std::isfinite(mahalanobis) || mahalanobis > vision_nis_threshold_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Rejecting vision q update by NIS gate: %.3f > %.3f.",
                mahalanobis, vision_nis_threshold_);
            return;
        }
    }

    // Per-particle independent per-joint Student-t log-likelihood.
    //   log p(r) = -((nu+1)/2) * Σ_d log(1 + (r_d/sigma_d)² / nu) + const
    // The constant is identical across particles and cancels in normalization.
    const double nu = std::max(1.0e-3, vision_student_t_dof_);
    const double coefficient = 0.5 * (nu + 1.0);
    std::vector<double> log_likelihoods(pf_.particles.size(), 0.0);
    for (std::size_t i = 0; i < pf_.particles.size(); ++i) {
        const Eigen::Matrix<double, kDof, 1> q_i = pf_.particles[i].x.head<kDof>();
        double log_likelihood = 0.0;
        for (int d = 0; d < kDof; ++d) {
            const double sigma = std::max(1.0e-6, vision_q_sigma_[d]);
            const double z = (q_i[d] - q_vis[d]) / sigma;
            log_likelihood -= coefficient * std::log1p(z * z / nu);
        }
        log_likelihoods[i] = log_likelihood;
    }
    updateWeightsFromLogLikelihoods(log_likelihoods, "vision q update");
}

// ─────────────────────────────────────────────────────────────────────────────
// PF PIPELINE — RESAMPLING AND CONSTRAINT ENFORCEMENT
// ─────────────────────────────────────────────────────────────────────────────

// Replaces a fraction of particles with noisy samples around the latest visual
// solver q after resampling.  This re-seeds hypotheses lost to null-space drift
// without letting noisy camera output dominate the weight update.
void ArmPosePfNode::injectForwardSolverProposalParticles()
{
    if (!forward_solver_proposal_enabled_ ||
        !has_latest_forward_solver_q_ ||
        pf_.particles.empty()) {
        return;
    }

    const std::size_t particle_count = pf_.particles.size();
    const std::size_t proposal_count = std::min(
        particle_count,
        static_cast<std::size_t>(std::round(
            forward_solver_proposal_resampling_fraction_ *
            static_cast<double>(particle_count))));
    if (proposal_count == 0) {
        return;
    }

    const Eigen::Matrix<double, kDof, 1> dq_reference = pf_.dq();
    const Eigen::Matrix<double, kDof, 1> q_stddev =
        jointLimitQStddev(joint_limit_proposal_q_stddev_fraction_,
                          forward_solver_proposal_q_stddev_);

    for (std::size_t i = particle_count - proposal_count; i < particle_count; ++i) {
        auto& particle = pf_.particles[i];
        for (int dof = 0; dof < kDof; ++dof) {
            std::normal_distribution<double> q_noise(0.0, std::max(0.0, q_stddev[dof]));
            std::normal_distribution<double> dq_noise(
                0.0, std::max(0.0, forward_solver_proposal_dq_stddev_[dof]));
            particle.x[dof] = latest_forward_solver_q_[dof] + q_noise(pf_.rng);
            particle.x[kDof + dof] = dq_reference[dof] + dq_noise(pf_.rng);
        }
    }

    const double uniform_log_weight = -std::log(static_cast<double>(particle_count));
    for (auto& particle : pf_.particles) {
        particle.log_weight = uniform_log_weight;
    }
    pf_.refreshEstimateFromParticles();

    RCLCPP_DEBUG(this->get_logger(),
        "Injected %zu forward-solver proposal particles.", proposal_count);
}

// Reads joint position/velocity limits from the rendered arm URDF via Pinocchio
// and stores them in compact PF-space arrays (one entry per independent DOF).
void ArmPosePfNode::loadStateConstraintsFromArmModel()
{
    for (int i = 0; i < kDof; ++i) {
        const auto joint_id = arm_joint_ids_[i];
        const auto q_index  = arm_model_.idx_qs[joint_id];
        const auto v_index  = arm_model_.idx_vs[joint_id];

        q_lower_limits_[i]   = arm_model_.lowerPositionLimit[q_index];
        q_upper_limits_[i]   = arm_model_.upperPositionLimit[q_index];
        dq_velocity_limits_[i] = arm_model_.velocityLimit[v_index];

        RCLCPP_INFO(this->get_logger(),
            "PF joint limit from URDF: %s  q=[%.6f, %.6f]  velocity=%.6f",
            joint_names_[i].c_str(),
            q_lower_limits_[i], q_upper_limits_[i], dq_velocity_limits_[i]);
    }
}

// Returns per-DOF q stddev as fraction of joint range; falls back to the
// provided vector for joints without finite limits.
Eigen::Matrix<double, ArmPosePfNode::kDof, 1> ArmPosePfNode::jointLimitQStddev(
    double fraction,
    const Eigen::Matrix<double, kDof, 1>& fallback) const
{
    Eigen::Matrix<double, kDof, 1> stddev;
    const double clamped_fraction = std::max(0.0, fraction);
    for (int i = 0; i < kDof; ++i) {
        const double lower = q_lower_limits_[i];
        const double upper = q_upper_limits_[i];
        const double range = upper - lower;
        stddev[i] = (std::isfinite(lower) && std::isfinite(upper) && range > 0.0)
                    ? clamped_fraction * range
                    : std::max(0.0, fallback[i]);
    }
    return stddev;
}

void ArmPosePfNode::applyStateConstraints(const char* source)
{
    if (!enforce_joint_limits_) {
        return;
    }
    for (auto& particle : pf_.particles) {
        applyStateConstraintsToParticle(particle.x, false, source);
    }
    pf_.refreshEstimateFromParticles();
    applyStateConstraintsToParticle(pf_.x, true, source);
}

void ArmPosePfNode::applyStateConstraintsToParticle(
    Eigen::Matrix<double, kStateSize, 1>& state,
    bool warn,
    const char* source)
{
    for (int i = 0; i < kDof; ++i) {
        const double before = state[i];
        const double dq_before = state[kDof + i];
        const double lower = q_lower_limits_[i];
        const double upper = q_upper_limits_[i];
        const double velocity_limit = dq_velocity_limits_[i];

        if (std::isfinite(lower) && state[i] < lower) { state[i] = lower; }
        if (std::isfinite(upper) && state[i] > upper) { state[i] = upper; }

        double& dq = state[kDof + i];
        if (std::isfinite(velocity_limit) && velocity_limit > 0.0) {
            dq = std::clamp(dq, -velocity_limit, velocity_limit);
        }

        if (zero_velocity_at_limits_) {
            // Suppress outward velocity at active bounds only.
            const bool at_lower = std::isfinite(lower) && state[i] <= lower + 1.0e-9;
            const bool at_upper = std::isfinite(upper) && state[i] >= upper - 1.0e-9;
            if ((at_lower && dq < 0.0) || (at_upper && dq > 0.0)) {
                dq = 0.0;
            }
        }

        if (warn && before != state[i]) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Projected %s from %.6f to %.6f during %s. Limit=[%.6f, %.6f]",
                joint_names_[i].c_str(), before, state[i], source, lower, upper);
        }
        if (warn && dq_before != dq) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Projected velocity of %s from %.6f to %.6f during %s. Limit=+/-%.6f",
                joint_names_[i].c_str(), dq_before, dq, source, velocity_limit);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KINEMATICS
// ─────────────────────────────────────────────────────────────────────────────

// Maps the 7 independent PF DOFs (+ mimic joint) to the full Pinocchio q vector.
Eigen::VectorXd ArmPosePfNode::modelConfigurationFromOpt(const double* values) const
{
    Eigen::VectorXd q_model = Eigen::VectorXd::Zero(arm_model_.nq);
    for (int i = 0; i < kDof; ++i) {
        q_model[arm_model_.idx_qs[arm_joint_ids_[i]]] = arm_joint_multipliers_[i] * values[i];
    }
    q_model[arm_model_.idx_qs[mimic_joint_id_]] = mimic_joint_multiplier_ * values[0];
    return q_model;
}

Eigen::VectorXd ArmPosePfNode::modelConfigurationFromOpt(
    const Eigen::Matrix<double, kDof, 1>& values) const
{
    return modelConfigurationFromOpt(values.data());
}

// Projects the full Pinocchio Jacobian down to the 7 independent PF columns.
Eigen::Matrix<double, 6, Eigen::Dynamic> ArmPosePfNode::independentJacobian(
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& full_jacobian) const
{
    Eigen::Matrix<double, 6, Eigen::Dynamic> reduced(6, kDof);
    reduced.setZero();
    for (int i = 0; i < kDof; ++i) {
        reduced.col(i) =
            arm_joint_multipliers_[i] *
            full_jacobian.col(arm_model_.idx_vs[arm_joint_ids_[i]]);
    }
    reduced.col(0) +=
        mimic_joint_multiplier_ * full_jacobian.col(arm_model_.idx_vs[mimic_joint_id_]);
    return reduced;
}

void ArmPosePfNode::updateArmKinematics(
    const Eigen::Matrix<double, kDof, 1>& q,
    bool compute_jacobians)
{
    const Eigen::VectorXd q_model = modelConfigurationFromOpt(q);
    if (compute_jacobians) {
        pinocchio::computeJointJacobians(arm_model_, arm_data_, q_model);
    }
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_model);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);
}

// Runs robot (manipulator) FK and optionally computes the EE Jacobian for twist.
pinocchio::SE3 ArmPosePfNode::robotEndEffectorPose(
    const sensor_msgs::msg::JointState& msg,
    Eigen::Matrix<double, 6, 1>& ee_twist_world,
    bool compute_twist)
{
    Eigen::VectorXd q = Eigen::VectorXd::Map(msg.position.data(), robot_model_.nq);
    pinocchio::framesForwardKinematics(robot_model_, robot_data_, q);
    pinocchio::updateFramePlacements(robot_model_, robot_data_);

    ee_twist_world.setZero();
    if (compute_twist) {
        Eigen::VectorXd dq = Eigen::VectorXd::Map(msg.velocity.data(), robot_model_.nv);
        pinocchio::computeJointJacobians(robot_model_, robot_data_, q);
        Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, robot_model_.nv);
        J = pinocchio::getFrameJacobian(
            robot_model_, robot_data_, robot_ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
        ee_twist_world = J * dq;
    }
    return robot_data_.oMf[robot_ee_frame_id_];
}

// ─────────────────────────────────────────────────────────────────────────────
// TF LOOKUPS
// ─────────────────────────────────────────────────────────────────────────────

pinocchio::SE3 ArmPosePfNode::lookupShoulderToEndEffector(rclcpp::Time& tf_stamp) const
{
    const auto tf = tf_buffer_->lookupTransform(
        shoulder_frame_name_, ee_frame_name_, tf2::TimePointZero);

    pinocchio::SE3 T = pinocchio::SE3::Identity();
    T.translation() = Eigen::Vector3d(
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z);
    Eigen::Quaterniond q(
        tf.transform.rotation.w, tf.transform.rotation.x,
        tf.transform.rotation.y, tf.transform.rotation.z);
    T.rotation() = q.normalized().toRotationMatrix();
    tf_stamp = tf.header.stamp;
    return T;
}

pinocchio::SE3 ArmPosePfNode::lookupWorldToShoulder(const rclcpp::Time& stamp) const
{
    geometry_msgs::msg::TransformStamped tf;
    if (use_latest_shoulder_tf_) {
        tf = tf_buffer_->lookupTransform(world_frame_, shoulder_frame_name_, tf2::TimePointZero);
    } else {
        try {
            tf = tf_buffer_->lookupTransform(
                world_frame_, shoulder_frame_name_, stamp,
                rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));
        } catch (const tf2::TransformException& ex) {
            if (!fallback_to_latest_tf_) {
                throw;
            }
            tf = tf_buffer_->lookupTransform(
                world_frame_, shoulder_frame_name_, tf2::TimePointZero);
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Timestamped shoulder TF unavailable, using latest: %s", ex.what());
        }
    }

    pinocchio::SE3 T = pinocchio::SE3::Identity();
    T.translation() = Eigen::Vector3d(
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z);
    Eigen::Quaterniond q(
        tf.transform.rotation.w, tf.transform.rotation.x,
        tf.transform.rotation.y, tf.transform.rotation.z);
    T.rotation() = q.normalized().toRotationMatrix();
    return T;
}

// ─────────────────────────────────────────────────────────────────────────────
// PUBLISHING
// ─────────────────────────────────────────────────────────────────────────────

void ArmPosePfNode::publishState(const rclcpp::Time& stamp)
{
    // Final guardrail: project before publishing so robot_state_publisher never
    // receives out-of-limit angles even if a future code path publishes early.
    applyStateConstraints("publish");

    sensor_msgs::msg::JointState msg;
    msg.header.stamp = stamp;
    msg.name = joint_names_;
    const auto q = pf_.q();
    const auto dq = pf_.dq();
    msg.position.assign(q.data(), q.data() + kDof);
    msg.velocity.assign(dq.data(), dq.data() + kDof);
    state_pub_->publish(msg);
}

void ArmPosePfNode::publishGroundTruthHandTransform(
    const pinocchio::SE3& T_shoulder_hand,
    const rclcpp::Time& stamp) const
{
    geometry_msgs::msg::TransformStamped gt_transform;
    gt_transform.header.stamp = stamp;
    gt_transform.header.frame_id = shoulder_frame_name_;
    gt_transform.child_frame_id = ground_truth_hand_frame_;
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

// ─────────────────────────────────────────────────────────────────────────────
// MATH UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

double ArmPosePfNode::gaussianLikelihood(
    const Eigen::VectorXd& residual,
    const Eigen::MatrixXd& covariance)
{
    if (residual.size() == 0 ||
        covariance.rows() != residual.size() ||
        covariance.cols() != residual.size() ||
        !residual.allFinite() || !covariance.allFinite()) {
        return 0.0;
    }

    const Eigen::LDLT<Eigen::MatrixXd> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return 0.0;
    }
    const Eigen::VectorXd solved = solver.solve(residual);
    if (!solved.allFinite()) {
        return 0.0;
    }

    double determinant = 1.0;
    const auto& vector_d = solver.vectorD();
    for (int i = 0; i < vector_d.size(); ++i) {
        const double value = vector_d[i];
        if (!(value > 0.0) || !std::isfinite(value)) {
            return 0.0;
        }
        determinant *= value;
    }
    if (!(determinant > 0.0) || !std::isfinite(determinant)) {
        return 0.0;
    }

    const double mahalanobis = residual.dot(solved);
    if (!std::isfinite(mahalanobis) || mahalanobis < 0.0) {
        return 0.0;
    }

    constexpr double kTwoPi = 6.283185307179586;
    const double normalizer =
        1.0 / std::sqrt(
            std::pow(kTwoPi, static_cast<double>(residual.size())) * determinant);
    const double likelihood = normalizer * std::exp(-0.5 * mahalanobis);
    return std::isfinite(likelihood) ? likelihood : 0.0;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<ArmPosePfNode>();
        rclcpp::spin(node);
    } catch (const std::exception& ex) {
        RCLCPP_FATAL(rclcpp::get_logger("arm_pose_pf"),
            "Failed to start arm_pose_pf: %s", ex.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
