#include "pose_optimization/joint_velocity_estimator.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace std::chrono_literals;

namespace {

std::string render_arm_xacro(
    const std::string &robot_prefix,
    const std::string &robot_color,
    double upper_arm_length,
    double forearm_length)
{
    const std::string xacro_path =
        ament_index_cpp::get_package_share_directory("image_pose_tracking") + "/config/right_arm_osim_shoulder.urdf.xacro";

    char temp_template[] = "/tmp/right_arm_osim_shoulder_velocity_XXXXXX.urdf";
    const int fd = mkstemps(temp_template, 5);
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary URDF path");
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
        throw std::runtime_error("Failed to render right_arm_osim_shoulder.urdf.xacro");
    }

    return temp_template;
}

}  // namespace

// -------------------- DQKalman methods --------------------
void VelocityEstimator::DQKalman::init(int n_, double dtc_, double Pq0, double Pdq0)
{
    n   = n_;
    dtc = dtc_;
    x   = Eigen::VectorXd::Zero(2*n);
    P   = Eigen::MatrixXd::Zero(2*n, 2*n);
    I   = Eigen::MatrixXd::Identity(2*n, 2*n);

    P.block(0, 0, n, n).setIdentity();
    P.block(0, 0, n, n) *= Pq0;

    P.block(n, n, n, n).setIdentity();
    P.block(n, n, n, n) *= Pdq0;

    initialized = true;
}

void VelocityEstimator::DQKalman::set_state(const Eigen::VectorXd& q0, const Eigen::VectorXd& dq0)
{
    x.head(n) = q0;
    x.tail(n) = dq0;
}

void VelocityEstimator::DQKalman::predict(double qa, double velocity_decay, double acc_cap, double dt)
{
    if (!initialized) return;

    const double step = (dt > 0.0) ? dt : dtc;
    const double decay = std::clamp(velocity_decay, 0.0, 1.0e6);
    const double alpha = std::exp(-decay * step);

    // Damped-velocity model: dq decays smoothly toward zero, which matches
    // stop-start human motion better than a pure constant-velocity prior.
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2*n, 2*n);
    F.block(0, n, n, n) = step * alpha * Eigen::MatrixXd::Identity(n, n);
    F.block(n, n, n, n) = alpha * Eigen::MatrixXd::Identity(n, n);

    const double dt2  = step * step;
    const double dt3  = dt2 * step;

    // Process noise (continuous white acceleration model)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2*n, 2*n);
    const double s00 = (dt3 / 3.0) * qa;
    const double s01 = (dt2 / 2.0) * qa;
    const double s11 =  step      * qa;

    Q.block(0,0, n,n).setIdentity();  Q.block(0,0, n,n) *= s00;
    Q.block(0,n, n,n).setIdentity();  Q.block(0,n, n,n) *= s01;
    Q.block(n,0, n,n).setIdentity();  Q.block(n,0, n,n) *= s01;
    Q.block(n,n, n,n).setIdentity();  Q.block(n,n, n,n) *= s11;

    Eigen::VectorXd dq_pred = alpha * x.tail(n);
    if (acc_cap > 0.0) {
        const double max_dq_change = acc_cap * step;
        for (int i = 0; i < n; ++i) {
            const double delta = dq_pred[i] - x[n + i];
            dq_pred[i] = x[n + i] + std::clamp(delta, -max_dq_change, max_dq_change);
        }
    }

    // Propagate
    x.head(n) += step * dq_pred;
    x.tail(n) = dq_pred;
    P = F * P * F.transpose() + Q;
}

void VelocityEstimator::DQKalman::correct_task_velocity(
    const Eigen::Matrix<double,6,Eigen::Dynamic>& J,
    const Eigen::Matrix<double,6,1>& v_meas,
    double r_sigma_lin, double r_sigma_rot)
{
    if (!initialized) return;

    // --- Twist rows: z_v = J*dq, H_v = [0 J]
    Eigen::MatrixXd H_v = Eigen::MatrixXd::Zero(6, 2*n);
    H_v.block(0, n, 6, n) = J;

    Eigen::Matrix<double,6,6> R_v = Eigen::Matrix<double,6,6>::Zero();
    R_v.block(0,0,3,3).setIdentity(); R_v.block(0,0,3,3) *= (r_sigma_lin*r_sigma_lin);
    R_v.block(3,3,3,3).setIdentity(); R_v.block(3,3,3,3) *= (r_sigma_rot*r_sigma_rot);

    Eigen::VectorXd r_v = v_meas - J * x.tail(n);
    Eigen::MatrixXd S = H_v * P * H_v.transpose() + R_v;
    Eigen::MatrixXd K = P * H_v.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(6, 6));

    x += K * r_v;
    const Eigen::MatrixXd IKH = I - K * H_v;
    P = IKH * P * IKH.transpose() + K * R_v * K.transpose();
    P = 0.5 * (P + P.transpose());
}

void VelocityEstimator::DQKalman::correct_joint_measurement(
    const Eigen::VectorXd& q_meas,
    const Eigen::VectorXd& dq_meas,
    double sigma_q,
    double sigma_dq)
{
    if (!initialized) return;

    // --- Vision q rows: z_q = q, H_q = [I 0]
    Eigen::MatrixXd H_q = Eigen::MatrixXd::Zero(n, 2*n);
    H_q.block(0, 0, n, n).setIdentity();
    Eigen::VectorXd r_q = q_meas - x.head(n);
    Eigen::MatrixXd R_q = (sigma_q*sigma_q) * Eigen::MatrixXd::Identity(n,n);

    // --- Vision dq rows: z_dq = dq, H_dq = [0 I]
    Eigen::MatrixXd H_dq = Eigen::MatrixXd::Zero(n, 2*n);
    H_dq.block(0, n, n, n).setIdentity();
    Eigen::VectorXd r_dq = dq_meas - x.tail(n);
    Eigen::MatrixXd R_dq = (sigma_dq*sigma_dq) * Eigen::MatrixXd::Identity(n,n);

    // --- Stack
    const int rows = n + n;
    Eigen::MatrixXd H(rows, 2*n);
    Eigen::VectorXd r(rows);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, rows);

    int o = 0;
    H.block(o, 0, n, 2*n) = H_q;  r.segment(o, n) = r_q;  R.block(o, o, n, n) = R_q;  o += n;
    H.block(o, 0, n, 2*n) = H_dq; r.segment(o, n) = r_dq; R.block(o, o, n, n) = R_dq; o += n;

    // --- Kalman update
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(rows, rows));

    x += K * r;
    const Eigen::MatrixXd IKH = I - K * H;
    P = IKH * P * IKH.transpose() + K * R * K.transpose();
    P = 0.5 * (P + P.transpose());
}

Eigen::VectorXd VelocityEstimator::DQKalman::q()  const { return x.head(n); }
Eigen::VectorXd VelocityEstimator::DQKalman::dq() const { return x.tail(n); }

// -------------------- Node methods --------------------
VelocityEstimator::VelocityEstimator()
: rclcpp::Node("joint_velocity_estimator_node")
{
    this->declare_parameter<std::string>("robot_prefix", "upt_");
    this->declare_parameter<std::string>("robot_color", "blue");
    this->declare_parameter<double>("upper_arm_length", 0.299);
    this->declare_parameter<double>("forearm_length", 0.248);

    const auto robot_prefix = this->get_parameter("robot_prefix").as_string();
    const auto robot_color = this->get_parameter("robot_color").as_string();
    const auto upper_arm_length = this->get_parameter("upper_arm_length").as_double();
    const auto forearm_length = this->get_parameter("forearm_length").as_double();

    RCLCPP_INFO(this->get_logger(), "Initializing robot model...");

    // --- Robot model
    std::string urdf_path = ament_index_cpp::get_package_share_directory("lbr_description")
                          + "/urdf/iiwa7/iiwa7.urdf";
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);

    q_  = Eigen::VectorXd::Zero(model_.nq);
    dq_ = Eigen::VectorXd::Zero(model_.nv);

    pinocchio::framesForwardKinematics(model_, data_, q_);

    RCLCPP_INFO(this->get_logger(), "Model loaded: %s", model_.name.c_str());

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/lbr/joint_states", 10,
      std::bind(&VelocityEstimator::jointCallback, this, std::placeholders::_1));

    twist_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/lbr/ee_velocity", 10);

    ee_frame_name_ = "lbr_link_ee";
    ee_frame_id_ = model_.getFrameId(ee_frame_name_);

  // --- Human arm model
    RCLCPP_INFO(this->get_logger(), "Initializing human arm model...");

    std::string human_arm_urdf_path = render_arm_xacro(
        robot_prefix, robot_color, upper_arm_length, forearm_length);
    pinocchio::urdf::buildModel(human_arm_urdf_path, arm_model_);
    arm_data_ = pinocchio::Data(arm_model_);
    std::remove(human_arm_urdf_path.c_str());

    joint_names_ = {
        robot_prefix + "jRightShoulder_elv_angle",
        robot_prefix + "jRightShoulder_shoulder_elv",
        robot_prefix + "jRightShoulder_shoulder_rot",
        robot_prefix + "jRightElbow_rotz",
        robot_prefix + "jRightElbow_roty",
        robot_prefix + "jRightWrist_rotx",
        robot_prefix + "jRightWrist_rotz"
    };

    for (std::size_t i = 0; i < kArmOptDof; ++i) {
        arm_opt_joint_ids_[i] = arm_model_.getJointId(joint_names_[i]);
        if (arm_opt_joint_ids_[i] >= static_cast<pinocchio::JointIndex>(arm_model_.njoints)) {
            throw std::runtime_error("Joint not found in right_arm_osim_shoulder model: " + joint_names_[i]);
        }
        arm_opt_joint_multipliers_[i] = 1.0;
    }

    arm_mimic_joint_id_ = arm_model_.getJointId(robot_prefix + "jRightShoulder_shoulder1_r2");
    if (arm_mimic_joint_id_ >= static_cast<pinocchio::JointIndex>(arm_model_.njoints)) {
        throw std::runtime_error("Mimic joint not found in right_arm_osim_shoulder model");
    }

    shoulder_frame_name_ = robot_prefix + "RightShoulder";
    hand_frame_name_ = robot_prefix + "RightHandCOM";
    hand_frame_id_ = arm_model_.getFrameId(hand_frame_name_);

    q_arm_   = Eigen::VectorXd::Zero(kArmOptDof);
    dq_arm_  = Eigen::VectorXd::Zero(kArmOptDof);
    dq_vis_  = Eigen::VectorXd::Zero(kArmOptDof);

    pinocchio::framesForwardKinematics(arm_model_, arm_data_, armModelConfigurationFromOpt(q_arm_));

    arm_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/optimized_arm/joint_states", 10,
      std::bind(&VelocityEstimator::jointCallback_arm, this, std::placeholders::_1));

    updated_arm_joint_states_pub_ =
      this->create_publisher<sensor_msgs::msg::JointState>("/updated_arm/joint_states", 10);

    RCLCPP_INFO(this->get_logger(), "Human arm model loaded: %s", arm_model_.name.c_str());


    // --- Load hand-to-EE transform (YAML)
    RCLCPP_INFO(this->get_logger(), "Loading hand-to-EE transform from YAML");
    std::string config_path = ament_index_cpp::get_package_share_directory("pose_optimization")
                            + "/config/parameters.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    if (!config["ee2hand"]) {
    RCLCPP_ERROR(this->get_logger(), "ee2hand configuration not found in %s", config_path.c_str());
    throw std::runtime_error("Missing ee2hand in YAML");
    }

    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    if (config["ee2hand"]["translation"]) {
    translation = Eigen::Vector3d(
        config["ee2hand"]["translation"][0].as<double>(),
        config["ee2hand"]["translation"][1].as<double>(),
        config["ee2hand"]["translation"][2].as<double>());
    }

    Eigen::Vector4d rotation = Eigen::Vector4d::Zero();
    if (config["ee2hand"]["rotation"]) {
    rotation = Eigen::Vector4d(
        config["ee2hand"]["rotation"][0].as<double>(),
        config["ee2hand"]["rotation"][1].as<double>(),
        config["ee2hand"]["rotation"][2].as<double>(),
        config["ee2hand"]["rotation"][3].as<double>());
    }

    T_eehand_ = pinocchio::SE3::Identity();
    Eigen::Quaterniond q(rotation[3], rotation[0], rotation[1], rotation[2]);
    T_eehand_.rotation() = q.toRotationMatrix();
    T_eehand_.translation() = translation;

    T_handee_ = T_eehand_.inverse();

    if (config["sensitivity_analysis"]) {
    apply_sensitivity = config["sensitivity_analysis"]["apply_sensitivity"].as<bool>(false);
    sigma_sh_x = config["sensitivity_analysis"]["sigma_sh_x"].as<double>(0.0);
    sigma_sh_y = config["sensitivity_analysis"]["sigma_sh_y"].as<double>(0.0);
    sigma_sh_z = config["sensitivity_analysis"]["sigma_sh_z"].as<double>(0.0);
    }

    // --- Kalman filter config
    if (!config["velocity_estimation"]) {
    RCLCPP_ERROR(this->get_logger(), "parameters not found in %s", config_path.c_str());
    throw std::runtime_error("Missing parameters in KF YAML");
    }

    use_kalman_filter_ = config["velocity_estimation"]["use_kalman_filter"].as<bool>(true);
    if (config["velocity_estimation"]["timing"]) {
    const auto timing_config = config["velocity_estimation"]["timing"];
    tf_lookup_timeout_sec_ = timing_config["tf_lookup_timeout_sec"].as<double>(0.05);
    fallback_to_latest_tf_ = timing_config["fallback_to_latest_tf"].as<bool>(true);
    }

    if (config["velocity_estimation"]["kf"]) {
    const auto kf_config = config["velocity_estimation"]["kf"];
    q_q_         = kf_config["q_q"].as<double>(1e-2);
    q_dq_        = kf_config["q_dq"].as<double>(1e-2);
    q_a_         = kf_config["q_a"].as<double>(1e-1);
    velocity_decay_ = kf_config["velocity_decay"].as<double>(0.0);
    acc_cap_     = kf_config["acc_cap"].as<double>(10.0);
    r_sigma_lin_ = kf_config["r_sigma_lin"].as<double>(0.01);
    r_sigma_rot_ = kf_config["r_sigma_rot"].as<double>(0.01);
    sigma_q_     = kf_config["sigma_q"].as<double>(0.1);
    sigma_dq_    = kf_config["sigma_dq"].as<double>(0.1);

    if (kf_config["control_rate_hz"]) {
        rate_hz_ = kf_config["control_rate_hz"].as<double>(200.0);
    } else {
        RCLCPP_WARN(this->get_logger(), "control_rate_hz not found in KF config, using default %.1f Hz", rate_hz_);
    }
    }

    if (config["velocity_estimation"]["output"]) {
    print_velocity_ = config["velocity_estimation"]["output"]["print_velocity"].as<bool>(true);
    print_sigmas_   = config["velocity_estimation"]["output"]["print_sigmas"].as<bool>(true);
    }

    kf_.init(kArmOptDof, 1.0 / rate_hz_, q_q_, q_dq_);


    // --- Shared initial state
    v_ee_in_world_ = Eigen::VectorXd::Zero(6);
    v_hand_in_world_ = Eigen::VectorXd::Zero(6);
    v_hand_in_shoulder_ = Eigen::VectorXd::Zero(6);

    T_worldshoulder_ = pinocchio::SE3::Identity();
    T_shoulderhand_  = pinocchio::SE3::Identity();

    tf_world2shoulder = geometry_msgs::msg::TransformStamped();

    // --- TF listener
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

Eigen::VectorXd VelocityEstimator::armModelConfigurationFromOpt(const double* values) const
{
    Eigen::VectorXd q_model = Eigen::VectorXd::Zero(arm_model_.nq);
    for (std::size_t i = 0; i < kArmOptDof; ++i) {
        q_model[arm_model_.idx_qs[arm_opt_joint_ids_[i]]] = arm_opt_joint_multipliers_[i] * values[i];
    }
    q_model[arm_model_.idx_qs[arm_mimic_joint_id_]] = arm_mimic_joint_multiplier_ * values[0];
    return q_model;
}

Eigen::VectorXd VelocityEstimator::armModelConfigurationFromOpt(const Eigen::VectorXd& values) const
{
    if (values.size() < static_cast<Eigen::Index>(kArmOptDof)) {
        throw std::runtime_error("Not enough independent joint values for right_arm_osim_shoulder model");
    }
    return armModelConfigurationFromOpt(values.data());
}

Eigen::Matrix<double, 6, Eigen::Dynamic> VelocityEstimator::independentArmJacobian(
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& full_jacobian) const
{
    Eigen::Matrix<double, 6, Eigen::Dynamic> reduced_jacobian(6, kArmOptDof);
    reduced_jacobian.setZero();

    for (std::size_t i = 0; i < kArmOptDof; ++i) {
        reduced_jacobian.col(i) =
            arm_opt_joint_multipliers_[i] * full_jacobian.col(arm_model_.idx_vs[arm_opt_joint_ids_[i]]);
    }
    reduced_jacobian.col(0) +=
        arm_mimic_joint_multiplier_ * full_jacobian.col(arm_model_.idx_vs[arm_mimic_joint_id_]);

    return reduced_jacobian;
}

void VelocityEstimator::jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->position.size() != static_cast<size_t>(model_.nq) ||
        msg->velocity.size() != static_cast<size_t>(model_.nv))
    {
        RCLCPP_WARN(this->get_logger(), "Joint state size mismatch with model.");
        return;
    }

    Eigen::VectorXd q  = Eigen::VectorXd::Map(msg->position.data(), model_.nq);
    Eigen::VectorXd dq = Eigen::VectorXd::Map(msg->velocity.data(), model_.nv);

    pinocchio::computeJointJacobians(model_, data_, q);
    pinocchio::framesForwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    T_worldee_ = data_.oMf[ee_frame_id_];
    if (!isValidTransform(T_worldee_)) {
        RCLCPP_ERROR(this->get_logger(), "Invalid end effector transform (NaNs detected)");
        return;
    }

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_.nv);
    J.setZero();
    J = pinocchio::getFrameJacobian(model_, data_, ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);

    v_ee_in_world_ = J * dq;

    // Prefer timestamp-aligned TF. During rosbag replay, TF can lag the joint
    // message by a few milliseconds, so wait briefly before falling back.
    try {
        tf_world2shoulder = tf_buffer_->lookupTransform(
        "lbr_link_0",
        shoulder_frame_name_,
        msg->header.stamp,
        rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));
    } catch (const tf2::TransformException &ex) {
        if (!fallback_to_latest_tf_) {
            RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
            return;
        }

        try {
            tf_world2shoulder = tf_buffer_->lookupTransform(
            "lbr_link_0",
            shoulder_frame_name_,
            tf2::TimePointZero);
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Timestamped TF unavailable, using latest TF: %s", ex.what());
        } catch (const tf2::TransformException &latest_ex) {
            RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", latest_ex.what());
            return;
        }
    }

    T_worldshoulder_.translation() = Eigen::Vector3d(
        tf_world2shoulder.transform.translation.x,
        tf_world2shoulder.transform.translation.y,
        tf_world2shoulder.transform.translation.z);

    Eigen::Quaterniond q_tf(
        tf_world2shoulder.transform.rotation.w,
        tf_world2shoulder.transform.rotation.x,
        tf_world2shoulder.transform.rotation.y,
        tf_world2shoulder.transform.rotation.z);
    T_worldshoulder_.rotation() = q_tf.toRotationMatrix();

    const Eigen::Vector3d ee_to_hand_world =
        T_worldee_.rotation() * T_eehand_.translation();
    const Eigen::Vector3d linear_ee_world = v_ee_in_world_.head<3>();
    const Eigen::Vector3d angular_world = v_ee_in_world_.tail<3>();

    v_hand_in_world_.head<3>() = linear_ee_world + angular_world.cross(ee_to_hand_world);
    v_hand_in_world_.tail<3>() = angular_world;

    const Eigen::Matrix3d R_shoulder_world = T_worldshoulder_.rotation().transpose();
    v_hand_in_shoulder_.head<3>() = R_shoulder_world * v_hand_in_world_.head<3>();
    v_hand_in_shoulder_.tail<3>() = R_shoulder_world * v_hand_in_world_.tail<3>();

    // print out the comparison of velocity euclidean norm in different frames for debugging
    if (print_velocity_) {
        double v_ee_norm = v_ee_in_world_.norm();
        double v_ee_shoulder_norm = v_hand_in_world_.norm();
        double v_hand_norm = v_hand_in_shoulder_.norm();
        RCLCPP_INFO(this->get_logger(), "Velocity norms: v_ee=%.3f, v_ee_in_shoulder=%3f, v_hand=%3f", v_ee_norm, v_ee_shoulder_norm, v_hand_norm); 
    
    }
    
    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header.stamp = msg->header.stamp;
    twist_msg.header.frame_id = shoulder_frame_name_;
    twist_msg.twist.linear.x  = v_hand_in_shoulder_[0];
    twist_msg.twist.linear.y  = v_hand_in_shoulder_[1];
    twist_msg.twist.linear.z  = v_hand_in_shoulder_[2];
    twist_msg.twist.angular.x = v_hand_in_shoulder_[3];
    twist_msg.twist.angular.y = v_hand_in_shoulder_[4];
    twist_msg.twist.angular.z = v_hand_in_shoulder_[5];
    twist_pub_->publish(twist_msg);


    const Eigen::VectorXd q_for_jacobian = use_kalman_filter_ ? kf_.q() : q_arm_;
    if (!q_for_jacobian.allFinite()) {
        RCLCPP_ERROR(this->get_logger(), "q_for_jacobian contains non-finite values");
        return;
    }
    const Eigen::VectorXd q_model_for_jacobian = armModelConfigurationFromOpt(q_for_jacobian);

    // Arm Jacobian. For KF mode, evaluate at the filter state, not the latest measurement.
    pinocchio::computeJointJacobians(arm_model_, arm_data_, q_model_for_jacobian);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_model_for_jacobian);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_arm_full(6, arm_model_.nv);
    J_arm_full.setZero();
    J_arm_full = pinocchio::getFrameJacobian(
        arm_model_, arm_data_, hand_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
    const Eigen::Matrix<double, 6, Eigen::Dynamic> J_arm = independentArmJacobian(J_arm_full);
    if (!isValidMatrix(J_arm)) {
        RCLCPP_ERROR(this->get_logger(), "Invalid arm Jacobian (NaNs detected)");
        return;
    }

    Eigen::VectorXd q_arm_updated = q_arm_;
    if (use_kalman_filter_) {
        if (!kf_.q().allFinite() || !kf_.dq().allFinite()) {
            RCLCPP_ERROR(this->get_logger(),
                         "Kalman state is non-finite before predict. q_finite=%d dq_finite=%d",
                         kf_.q().allFinite(), kf_.dq().allFinite());
            return;
        }

        double predict_dt = 1.0 / rate_hz_;
        const rclcpp::Time current_robot_stamp(msg->header.stamp);
        if (first_robot_) {
            const double stamp_dt = (current_robot_stamp - last_robot_stamp_).seconds();
            if (stamp_dt > 0.0 && stamp_dt < 1.0) {
                predict_dt = stamp_dt;
            }
        } else {
            first_robot_ = true;
        }
        last_robot_stamp_ = current_robot_stamp;

        kf_.predict(q_a_, velocity_decay_, acc_cap_, predict_dt);
        if (!kf_.q().allFinite() || !kf_.dq().allFinite()) {
            RCLCPP_ERROR(this->get_logger(),
                         "Kalman state became non-finite after predict. q_finite=%d dq_finite=%d",
                         kf_.q().allFinite(), kf_.dq().allFinite());
            return;
        }

        kf_.correct_task_velocity(J_arm, v_hand_in_shoulder_,
                                  r_sigma_lin_, r_sigma_rot_);
        if (!kf_.q().allFinite() || !kf_.dq().allFinite()) {
            RCLCPP_ERROR(this->get_logger(),
                         "Kalman state became non-finite after task-velocity correction. q_finite=%d dq_finite=%d",
                         kf_.q().allFinite(), kf_.dq().allFinite());
            return;
        }

        if (new_vis_measurement_) {
            kf_.correct_joint_measurement(q_arm_, dq_vis_,
                                          sigma_q_,
                                          sigma_dq_);
            if (!kf_.q().allFinite() || !kf_.dq().allFinite()) {
                RCLCPP_ERROR(this->get_logger(),
                             "Kalman state became non-finite after joint correction. q_finite=%d dq_finite=%d",
                             kf_.q().allFinite(), kf_.dq().allFinite());
                return;
            }
            new_vis_measurement_ = false;
        }

        dq_arm_ = kf_.dq();
        q_arm_updated = kf_.q();
    } else {
        dq_arm_ = dq_vis_;
    }

    if (print_velocity_) {
        RCLCPP_INFO(this->get_logger(), "Hand velocity: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
                    v_hand_in_shoulder_[0], v_hand_in_shoulder_[1], v_hand_in_shoulder_[2], v_hand_in_shoulder_[3], v_hand_in_shoulder_[4], v_hand_in_shoulder_[5]);
        RCLCPP_INFO(this->get_logger(), "%s arm joint velocities (deg): %f, %f, %f, %f, %f, %f, %f",
                    use_kalman_filter_ ? "Estimated" : "Measured",
                    dq_arm_[0]*180.0/M_PI, dq_arm_[1]*180.0/M_PI, dq_arm_[2]*180.0/M_PI,
                    dq_arm_[3]*180.0/M_PI, dq_arm_[4]*180.0/M_PI, dq_arm_[5]*180.0/M_PI, dq_arm_[6]*180.0/M_PI);
    }

    // Publish updated
    sensor_msgs::msg::JointState updated_arm_joint_state;
    updated_arm_joint_state.header.stamp = msg->header.stamp;
    updated_arm_joint_state.name = joint_names_;

    updated_arm_joint_state.position =
        std::vector<double>(q_arm_updated.data(), q_arm_updated.data() + q_arm_updated.size());
    updated_arm_joint_state.velocity =
        std::vector<double>(dq_arm_.data(), dq_arm_.data() + dq_arm_.size());
    updated_arm_joint_states_pub_->publish(updated_arm_joint_state);
}

void VelocityEstimator::jointCallback_arm(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->name.size() != kArmOptDof ||
        msg->position.size() != kArmOptDof ||
        msg->velocity.size() != kArmOptDof)
    {
        RCLCPP_WARN(this->get_logger(), "Arm joint state size mismatch with model.");
        return;
    }

    q_arm_  = Eigen::VectorXd::Map(msg->position.data(), kArmOptDof);
    dq_vis_ = Eigen::VectorXd::Map(msg->velocity.data(), kArmOptDof);
    if (!q_arm_.allFinite() || !dq_vis_.allFinite()) {
        RCLCPP_ERROR(this->get_logger(),
                     "Received non-finite arm measurement. q_finite=%d dq_finite=%d",
                     q_arm_.allFinite(), dq_vis_.allFinite());
        return;
    }
    last_vis_stamp_ = msg->header.stamp;
    t_vis_  = last_vis_stamp_.seconds();
    new_vis_measurement_ = true;

    pinocchio::forwardKinematics(arm_model_, arm_data_, armModelConfigurationFromOpt(q_arm_));
    pinocchio::updateFramePlacements(arm_model_, arm_data_);

    if (!first_vis_) {
        first_vis_ = true;
        if (use_kalman_filter_) {
            kf_.set_state(q_arm_, dq_vis_);
        }
        return;
    }
}

bool VelocityEstimator::isValidTransform(const pinocchio::SE3 &T)
{
    return T.rotation().allFinite() && T.translation().allFinite();
}

bool VelocityEstimator::isValidMatrix(const Eigen::MatrixXd &M)
{
    return M.allFinite();
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<VelocityEstimator>();
        rclcpp::spin(node);
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception &ex) {
        RCLCPP_FATAL(rclcpp::get_logger("joint_velocity_estimator"),
                     "Failed to start joint_velocity_estimator: %s", ex.what());
    } catch (...) {
        RCLCPP_FATAL(rclcpp::get_logger("joint_velocity_estimator"),
                     "Failed to start joint_velocity_estimator: unknown exception");
    }

    rclcpp::shutdown();
    return 1;
}
