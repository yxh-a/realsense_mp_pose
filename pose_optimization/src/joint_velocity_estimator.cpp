#include "pose_optimization/joint_velocity_estimator.hpp"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

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

void VelocityEstimator::DQKalman::predict(double qa)
{
    if (!initialized) return;

    // Constant-velocity model
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2*n, 2*n);
    F.block(0, n, n, n) = dtc * Eigen::MatrixXd::Identity(n, n);

    const double dt   = dtc;
    const double dt2  = dt * dt;
    const double dt3  = dt2 * dt;

    // Process noise (continuous white acceleration model)
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(2*n, 2*n);
    const double s00 = (dt3 / 3.0) * qa;
    const double s01 = (dt2 / 2.0) * qa;
    const double s11 =  dt        * qa;

    Q.block(0,0, n,n).setIdentity();  Q.block(0,0, n,n) *= s00;
    Q.block(0,n, n,n).setIdentity();  Q.block(0,n, n,n) *= s01;
    Q.block(n,0, n,n).setIdentity();  Q.block(n,0, n,n) *= s01;
    Q.block(n,n, n,n).setIdentity();  Q.block(n,n, n,n) *= s11;

    // Propagate
    x.head(n) += dtc * x.tail(n);
    P = F * P * F.transpose() + Q;
}

void VelocityEstimator::DQKalman::correct(
    const Eigen::Matrix<double,6,Eigen::Dynamic>& J,
    const Eigen::Matrix<double,6,1>& v_meas,
    const Eigen::VectorXd& q_meas,
    const Eigen::VectorXd& dq_meas,
    double r_sigma_lin, double r_sigma_rot,
    double sigma_q, double sigma_dq)
{
    if (!initialized) return;

    // --- Twist rows: z_v = J*dq, H_v = [0 J]
    Eigen::MatrixXd H_v = Eigen::MatrixXd::Zero(6, 2*n);
    H_v.block(0, n, 6, n) = J;

    Eigen::VectorXd z_pred = J * x.tail(n);
    Eigen::VectorXd r_v    = v_meas - z_pred;

    Eigen::Matrix<double,6,6> R_v = Eigen::Matrix<double,6,6>::Zero();
    R_v.block(0,0,3,3).setIdentity(); R_v.block(0,0,3,3) *= (r_sigma_lin*r_sigma_lin);
    R_v.block(3,3,3,3).setIdentity(); R_v.block(3,3,3,3) *= (r_sigma_rot*r_sigma_rot);

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
    const int rows = 6 + n + n;
    Eigen::MatrixXd H(rows, 2*n);
    Eigen::VectorXd r(rows);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, rows);

    int o = 0;
    H.block(o, 0, 6, 2*n) = H_v;  r.segment(o, 6) = r_v;  R.block(o, o, 6, 6) = R_v;  o += 6;
    H.block(o, 0, n, 2*n) = H_q;  r.segment(o, n) = r_q;  R.block(o, o, n, n) = R_q;  o += n;
    H.block(o, 0, n, 2*n) = H_dq; r.segment(o, n) = r_dq; R.block(o, o, n, n) = R_dq; o += n;

    // --- Kalman update
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(rows, rows));

    x += K * r;
    P  = (I - K * H) * P;
}

Eigen::VectorXd VelocityEstimator::DQKalman::q()  const { return x.head(n); }
Eigen::VectorXd VelocityEstimator::DQKalman::dq() const { return x.tail(n); }

// -------------------- Node methods --------------------
VelocityEstimator::VelocityEstimator()
: rclcpp::Node("joint_velocity_estimator_node")
{
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

    std::string human_arm_urdf_path = ament_index_cpp::get_package_share_directory("image_pose_tracking")
                                    + "/config/right_arm.urdf";
    pinocchio::urdf::buildModel(human_arm_urdf_path, arm_model_);
    arm_data_ = pinocchio::Data(arm_model_);

    hand_frame_name_ = "RightHandCOM";
    hand_frame_id_ = arm_model_.getFrameId(hand_frame_name_);

    q_arm_   = Eigen::VectorXd::Zero(arm_model_.nq);
    dq_arm_  = Eigen::VectorXd::Zero(arm_model_.nv);
    dq_vis_  = Eigen::VectorXd::Zero(arm_model_.nv);

    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_arm_);

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

    if (config["velocity_estimation"]["kf"]) {
    q_q_         = config["velocity_estimation"]["kf"]["q_q"].as<double>(1e-2);
    q_dq_        = config["velocity_estimation"]["kf"]["q_dq"].as<double>(1e-2);
    q_a_         = config["velocity_estimation"]["kf"]["q_a"].as<double>(1e-1);
    rate_hz_     = config["velocity_estimation"]["rates"]["control_rate_hz"].as<double>(200.0);
    sigma_task_  = config["velocity_estimation"]["kf"]["sigma_task"].as<double>(0.5);
    sigma_null_  = config["velocity_estimation"]["kf"]["sigma_null"].as<double>(0.1);
    r_sigma_lin_ = config["velocity_estimation"]["kf"]["r_sigma_lin"].as<double>(0.01);
    r_sigma_rot_ = config["velocity_estimation"]["kf"]["r_sigma_rot"].as<double>(0.01);
    sigma_q_     = config["velocity_estimation"]["kf"]["sigma_q"].as<double>(0.1);
    sigma_dq_    = config["velocity_estimation"]["kf"]["sigma_dq"].as<double>(0.1);
    }

    if (config["velocity_estimation"]["output"]) {
    print_velocity_ = config["velocity_estimation"]["output"]["print_velocity"].as<bool>(true);
    print_sigmas_   = config["velocity_estimation"]["output"]["print_sigmas"].as<bool>(true);
    }

    kf_.init(model_.nv, 1.0/200.0, q_q_, q_dq_);


    // --- Shared initial state
    v_ee_            = Eigen::VectorXd::Zero(6);
    v_ee_in_shoulder_= Eigen::VectorXd::Zero(6);
    v_hand_          = Eigen::VectorXd::Zero(6);

    T_worldshoulder_ = pinocchio::SE3::Identity();
    T_shoulderhand_  = pinocchio::SE3::Identity();

    // --- TF listener
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
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
    J = pinocchio::getFrameJacobian(model_, data_, ee_frame_id_, pinocchio::WORLD);

    v_ee_ = J * dq;

    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header.stamp = msg->header.stamp;
    twist_msg.header.frame_id = "lbr_link_ee";
    twist_msg.twist.linear.x  = v_ee_[0];
    twist_msg.twist.linear.y  = v_ee_[1];
    twist_msg.twist.linear.z  = v_ee_[2];
    twist_msg.twist.angular.x = v_ee_[3];
    twist_msg.twist.angular.y = v_ee_[4];
    twist_msg.twist.angular.z = v_ee_[5];
    twist_pub_->publish(twist_msg);

    // TF lookup (latest)
    try {
        tf_shoulder2ee = tf_buffer_->lookupTransform(
        "RightShoulder",
        "lbr_link_ee",
        tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
        return;
    }

    T_worldshoulder_.translation() = Eigen::Vector3d(
        tf_shoulder2ee.transform.translation.x,
        tf_shoulder2ee.transform.translation.y,
        tf_shoulder2ee.transform.translation.z);

    Eigen::Quaterniond q_tf(
        tf_shoulder2ee.transform.rotation.w,
        tf_shoulder2ee.transform.rotation.x,
        tf_shoulder2ee.transform.rotation.y,
        tf_shoulder2ee.transform.rotation.z);
    T_worldshoulder_.rotation() = q_tf.toRotationMatrix();

    Ad_T = T_worldshoulder_.inverse().toActionMatrix();
    if (!isValidMatrix(Ad_T)) {
        RCLCPP_ERROR(this->get_logger(), "Invalid action matrix (NaNs detected)");
        return;
    }
    v_ee_in_shoulder_ = Ad_T * v_ee_;

    Ad_T = T_eehand_.toActionMatrix();
    if (!isValidMatrix(Ad_T)) {
        RCLCPP_ERROR(this->get_logger(), "Invalid action matrix (NaNs detected)");
        return;
    }
    v_hand_ = Ad_T * v_ee_in_shoulder_;

    // Arm Jacobian
    pinocchio::computeJointJacobians(arm_model_, arm_data_, q_arm_);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_arm_);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_arm(6, arm_model_.nv);
    J_arm.setZero();
    J_arm = pinocchio::getFrameJacobian(arm_model_, arm_data_, hand_frame_id_, pinocchio::WORLD);
    if (!isValidMatrix(J_arm)) {
        RCLCPP_ERROR(this->get_logger(), "Invalid arm Jacobian (NaNs detected)");
        return;
    }

    kf_.predict(q_a_);
    now_ = this->now().seconds();
    double latency = now_ - t_vis_;
    double scale = std::max(1.0, latency / 0.05);

    kf_.correct(J_arm, v_hand_, q_arm_, dq_vis_,
                r_sigma_lin_, r_sigma_rot_,
                sigma_q_ * scale, sigma_dq_ * scale);

    dq_arm_ = kf_.dq();

    if (print_velocity_) {
        RCLCPP_INFO(this->get_logger(), "Hand velocity: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
                    v_hand_[0], v_hand_[1], v_hand_[2], v_hand_[3], v_hand_[4], v_hand_[5]);
        RCLCPP_INFO(this->get_logger(), "Estimated arm joint velocities (deg): %f, %f, %f, %f, %f, %f, %f",
                    dq_arm_[0]*180.0/M_PI, dq_arm_[1]*180.0/M_PI, dq_arm_[2]*180.0/M_PI,
                    dq_arm_[3]*180.0/M_PI, dq_arm_[4]*180.0/M_PI, dq_arm_[5]*180.0/M_PI, dq_arm_[6]*180.0/M_PI);
    }

    // Publish updated
    sensor_msgs::msg::JointState updated_arm_joint_state;
    updated_arm_joint_state.header.stamp = this->now();
    updated_arm_joint_state.name = joint_names_;

    Eigen::VectorXd q_arm_updated = kf_.q();
    updated_arm_joint_state.position =
        std::vector<double>(q_arm_updated.data(), q_arm_updated.data() + q_arm_updated.size());
    updated_arm_joint_state.velocity =
        std::vector<double>(dq_arm_.data(), dq_arm_.data() + dq_arm_.size());
    updated_arm_joint_states_pub_->publish(updated_arm_joint_state);
}

void VelocityEstimator::jointCallback_arm(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (msg->name.size() != static_cast<size_t>(arm_model_.nq) ||
        msg->position.size() != static_cast<size_t>(arm_model_.nq))
    {
        RCLCPP_WARN(this->get_logger(), "Arm joint state size mismatch with model.");
        return;
    }

    q_arm_  = Eigen::VectorXd::Map(msg->position.data(), arm_model_.nq);
    dq_vis_ = Eigen::VectorXd::Map(msg->velocity.data(), arm_model_.nv);
    t_vis_  = this->now().seconds();

    pinocchio::forwardKinematics(arm_model_, arm_data_, q_arm_);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);

    if (!first_vis_) {
        first_vis_ = true;
        kf_.set_state(q_arm_, dq_vis_);
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
    auto node = std::make_shared<VelocityEstimator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}