#include "pose_optimization/optimizer.hpp"

#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/spatial/log.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace {

std::string render_arm_xacro(
    const std::string &robot_prefix,
    const std::string &robot_color,
    double upper_arm_length,
    double forearm_length)
{
    const std::string xacro_path =
        ament_index_cpp::get_package_share_directory("image_pose_tracking") + "/config/right_arm_osim_shoulder.urdf.xacro";

    char temp_template[] = "/tmp/right_arm_osim_shoulder_optimizer_XXXXXX.urdf";
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

PoseOptimizer::PoseOptimizer()
    : Node("pose_optimizer"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
{   
    this->declare_parameter<std::string>("robot_prefix", "upt_");
    this->declare_parameter<std::string>("robot_color", "blue");
    this->declare_parameter<double>("upper_arm_length", 0.299);
    this->declare_parameter<double>("forearm_length", 0.248);

    const auto robot_prefix = this->get_parameter("robot_prefix").as_string();
    const auto robot_color = this->get_parameter("robot_color").as_string();
    const auto upper_arm_length = this->get_parameter("upper_arm_length").as_double();
    const auto forearm_length = this->get_parameter("forearm_length").as_double();
    shoulder_frame_name_ = robot_prefix + "RightShoulder";
    hand_frame_name_ = robot_prefix + "RightHandCOM";

    // Load the URDF model into Pinocchio
    RCLCPP_INFO(this->get_logger(), "Loading robot model...");
    std::string urdf_path = render_arm_xacro(
        robot_prefix, robot_color, upper_arm_length, forearm_length);

    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);

    joint_names_ = {
        robot_prefix + "jRightShoulder_elv_angle",
        robot_prefix + "jRightShoulder_shoulder_elv",
        robot_prefix + "jRightShoulder_shoulder_rot",
        robot_prefix + "jRightElbow_rotz",
        robot_prefix + "jRightElbow_roty",
        robot_prefix + "jRightWrist_rotx",
        robot_prefix + "jRightWrist_rotz"
    };

    for (std::size_t i = 0; i < kOptDof; ++i) {
        opt_joint_ids_[i] = model_.getJointId(joint_names_[i]);
        if (opt_joint_ids_[i] >= static_cast<pinocchio::JointIndex>(model_.njoints)) {
            throw std::runtime_error("Joint not found in right_arm_osim_shoulder model: " + joint_names_[i]);
        }
        opt_joint_multipliers_[i] = 1.0;
    }

    mimic_joint_id_ = model_.getJointId(robot_prefix + "jRightShoulder_shoulder1_r2");
    if (mimic_joint_id_ >= static_cast<pinocchio::JointIndex>(model_.njoints)) {
        throw std::runtime_error("Mimic joint not found in right_arm_osim_shoulder model");
    }


    q = Eigen::VectorXd::Zero(model_.nq);
    q_init_ = q; // Initialize with zero joint angles
    q_prev_ = q;
    q_prev2_ = q;
    have_prev_ = false;
    have_prev2_ = false;
    
    RCLCPP_INFO(this->get_logger(), "Robot model loaded with %d DOF", model_.nq);
    

    // add a "opt_" prefix to joint names for optimization variables
    opt_joint_names_.reserve(joint_names_.size());
    for (const auto &name : joint_names_)
        opt_joint_names_.push_back("opt_" + name);

    RCLCPP_INFO(this->get_logger(), "Joint names: %s", 
        std::accumulate(opt_joint_names_.begin(), opt_joint_names_.end(), std::string(),
            [](const std::string& a, const std::string& b) { return a + (a.length() > 0 ? ", " : "") + b; }).c_str());
    // TF Hand-to-EE transform
    // read the hand-to-EE transform from the YAML file
    RCLCPP_INFO(this->get_logger(), "Loading hand-to-EE transform from YAML");
    std::string config_path = ament_index_cpp::get_package_share_directory("pose_optimization") + "/config/parameters.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    if (!config["ee2hand"])
    {
        RCLCPP_ERROR(this->get_logger(), "ee2hand configuration not found in %s", config_path.c_str());
        return;
    }
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    if (config["ee2hand"]["translation"])
    {
        translation = Eigen::Vector3d(
            config["ee2hand"]["translation"][0].as<double>(),
            config["ee2hand"]["translation"][1].as<double>(),
            config["ee2hand"]["translation"][2].as<double>()
        );
    }
    Eigen::Vector4d rotation = Eigen::Vector4d::Zero();
    if (config["ee2hand"]["rotation"])
    {
        rotation = Eigen::Vector4d(
            config["ee2hand"]["rotation"][0].as<double>(),
            config["ee2hand"]["rotation"][1].as<double>(),
            config["ee2hand"]["rotation"][2].as<double>(),
            config["ee2hand"]["rotation"][3].as<double>()
        );
    }

    if (config["operational"]) {
        enable_optimization_ = config["operational"]["enable_optimization"].as<bool>(true);
        print_error_before_loop = config["operational"]["print_error_before_loop"].as<bool>(false);
        print_error_after_loop = config["operational"]["print_error_after_loop"].as<bool>(false);
        print_joint_angles = config["operational"]["print_joint_angles"].as<bool>(false);
        print_critical_transforms = config["operational"]["print_critical_transforms"].as<bool>(false);
    }
    RCLCPP_INFO(this->get_logger(), "NLopt optimization enabled: %s", enable_optimization_ ? "true" : "false");

    if (config["sensitivity_analysis"])
    {
        apply_sensitivity = config["sensitivity_analysis"]["apply_sensitivity"].as<bool>(false);
        sigma = config["sensitivity_analysis"]["sigma"].as<double>(0.001);
        RCLCPP_INFO(this->get_logger(), "Sensitivity analysis: apply=%d, sigma=%.4f",
            apply_sensitivity, sigma);
    }
    else
    {
        apply_sensitivity = false;
        sigma = 0.0;
    }

    RCLCPP_INFO(this->get_logger(), "Using NLopt for optimization");
    max_iterations = config["NLopt"]["max_iterations"].as<int>(100);
    algorithm = config["NLopt"]["algorithm"].as<std::string>("LN_COBYLA");
    tolerance = config["NLopt"]["tolerance"].as<double>(1e-4);
    pos_weight = config["NLopt"]["pos_weight"].as<double>(10.0);
    rot_weight = config["NLopt"]["rot_weight"].as<double>(1.0);
    joint_weights = config["NLopt"]["joint_weights"].as<std::vector<double>>();
    w_vel_ = config["NLopt"]["velocity_weight"].as<double>(5.0);
    w_acc_ = config["NLopt"]["acceleration_weight"].as<double>(1.0);
    // normalize joint weights to have a sum of 1
    double sum_weights = std::accumulate(joint_weights.begin(), joint_weights.end(), 0.0);
    if (sum_weights > 0)
    {
        for (auto &weight : joint_weights)
            weight /= sum_weights;
    }
    joint_penalty_weight = config["NLopt"]["joint_penalty_weight"].as<double>(1.0);
    RCLCPP_INFO(this->get_logger(), "NLopt parameters: max_iterations=%d, algorithm=%s, tolerance=%.6f",
        max_iterations, algorithm.c_str(), tolerance);

    opt_ = nlopt_create(nlopt_algorithm_from_string(algorithm.c_str()), kOptDof);
    std::vector<double> lb = { -1.65806279, 0.0, -1.57079633, 0.0, -1.5708, -1.5708, -0.723599 };
    std::vector<double> ub = {  2.26892803, 3.14159265, 2.09439510, 2.53073, 1.48353, 1.5708, 0.549066 };
    nlopt_set_lower_bounds(opt_, lb.data());
    nlopt_set_upper_bounds(opt_, ub.data());
    nlopt_set_min_objective(opt_, PoseOptimizer::costFunction, this);
    nlopt_set_xtol_rel(opt_, tolerance);
    nlopt_set_maxeval(opt_, max_iterations);

    RCLCPP_INFO(this->get_logger(), "NLopt optimization initialized.");

    ee_to_hand_ = Eigen::Isometry3d::Identity();
    Eigen::Quaterniond q (rotation[3], rotation[0], rotation[1], rotation[2]);
    Eigen::Matrix3d R = q.toRotationMatrix();
    ee_to_hand_.linear() = R;
    ee_to_hand_.translation() = translation;
    hand_to_ee_ = ee_to_hand_.inverse();
    RCLCPP_INFO(this->get_logger(), "EE to Hand transform initialized");

    // Initialize the TF broadcaster
    RCLCPP_INFO(this->get_logger(), "Initializing TF broadcaster");
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // optimized joint state publisher
    RCLCPP_INFO(this->get_logger(), "Creating joint state publisher");
    joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "/optimized_arm/joint_states", 10);
    
    // kinematics constants
    hand_idx = model_.getFrameId(hand_frame_name_);
    sh_idx = model_.getFrameId(shoulder_frame_name_);

    // Subscribe to joint states
    RCLCPP_INFO(this->get_logger(), "Subscribing to joint states on /arm/joint_states");
    joint_state_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/arm/joint_states", 10, std::bind(&PoseOptimizer::joint_state_callback, this, std::placeholders::_1));

    // initialize timer
    last_update_time_ = this->get_clock()->now();
    RCLCPP_INFO(this->get_logger(), "PoseOptimizer initialized successfully");

    //intialize a window for velocity smoothing
    int window_size_ = 5;
    dq_window_.resize(window_size_, std::vector<double>(kOptDof, 0.0));

    noise = Eigen::Vector3d::Zero();
    R_noise = Eigen::Matrix3d::Zero();

    std::remove(urdf_path.c_str());

}

Eigen::VectorXd PoseOptimizer::modelConfigurationFromOpt(const double* values) const
{
    Eigen::VectorXd q_model = Eigen::VectorXd::Zero(model_.nq);
    for (std::size_t i = 0; i < kOptDof; ++i) {
        q_model[model_.idx_qs[opt_joint_ids_[i]]] = opt_joint_multipliers_[i] * values[i];
    }
    q_model[model_.idx_qs[mimic_joint_id_]] = mimic_joint_multiplier_ * values[0];
    return q_model;
}

Eigen::VectorXd PoseOptimizer::modelConfigurationFromOpt(const std::vector<double>& values) const
{
    if (values.size() < kOptDof) {
        throw std::runtime_error("Not enough independent joint values for right_arm_osim_shoulder model");
    }
    return modelConfigurationFromOpt(values.data());
}

double PoseOptimizer::costFunction(unsigned n, const double* x, double* grad, void* data) {
    auto* self = reinterpret_cast<PoseOptimizer*>(data);
    (void)n;
    (void)grad;

    Eigen::VectorXd q_model = self->modelConfigurationFromOpt(x);

    // FK
    pinocchio::forwardKinematics(self->model_, self->data_, q_model);
    pinocchio::updateFramePlacements(self->model_, self->data_);


    pinocchio::SE3 T_bh = self->data_.oMf[self->hand_idx];
    if (self->apply_sensitivity)
    {
        T_bh.translation() += self->noise;
        T_bh.rotation() = T_bh.rotation() * self->R_noise;
    }
    pinocchio::SE3 T_model = T_bh;
    pinocchio::SE3 delta = self->T_shoulder_hand_ref.inverse() * T_model;
    pinocchio::Motion error_twist = pinocchio::log6(delta);

    double pose_cost = error_twist.linear().squaredNorm() * self->pos_weight +
                       error_twist.angular().squaredNorm() * self->rot_weight;

    // calculate joint cost to hold independent joints
    double joint_cost = 0.0;
    for (std::size_t i = 0; i < kOptDof; ++i) {
        const int q_idx = self->model_.idx_qs[self->opt_joint_ids_[i]];
        joint_cost += self->joint_weights[i] * std::pow(q_model[q_idx] - self->q_init_[q_idx], 2);

    }

    // soft constraint on smooth movement
    double smooth_cost = 0.0;
    if (self->have_prev_) {
        for (std::size_t i = 0; i < kOptDof; ++i) {
            const int q_idx = self->model_.idx_qs[self->opt_joint_ids_[i]];
            double dv = q_model[q_idx] - self->q_prev_[q_idx];
            smooth_cost += self->w_vel_ * dv * dv;
        }
        if (self->have_prev2_) {
            for (std::size_t i = 0; i < kOptDof; ++i) {
                const int q_idx = self->model_.idx_qs[self->opt_joint_ids_[i]];
                double da = q_model[q_idx] - 2.0*self->q_prev_[q_idx] + self->q_prev2_[q_idx];
                smooth_cost += self->w_acc_ * da * da;
            }
        }
    }


    return pose_cost + self->joint_penalty_weight * joint_cost + smooth_cost;
}

void PoseOptimizer::publishJointState(const Eigen::VectorXd& q_current)
{
    sensor_msgs::msg::JointState optimized_joint_state;
    optimized_joint_state.header.stamp = this->get_clock()->now();
    optimized_joint_state.name = opt_joint_names_;
    optimized_joint_state.position.resize(kOptDof);
    for (std::size_t i = 0; i < kOptDof; ++i)
    {
        const int q_idx = model_.idx_qs[opt_joint_ids_[i]];
        if (!std::isfinite(q_current[q_idx])) {
            RCLCPP_ERROR(this->get_logger(), "Non-finite value in joint %s: %f", opt_joint_names_[i].c_str(), q_current[q_idx]);
            return;
        }
        optimized_joint_state.position[i] = q_current[q_idx];
    }

    optimized_joint_state.velocity.resize(kOptDof, 0.0);
    if (have_prev_) {
        double dt = (this->now() - last_update_time_).seconds();
        for (std::size_t i = 0; i < kOptDof; ++i) {
            const int q_idx = model_.idx_qs[opt_joint_ids_[i]];
            optimized_joint_state.velocity[i] = dt > 0.0 ? (q_current[q_idx] - q_prev_[q_idx]) / dt : 0.0;
        }
    }
    last_update_time_ = this->now();

    joint_state_publisher_->publish(optimized_joint_state);

    if (have_prev_) {
        q_prev2_ = q_prev_;
        have_prev2_ = true;
    }
    q_prev_ = q_current;
    have_prev_ = true;
}


void PoseOptimizer::joint_state_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{   

    // update the robot state based on the received joint states
    // if (msg->data.size() != static_cast<std::size_t>(model_.nq))
    // {
    //     RCLCPP_ERROR(this->get_logger(), "Received joint states size (%zu) does not match model DOF (%d)", msg->data.size(), model_.nq);
    //     return;
    // }

    if (msg->data.size() < kOptDof)
    {
        RCLCPP_ERROR(this->get_logger(), "Received joint states size (%zu) is smaller than optimizer DOF (%zu)", msg->data.size(), kOptDof);
        return;
    }

    // The solver appends points_too_close after the 7 independent OSIM joints.
    if (msg->data.size() > kOptDof && msg->data.back() > 0.5) {
        RCLCPP_WARN(this->get_logger(), "Received joint states with points_too_close flag set, skipping optimization for this frame.");
        return;
    }

    std::vector<double> measured_joints(msg->data.begin(), msg->data.begin() + kOptDof);
    q = modelConfigurationFromOpt(measured_joints);

    q_init_ = q; // Store initial joint angles

    pinocchio::forwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    

    geometry_msgs::msg::TransformStamped tf;
    try
    {
        // tf_shoulder2ee = tf_buffer_.lookupTransform("camera_depth_optical_frame","lbr_link_ee", tf2::TimePointZero);
        tf_shoulder2ee = tf_buffer_.lookupTransform(shoulder_frame_name_, "lbr_link_ee", tf2::TimePointZero);
    }
    catch (const tf2::TransformException &ex)
    {
        RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
        return;
    }

    Eigen::Isometry3d shoulder_to_ee = Eigen::Isometry3d::Identity();
    shoulder_to_ee.translation() = Eigen::Vector3d(tf_shoulder2ee.transform.translation.x,
                                                    tf_shoulder2ee.transform.translation.y,
                                                    tf_shoulder2ee.transform.translation.z);
    Eigen::Quaterniond q_ee(tf_shoulder2ee.transform.rotation.w,
                            tf_shoulder2ee.transform.rotation.x,
                            tf_shoulder2ee.transform.rotation.y,
                            tf_shoulder2ee.transform.rotation.z);
    shoulder_to_ee.linear() = q_ee.toRotationMatrix();

    Eigen::Isometry3d shoulder_to_hand_ref = shoulder_to_ee * ee_to_hand_;
    
    // check if there is nan 
    if (!shoulder_to_hand_ref.linear().allFinite() || 
        !shoulder_to_hand_ref.translation().allFinite()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid shoulder_to_hand_ref transform (NaNs detected)");
        return;
    }

    // turn it into Pinocchio SE3
    T_shoulder_hand_ref.translation() = shoulder_to_hand_ref.translation();
    T_shoulder_hand_ref.rotation() = shoulder_to_hand_ref.linear();
        
    // current estimate of shoulder to hand transform
    T_base_hand = data_.oMf[hand_idx];
    if (print_critical_transforms) {
        RCLCPP_INFO(this->get_logger(), "T_base_hand translation: [%.4f, %.4f, %.4f]", 
            T_base_hand.translation()[0], T_base_hand.translation()[1], T_base_hand.translation()[2]);
    }

    // apply sensitivity analysis if enabled

    T_shoulder_hand = T_base_hand;

    if (apply_sensitivity)
    {
        static std::default_random_engine generator;
        static std::normal_distribution<double> dist_x(0.0, sigma);
        static std::normal_distribution<double> dist_y(0.0, sigma);
        static std::normal_distribution<double> dist_z(0.0, sigma);
        noise = Eigen::Vector3d(dist_x(generator), dist_y(generator), dist_z(generator));
        T_shoulder_hand.translation() += noise;

        // apply roational noise
        static std::normal_distribution<double> dist_rot_x(0.0, sigma/0.6);
        static std::normal_distribution<double> dist_rot_y(0.0, sigma/0.6);
        static std::normal_distribution<double> dist_rot_z(0.0, sigma/0.6);
        Eigen::Vector3d rot_noise(dist_rot_x(generator), dist_rot_y(generator), dist_rot_z(generator));
        Eigen::AngleAxisd aa_x(rot_noise[0], Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd aa_y(rot_noise[1], Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd aa_z(rot_noise[2], Eigen::Vector3d::UnitZ());
        R_noise = (aa_z * aa_y * aa_x).toRotationMatrix();
        T_shoulder_hand.rotation() = T_shoulder_hand.rotation() * R_noise;
    }
    // check if there is nan in the current estimate
    if (print_critical_transforms) {
        RCLCPP_INFO(this->get_logger(), "T_shoulder_hand translation: [%.4f, %.4f, %.4f]", 
            T_shoulder_hand.translation()[0], T_shoulder_hand.translation()[1], T_shoulder_hand.translation()[2]);
    }

    if (!T_shoulder_hand.rotation().allFinite() ||
        !T_shoulder_hand.translation().allFinite()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid T_shoulder_hand transform (NaNs detected)");
        return;
    }
    // RCLCPP_INFO(this->get_logger(), "T_shoulder_hand OK");
    // calculate the error between the current estimate and the reference
    pinocchio::SE3 delta_T = T_shoulder_hand_ref.inverse() * T_shoulder_hand;
    pinocchio::Motion error_twist = pinocchio::log6(delta_T);

    // publish ground truth transform from shoulder to hand
    geometry_msgs::msg::TransformStamped gt_transform;
    gt_transform.header.stamp = tf_shoulder2ee.header.stamp;
    gt_transform.header.frame_id = shoulder_frame_name_;
    gt_transform.child_frame_id = "RightHand (Ground Truth)";
    gt_transform.transform.translation.x = shoulder_to_hand_ref.translation().x();
    gt_transform.transform.translation.y = shoulder_to_hand_ref.translation().y();
    gt_transform.transform.translation.z = shoulder_to_hand_ref.translation().z();
    Eigen::Quaterniond q_gt(shoulder_to_hand_ref.rotation());
    gt_transform.transform.rotation.w = q_gt.w();
    gt_transform.transform.rotation.x = q_gt.x();
    gt_transform.transform.rotation.y = q_gt.y();
    gt_transform.transform.rotation.z = q_gt.z();
    // gt_transform.transform.translation.x = T_shoulder_hand.translation()[0];
    // gt_transform.transform.translation.y = T_shoulder_hand.translation()[1];
    // gt_transform.transform.translation.z = T_shoulder_hand.translation()[2];
    // Eigen::Quaterniond q_gt(T_shoulder_hand.rotation());
    // gt_transform.transform.rotation.w = q_gt.w();
    // gt_transform.transform.rotation.x = q_gt.x();
    // gt_transform.transform.rotation.y = q_gt.y();
    // gt_transform.transform.rotation.z = q_gt.z();
    // RCLCPP_INFO(this->get_logger(), "Publishing ground truth transform from RightShoulder to RightHand");
    tf_broadcaster_->sendTransform(gt_transform);


    if (!enable_optimization_) {
            publishJointState(q);
            return;
        }


    //correct the shoulder to hand transform using the error
    if (print_error_before_loop)
    {
        RCLCPP_INFO(this->get_logger(), "Initial error norm: %.6f", error_twist.toVector().norm());
    }

    double minf;
    std::vector<double> x(kOptDof);
    for (std::size_t i = 0; i < kOptDof; ++i)
    {
        x[i] = q[model_.idx_qs[opt_joint_ids_[i]]]; // initialize with current independent joint angles
    }
    nlopt_result result = nlopt_optimize(opt_, x.data(), &minf);
    if (print_error_after_loop)
    {
        RCLCPP_INFO(this->get_logger(), "NLopt result = %d, final cost = %.6f", result, minf);
        RCLCPP_INFO(this->get_logger(), "Converged after %d iterations", nlopt_get_numevals(opt_));
    }

    if (print_joint_angles)
    {
        RCLCPP_INFO(this->get_logger(), "Optimized joint angles: %f, %f, %f, %f, %f, %f, %f",
            x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
    }
    for (std::size_t i = 0; i < kOptDof; ++i)
    {
        if (!std::isfinite(x[i]))
        {
            RCLCPP_ERROR(this->get_logger(), "Non-finite value in optimized joint %zu: %f", i, x[i]);
            return;
        }
    }
    q = modelConfigurationFromOpt(x);
    publishJointState(q);

}
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseOptimizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
