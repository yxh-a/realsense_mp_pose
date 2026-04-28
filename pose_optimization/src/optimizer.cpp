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
        ament_index_cpp::get_package_share_directory("image_pose_tracking") + "/config/right_arm.urdf.xacro";

    char temp_template[] = "/tmp/right_arm_optimizer_XXXXXX.urdf";
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
        throw std::runtime_error("Failed to render right_arm.urdf.xacro");
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

    joint_names_.reserve(model_.nq);
    for (int i = 1; i < model_.njoints; ++i)  // start from 1 to skip universe joint
        joint_names_.push_back(model_.names[i]);


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
        RCLCPP_ERROR(this->get_logger(), "ee2hand configuration not found in %s", config_path);
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

    if (config["operational"]["method"])
    {
        method_ = config["operational"]["method"].as<std::string>();
        RCLCPP_INFO(this->get_logger(), "Optimization method: %s", method_.c_str());
        print_error_before_loop = config["operational"]["print_error_before_loop"].as<bool>(false);
        print_error_after_loop = config["operational"]["print_error_after_loop"].as<bool>(false);
        print_error_in_loop = config["operational"]["print_error_in_loop"].as<bool>(false);
        print_joint_angles = config["operational"]["print_joint_angles"].as<bool>(false);
        print_critical_transforms = config["operational"]["print_critical_transforms"].as<bool>(false);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "No optimization method specified, using default SVD");
        method_ = "NLopt";  // Default to NLopt if not specified
    }
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

    if (method_ == "NLopt")
    {
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

        opt_ = nlopt_create(nlopt_algorithm_from_string(algorithm.c_str()), 7);
        std::vector<double> lb = { -2.35619, -0.785398, -1.5708, 0, -0.872665, -0.523599, -0.523599 };
        std::vector<double> ub = {  1.5708,   3.14159,   1.5708, 2.53073, 1.0472, 0.349066, 0.349066 };
        nlopt_set_lower_bounds(opt_, lb.data());
        nlopt_set_upper_bounds(opt_, ub.data());
        nlopt_set_min_objective(opt_, PoseOptimizer::costFunction, this);
        nlopt_set_xtol_rel(opt_, tolerance);
        nlopt_set_maxeval(opt_, max_iterations);

        RCLCPP_INFO(this->get_logger(), "NLopt optimization initialized.");

    }
    // getting method from the yaml
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
    dq_window_.resize(window_size_, std::vector<double>(7, 0.0));

    noise = Eigen::Vector3d::Zero();
    R_noise = Eigen::Matrix3d::Zero();

    std::remove(urdf_path.c_str());

}



double PoseOptimizer::costFunction(unsigned n, const double* x, double* grad, void* data) {
    auto* self = reinterpret_cast<PoseOptimizer*>(data);

    // Convert x to Eigen vector
    Eigen::VectorXd q(7);
    for (size_t i = 0; i < 7; ++i)
        q[i] = x[i];

    // FK
    pinocchio::forwardKinematics(self->model_, self->data_, q);
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

    // calculate joint cost to hold first 7 joints
    double joint_cost = 0.0;
    for (size_t i = 0; i < 7; ++i) {
        joint_cost += self->joint_weights[i] * std::pow(q[i] - self->q_init_[i], 2);

    }

    // soft constraint on smooth movement
    double smooth_cost = 0.0;
    if (self->have_prev_) {
        for (int i = 0; i < 7; ++i) {
            double dv = q[i] - self->q_prev_[i];
            smooth_cost += self->w_vel_ * dv * dv;
        }
        if (self->have_prev2_) {
            for (int i = 0; i < 7; ++i) {
                double da = q[i] - 2.0*self->q_prev_[i] + self->q_prev2_[i];
                smooth_cost += self->w_acc_ * da * da;
            }
        }
    }


    return pose_cost + self->joint_penalty_weight * joint_cost + smooth_cost;
}


void PoseOptimizer::joint_state_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{   
    // update the robot state based on the received joint states

    if (msg->data.size() != static_cast<std::size_t>(model_.nq))
    {
        RCLCPP_ERROR(this->get_logger(), "Received joint states size (%zu) does not match model DOF (%d)", msg->data.size(), model_.nq);
        return;
    }
    
    q = Eigen::VectorXd::Zero(model_.nq);
    for (size_t i = 0; i < joint_names_.size(); ++i)
    {
        q[i] = msg->data[i];
    }

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


    //correct the shoulder to hand transform using the error
    if (print_error_before_loop)
    {
        RCLCPP_INFO(this->get_logger(), "Initial error norm: %.6f", error_twist.toVector().norm());
    }

    if (method_ == "NLopt")
    {
        double minf;
        std::vector<double> x(7);
        for (int i = 0; i < 7; ++i)
        {
            x[i] = q[i]; // initialize with current joint angles
        }
        nlopt_result result = nlopt_optimize(opt_, x.data(), &minf);
        if (print_error_after_loop)
        {
            RCLCPP_INFO(this->get_logger(), "NLopt result = %d, final cost = %.6f", result, minf);
            RCLCPP_INFO(this->get_logger(), "Converged after %d iterations", nlopt_get_numevals(opt_));
        }

        if (print_joint_angles)
        {
            RCLCPP_INFO(this->get_logger(), "Initial joint angles: %f, %f, %f, %f, %f, %f, %f",
                q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
        }
        for (int i = 0; i < 7; ++i)
        {
            if (!std::isfinite(x[i]))
            {
                RCLCPP_ERROR(this->get_logger(), "Non-finite value in optimized joint %d: %f", i, x[i]);
                return;
            }
            q[i] = x[i];
        }
    }
    // publish the optimized joint states
    sensor_msgs::msg::JointState optimized_joint_state;
    optimized_joint_state.header.stamp = this->get_clock()->now();
    optimized_joint_state.name = opt_joint_names_;
    optimized_joint_state.position.resize(model_.nq);
    for (size_t i = 0; i < model_.nq; ++i)
    {   
        if (!std::isfinite(q[i])) {
            RCLCPP_ERROR(this->get_logger(), "Non-finite value in joint %s: %f", opt_joint_names_[i].c_str(), q[i]);
            return;
        }
        optimized_joint_state.position[i] = q[i];
    }

    optimized_joint_state.velocity.resize(model_.nq, 0.0); // Set velocities to zero
    // calculate the velocity of each joint bbased on q and q_prev_
    if (have_prev_) {
        double dt = (this->now() - last_update_time_).seconds();
        for (size_t i = 0; i < model_.nq; ++i) {
            // if (dt > 0){
            //     double raw_dq = (q[i] - q_prev_[i]) / dt;
            //     // add the new velocity to the window
            //     dq_window_.erase(dq_window_.begin());
            //     dq_window_.push_back(std::vector<double>(7, 0.0));
            //     dq_window_.back()[i] = raw_dq;
            //     // calculate the smoothed velocity
            //     double smoothed_dq = 0.0;
            //     for (const auto& vel_vec : dq_window_) {
            //         smoothed_dq += vel_vec[i];
            //     }
            //     smoothed_dq /= dq_window_.size();
            //     optimized_joint_state.velocity[i] = smoothed_dq;
            // }
            // else {
            //     optimized_joint_state.velocity[i] = 0.0;
            // }
            optimized_joint_state.velocity[i] = (q[i] - q_prev_[i]) / dt;
        }
    }
    last_update_time_ = this->now(); // Update the last update time

    joint_state_publisher_->publish(optimized_joint_state);

    // update previous joint states
    if (have_prev_) {
        q_prev2_ = q_prev_;
        have_prev2_ = true;
    }
    q_prev_ = q;
    have_prev_ = true;

}
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PoseOptimizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
