#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <urdf_parser/urdf_parser.h>
#include <fstream>
#include <streambuf>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>

using namespace std;

// Per-subject arm geometry, keyed by subject_id in subject_arm_lengths.yaml. This
// mirrors the loader in arm_pose_ekf so the IK solver renders the same human-arm
// model the estimators do.
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

std::string loadFile(const std::string& path)
{
    std::ifstream t(path);
    if (!t)
    throw std::runtime_error("Could not open file: " + path);
    return std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
}

// Config files live in image_pose_tracking/share/config; an absolute path is
// passed through unchanged.
std::string resolveConfigPath(const std::string& file)
{
    if (!file.empty() && file.front() == '/')
    {
        return file;
    }
    return ament_index_cpp::get_package_share_directory("image_pose_tracking") +
        "/config/" + file;
}

ArmLengths loadSubjectArmLengths(
    const std::string& arm_lengths_file,
    int subject_id,
    const ArmLengths& fallback)
{
    const std::string yaml_path = resolveConfigPath(arm_lengths_file);
    const YAML::Node root = YAML::LoadFile(yaml_path);
    const std::string subject_key = "subject_" + std::to_string(subject_id);
    const YAML::Node subject = root["subjects"][subject_key];
    if (!subject)
    {
        throw std::runtime_error(
            "Subject arm lengths not found for " + subject_key + " in " + yaml_path);
    }

    ArmLengths lengths = fallback;
    if (subject["upper_arm_length"])
    {
        lengths.upper_arm_length = subject["upper_arm_length"].as<double>();
    }
    if (subject["forearm_length"])
    {
        lengths.forearm_length = subject["forearm_length"].as<double>();
    }
    if (subject["hand_length"])
    {
        lengths.hand_length = subject["hand_length"].as<double>();
    }
    if (subject["grip_offset"])
    {
        const auto offset = subject["grip_offset"].as<std::vector<double>>();
        if (offset.size() != 3)
        {
            throw std::runtime_error(
                "grip_offset must have 3 entries [x, y, z] for " + subject_key + " in " + yaml_path);
        }
        constexpr double kDegToRad = M_PI / 180.0;
        lengths.grip_offset_rad =
            Eigen::Vector3d(offset[0], offset[1], offset[2]) * kDegToRad;
    }
    if (!(lengths.upper_arm_length > 0.0) || !(lengths.forearm_length > 0.0) ||
        !(lengths.hand_length > 0.0))
    {
        throw std::runtime_error(
            "Subject arm lengths must be positive for " + subject_key + " in " + yaml_path);
    }
    return lengths;
}

std::string shellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (const char c : value)
    {
        if (c == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::string runCommand(const std::string& command)
{
    std::array<char, 4096> buffer{};
    std::string output;

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe)
    {
        throw std::runtime_error("Failed to run command: " + command);
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
    {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        throw std::runtime_error("Command failed: " + command);
    }

    return output;
}

std::string renderArmXacro(
    const std::string& robot_prefix,
    const std::string& robot_color,
    const ArmLengths& lengths,
    const std::string& arm_xacro_file)
{
    const std::string xacro_path = resolveConfigPath(arm_xacro_file);

    const std::string command =
        "xacro " + shellQuote(xacro_path) +
        " robot_prefix:=" + shellQuote(robot_prefix) +
        " robot_color:=" + shellQuote(robot_color) +
        " upper_arm_length:=" + std::to_string(lengths.upper_arm_length) +
        " forearm_length:=" + std::to_string(lengths.forearm_length) +
        " hand_length:=" + std::to_string(lengths.hand_length);

    return runCommand(command);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("ik_solver_node");

    node->declare_parameter<std::string>("robot_prefix", "");
    node->declare_parameter<std::string>("robot_color", "blue");
    node->declare_parameter<std::string>("arm_xacro_file", "right_arm_osim_shoulder_mesh.urdf.xacro");
    node->declare_parameter<std::string>("subject_arm_lengths_file", "subject_arm_lengths.yaml");
    node->declare_parameter<int>("subject_id", 1);
    // Fallback lengths, used only if the subject is missing from the YAML or omits a field.
    node->declare_parameter<double>("upper_arm_length", 0.299);
    node->declare_parameter<double>("forearm_length", 0.248);
    node->declare_parameter<double>("hand_length", 0.0918);

    const auto robot_prefix = node->get_parameter("robot_prefix").as_string();
    const auto robot_color = node->get_parameter("robot_color").as_string();
    const auto arm_xacro_file = node->get_parameter("arm_xacro_file").as_string();
    const auto subject_arm_lengths_file = node->get_parameter("subject_arm_lengths_file").as_string();
    const int subject_id = std::max<int64_t>(1, node->get_parameter("subject_id").as_int());

    ArmLengths fallback_lengths;
    fallback_lengths.upper_arm_length = node->get_parameter("upper_arm_length").as_double();
    fallback_lengths.forearm_length = node->get_parameter("forearm_length").as_double();
    fallback_lengths.hand_length = node->get_parameter("hand_length").as_double();
    const ArmLengths lengths =
        loadSubjectArmLengths(subject_arm_lengths_file, subject_id, fallback_lengths);

    // === Load URDF and SRDF ===
    std::string srdf_path = ament_index_cpp::get_package_share_directory("arm_osim_moveit_config") +
        "/config/upper_arm_osim_shoulder_mesh.srdf";
    std::string urdf_string = renderArmXacro(robot_prefix, robot_color, lengths, arm_xacro_file);
    std::string srdf_string = loadFile(srdf_path);

    RCLCPP_INFO(
        node->get_logger(),
        "Rendered %s for subject_%d with prefix='%s', color='%s', "
        "upper_arm_length=%.3f, forearm_length=%.3f, hand_length=%.3f",
        arm_xacro_file.c_str(), subject_id, robot_prefix.c_str(), robot_color.c_str(),
        lengths.upper_arm_length, lengths.forearm_length, lengths.hand_length);

    // === Load the robot model ===
    // auto urdf_model = urdf::parseURDF(urdf_string);
    // auto srdf_model = std::make_shared<srdf::Model>();
    // srdf_model->initString(*urdf_model, srdf_string);

    // auto robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
    urdf_string = node->declare_parameter<std::string>("robot_description", urdf_string);
    srdf_string = node->declare_parameter<std::string>("robot_description_semantic", srdf_string);
    
    // load solver
    // Parse manually
    std::string kin_path = ament_index_cpp::get_package_share_directory("arm_osim_moveit_config") + "/config/kinematics.yaml";
    YAML::Node config_kin = YAML::LoadFile(kin_path);

    // Set robot_description_kinematics.arm.* as individual parameters
    if (config_kin["arm"]) {
        auto arm_config = config_kin["arm"];
        node->declare_parameter("robot_description_kinematics.arm.kinematics_solver", arm_config["kinematics_solver"].as<std::string>());
        node->declare_parameter("robot_description_kinematics.arm.kinematics_solver_timeout", arm_config["kinematics_solver_timeout"].as<double>());
        node->declare_parameter("robot_description_kinematics.arm.kinematics_solver_search_resolution", arm_config["kinematics_solver_search_resolution"].as<double>());
    }

    robot_model_loader::RobotModelLoader robot_model_loader(node, "robot_description");
    auto robot_model = robot_model_loader.getModel();

    // === Create a robot state and set to default ===
    moveit::core::RobotState robot_state(robot_model);
    robot_state.setToDefaultValues();

    // === Get joint group ===
    const std::string group_name = "arm";
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup(group_name);

    if (!joint_model_group)
    {
    RCLCPP_ERROR(node->get_logger(), "Joint group '%s' not found in SRDF.", group_name.c_str());
    return 1;
    }

    //   === set up the ee->hand transform from parameters ===
    // Hand-eye (ee_to_hand) transform, matching the convention used by
    // arm_pose_ekf / arm_pose_pf: translation [x, y, z], rotation quaternion
    // [x, y, z, w].
    node->declare_parameter<std::vector<double>>(
        "ee_to_hand.translation", std::vector<double>{0.0, -0.03, 0.1});
    node->declare_parameter<std::vector<double>>(
        "ee_to_hand.rotation",
        std::vector<double>{0.2988362387, 0.6408563821, -0.2988362387, 0.6408563821});

    const auto translation_param = node->get_parameter("ee_to_hand.translation").as_double_array();
    const auto rotation_param = node->get_parameter("ee_to_hand.rotation").as_double_array();
    if (translation_param.size() != 3 || rotation_param.size() != 4)
    {
        RCLCPP_ERROR(node->get_logger(),
                     "ee_to_hand.translation must have 3 entries and ee_to_hand.rotation 4 (x,y,z,w).");
        return 1;
    }

    Eigen::Isometry3d T_ee_hand = Eigen::Isometry3d::Identity();
    T_ee_hand.translation() =
        Eigen::Vector3d(translation_param[0], translation_param[1], translation_param[2]);
    Eigen::Quaterniond q_ee_hand(
        rotation_param[3], rotation_param[0], rotation_param[1], rotation_param[2]);
    T_ee_hand.linear() = q_ee_hand.normalized().toRotationMatrix();

    // Apply the per-subject grip-compliance offset to the nominal ee->hand
    // rotation. The offset is intrinsic X->Y->Z Euler angles about the HAND-frame
    // axes, so it right-multiplies the nominal rotation:
    // R_ee_hand <- R_ee_hand * Rx * Ry * Rz. Translation is unchanged; a
    // [0, 0, 0] offset is a no-op. (Mirrors arm_pose_ekf.)
    const Eigen::Vector3d& grip = lengths.grip_offset_rad;
    const Eigen::Matrix3d R_grip_offset =
        (Eigen::AngleAxisd(grip.x(), Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(grip.y(), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(grip.z(), Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    T_ee_hand.linear() = T_ee_hand.linear() * R_grip_offset;
    RCLCPP_INFO(node->get_logger(),
                "ee_to_hand transform loaded from parameters (grip_offset deg [%.3f, %.3f, %.3f])",
                grip.x() * 180.0 / M_PI, grip.y() * 180.0 / M_PI, grip.z() * 180.0 / M_PI);

    // The osim arm group includes a mimic joint (shoulder1_r2 mimics
    // shoulder_rot): the group has 8 variables but only 7 are independent.
    // Work with the 7 active DOF (matching arm_pose_ekf / arm_pose_pf output);
    // MoveIt resolves the mimic internally during IK.
    std::vector<std::string> joint_names;
    for (const moveit::core::JointModel* jm : joint_model_group->getActiveJointModels())
    {
        joint_names.push_back(jm->getName());
    }
    const std::size_t active_count = joint_names.size();
    RCLCPP_INFO(node->get_logger(), "Active joints in group '%s' (%zu):",
                group_name.c_str(), active_count);
    for (const auto& name : joint_names)
    {
        RCLCPP_INFO(node->get_logger(), "  %s", name.c_str());
    }

    // Initial IK seed for the active DOF.
    node->declare_parameter<std::vector<double>>(
        "initial_joint_states", std::vector<double>{0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0});
    const std::vector<double> initial_joint_states =
        node->get_parameter("initial_joint_states").as_double_array();
    if (initial_joint_states.size() != active_count)
    {
        RCLCPP_ERROR(node->get_logger(),
                     "initial_joint_states has %zu entries but group '%s' has %zu active joints.",
                     initial_joint_states.size(), group_name.c_str(), active_count);
        return 1;
    }
    robot_state.setJointGroupActivePositions(joint_model_group, initial_joint_states);
    robot_state.update();
    std::vector<double> joint_values = initial_joint_states;

    //   === Initialize TF listener ===
    tf2_ros::Buffer tf_buffer(node->get_clock());
    tf2_ros::TransformListener tf_listener(tf_buffer);

    // Initialize the TF broadcaster
    RCLCPP_INFO(node->get_logger(), "Initializing TF broadcaster");
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);

    const int baseline_hz = node->declare_parameter<int>("baseline_hz", 20);
    RCLCPP_INFO(node->get_logger(), "Baseline frequency: %d Hz", baseline_hz);

    // Frames for the shoulder->ee lookup; default to the unprefixed osim model.
    const auto shoulder_frame = node->declare_parameter<std::string>("shoulder_frame", "RightShoulder");
    const auto robot_ee_frame = node->declare_parameter<std::string>("robot_ee_frame", "lbr_link_ee");

    auto joint_state_pub = node->create_publisher<sensor_msgs::msg::JointState>("/based/joint_states", 10);


    while (rclcpp::ok())
    {
        // read end-effector pose from tf2 ros
        geometry_msgs::msg::TransformStamped transform_stamped;
        try
        {
            transform_stamped = tf_buffer.lookupTransform(shoulder_frame, robot_ee_frame, tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(node->get_logger(), "Could not get transform: %s", ex.what());
            rclcpp::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        // Convert the transform to hand pose
        Eigen::Isometry3d ee_pose = Eigen::Isometry3d::Identity();
        ee_pose.translation() = Eigen::Vector3d(
            transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z
        );
        Eigen::Quaterniond q(transform_stamped.transform.rotation.w,
                             transform_stamped.transform.rotation.x,
                             transform_stamped.transform.rotation.y,
                             transform_stamped.transform.rotation.z);
        ee_pose.linear() = q.toRotationMatrix(); 
        // Apply the hand-to-EE transform
        Eigen::Isometry3d hand_pose = ee_pose * T_ee_hand;

        RCLCPP_INFO(node->get_logger(), "hand_pose: translation: [%f, %f, %f], rotation: [%f, %f, %f, %f]",
                    hand_pose.translation().x(), hand_pose.translation().y(), hand_pose.translation().z(),
                    q.w(), q.x(), q.y(), q.z());
        // solve inverse kinematics based on the hand pose
        bool ik_solved = robot_state.setFromIK(joint_model_group, hand_pose, 0.1);
        if (!ik_solved)
        {
            RCLCPP_ERROR(node->get_logger(), "Inverse kinematics could not be solved for the hand pose.");
        }
        else
        {
            RCLCPP_INFO(node->get_logger(), "Inverse kinematics solved successfully.");
            for (std::size_t i = 0; i < joint_names.size(); ++i)
            {
                joint_values[i] = robot_state.getVariablePosition(joint_names[i]);
            }
        }

        // Publish the joint states
        sensor_msgs::msg::JointState joint_state_msg;
        joint_state_msg.header.stamp = transform_stamped.header.stamp;
        joint_state_msg.name = joint_names;
        joint_state_msg.position = joint_values;

        joint_state_pub->publish(joint_state_msg);


        // rest based on baseline frequency
        rclcpp::Rate rate(baseline_hz);
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
