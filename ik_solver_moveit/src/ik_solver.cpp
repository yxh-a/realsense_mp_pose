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

using namespace std;

std::string loadFile(const std::string& path)
{
    std::ifstream t(path);
    if (!t)
    throw std::runtime_error("Could not open file: " + path);
    return std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("ik_solver_node");

    // === Load URDF and SRDF from file ===
    std::string urdf_path = ament_index_cpp::get_package_share_directory("image_pose_tracking") + "/config/right_arm.urdf";
    std::string srdf_path = ament_index_cpp::get_package_share_directory("arm_moveit_config") + "/config/upper_arm.srdf";
    std::string urdf_string = loadFile(urdf_path);
    std::string srdf_string = loadFile(srdf_path);

    // === Load the robot model ===
    // auto urdf_model = urdf::parseURDF(urdf_string);
    // auto srdf_model = std::make_shared<srdf::Model>();
    // srdf_model->initString(*urdf_model, srdf_string);

    // auto robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
    node->declare_parameter("robot_description", urdf_string);
    node->declare_parameter("robot_description_semantic", srdf_string);
    
    // load solver
    // Parse manually
    std::string kin_path = ament_index_cpp::get_package_share_directory("arm_moveit_config") + "/config/kinematics.yaml";
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

    //   === set up static transforms from yaml ===
    // TF Hand-to-EE transform
    // read the hand-to-EE transform from the YAML file
    RCLCPP_INFO(node->get_logger(), "Loading hand-to-EE transform from YAML");
    std::string config_path = ament_index_cpp::get_package_share_directory("pose_optimization") + "/config/parameters.yaml";
    YAML::Node config = YAML::LoadFile(config_path);
    if (!config["ee2hand"])
    {
        RCLCPP_ERROR(node->get_logger(), "ee2hand configuration not found in %s", config_path);
        return 0;
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
    Eigen::Vector3d rotation = Eigen::Vector3d::Zero();
    if (config["ee2hand"]["rotation"])
    {
        rotation = Eigen::Vector3d(
            config["ee2hand"]["rotation"][0].as<double>(),
            config["ee2hand"]["rotation"][1].as<double>(),
            config["ee2hand"]["rotation"][2].as<double>()
        );
    }

    Eigen::Isometry3d T_ee_hand = Eigen::Isometry3d::Identity();
    T_ee_hand.translation() = translation;
    Eigen::Matrix3d R = Eigen::AngleAxisd(rotation[0], Eigen::Vector3d::UnitX()).toRotationMatrix()
                            * Eigen::AngleAxisd(rotation[1], Eigen::Vector3d::UnitY()).toRotationMatrix()
                            * Eigen::AngleAxisd(rotation[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_ee_hand.linear() = R;
    Eigen::Isometry3d T_hand_ee = T_ee_hand.inverse();
    RCLCPP_INFO(node->get_logger(), "Hand to EE transform loaded successfully");

    //   === Initialize TF listener ===
    tf2_ros::Buffer tf_buffer(node->get_clock());
    tf2_ros::TransformListener tf_listener(tf_buffer);

    // Initialize the TF broadcaster
    RCLCPP_INFO(node->get_logger(), "Initializing TF broadcaster");
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);

    int baseline_hz = 20; // Default value
    if (config["operational"]["baseline_hz"])
    {
        baseline_hz = config["operational"]["baseline_hz"].as<int>(20);
        RCLCPP_INFO(node->get_logger(), "Baseline frequency for optimization: %d Hz", baseline_hz);
    }
    
    auto joint_state_pub = node->create_publisher<sensor_msgs::msg::JointState>("/based/joint_states", 10);


    while (rclcpp::ok())
    {
        // read end-effector pose from tf2 ros
        geometry_msgs::msg::TransformStamped transform_stamped;
        try
        {
            transform_stamped = tf_buffer.lookupTransform("RightShoulder", "lbr_link_ee", tf2::TimePointZero);
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

        // publish the hand pose as a transform to tf
        geometry_msgs::msg::TransformStamped hand_transform_stamped;
        hand_transform_stamped.header.stamp = node->now();
        hand_transform_stamped.header.frame_id = "RightShoulder";
        hand_transform_stamped.child_frame_id = "hand_frame";
        hand_transform_stamped.transform.translation.x = hand_pose.translation().x();
        hand_transform_stamped.transform.translation.y = hand_pose.translation().y();
        hand_transform_stamped.transform.translation.z = hand_pose.translation().z();
        Eigen::Quaterniond hand_quat(hand_pose.linear());
        hand_transform_stamped.transform.rotation.x = hand_quat.x();
        hand_transform_stamped.transform.rotation.y = hand_quat.y();
        hand_transform_stamped.transform.rotation.z = hand_quat.z();
        hand_transform_stamped.transform.rotation.w = hand_quat.w();   
        tf_broadcaster->sendTransform(hand_transform_stamped);

        RCLCPP_INFO(node->get_logger(), "hand_pose: translation: [%f, %f, %f], rotation: [%f, %f, %f, %f]",
                    hand_pose.translation().x(), hand_pose.translation().y(), hand_pose.translation().z(),
                    q.w(), q.x(), q.y(), q.z());
        // solve inverse kinematics based on the hand pose
        std::vector<double> joint_values(joint_model_group->getVariableCount(), 0.0);
        
        bool ik_solved = robot_state.setFromIK(joint_model_group, hand_pose, "RightHand", 0.1);
        if (!ik_solved)
        {
            RCLCPP_ERROR(node->get_logger(), "Inverse kinematics could not be solved for the hand pose.");
        }
        else
        {
            RCLCPP_INFO(node->get_logger(), "Inverse kinematics solved successfully.");
            // Update the robot state with the new joint values
            robot_state.setVariablePositions(joint_values);
        }

        // Publish the joint states
        sensor_msgs::msg::JointState joint_state_msg;
        joint_state_msg.header.stamp = node->now();
        joint_state_msg.name = joint_model_group->getVariableNames();
        joint_state_msg.position = joint_values;

        joint_state_pub->publish(joint_state_msg);


        // rest based on baseline frequency
        rclcpp::Rate rate(baseline_hz);
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
