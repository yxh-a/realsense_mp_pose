#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/log.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

using namespace std;

class PoseOptimizer : public rclcpp::Node
{
public:
    PoseOptimizer()
        : Node("pose_optimizer"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {   
        // Load the URDF model into Pinocchio
        RCLCPP_INFO(this->get_logger(), "Loading robot model...");
        std::string urdf_path = ament_index_cpp::get_package_share_directory("image_pose_tracking") + "/config/right_arm.urdf";

        pinocchio::urdf::buildModel(urdf_path, model_);
        data_ = pinocchio::Data(model_);

        joint_names_.reserve(model_.nq);
        for (int i = 1; i < model_.njoints; ++i)  // start from 1 to skip universe joint
            joint_names_.push_back(model_.names[i]);


        RCLCPP_INFO(this->get_logger(), "Robot model loaded with %d DOF", model_.nq);
        RCLCPP_INFO(this->get_logger(), "Joint names: %s", 
            std::accumulate(joint_names_.begin(), joint_names_.end(), std::string(),
                [](const std::string& a, const std::string& b) { return a + (a.length() > 0 ? ", " : "") + b; }).c_str());

        // TF Hand-to-EE transform
        // read the hand-to-EE transform from the YAML file
        RCLCPP_INFO(this->get_logger(), "Loading hand-to-EE transform from YAML");
        std::string config_path = ament_index_cpp::get_package_share_directory("pose_optimization") + "/config/ee2hand.yaml";
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
        Eigen::Vector3d rotation = Eigen::Vector3d::Zero();
        if (config["ee2hand"]["rotation"])
        {
            rotation = Eigen::Vector3d(
                config["ee2hand"]["rotation"][0].as<double>(),
                config["ee2hand"]["rotation"][1].as<double>(),
                config["ee2hand"]["rotation"][2].as<double>()
            );
        }

        if (config["operational"]["method"])
        {
            method_ = config["operational"]["method"].as<std::string>();
            RCLCPP_INFO(this->get_logger(), "Optimization method: %s", method_.c_str());
            print_error_before_loop = config["operational"]["print_error_before_loop"].as<bool>(false);
            print_error_after_loop = config["operational"]["print_error_after_loop"].as<bool>(false);
            print_error_in_loop = config["operational"]["print_error_in_loop"].as<bool>(false);
            print_jacobian = config["operational"]["print_jacobian"].as<bool>(false);
            convergence_threshold = config["operational"]["convergence_threshold"].as<float>(1e-2);
            max_iterations = config["operational"]["max_iterations"].as<int>(1000);
            damping_factor = config["operational"]["damping_factor"].as<float>(1e-2);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No optimization method specified, using default SVD");
            method_ = "SVD";
        }


        // getting method from the yaml
        ee_to_hand_ = Eigen::Isometry3d::Identity();
        Eigen::Matrix3d R = Eigen::AngleAxisd(rotation[0], Eigen::Vector3d::UnitX()).toRotationMatrix()
                            * Eigen::AngleAxisd(rotation[1], Eigen::Vector3d::UnitY()).toRotationMatrix()
                            * Eigen::AngleAxisd(rotation[2], Eigen::Vector3d::UnitZ()).toRotationMatrix();
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
        hand_idx = model_.getFrameId("RightHand");
        sh_idx = model_.getFrameId("RightShoulder");

        // Subscribe to joint states
        RCLCPP_INFO(this->get_logger(), "Subscribing to joint states on /arm/joint_states");
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/arm/joint_states", 10, std::bind(&PoseOptimizer::joint_state_callback, this, std::placeholders::_1));

    }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    std::vector<std::string> joint_names_;
    Eigen::Isometry3d hand_to_ee_, ee_to_hand_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    geometry_msgs::msg::TransformStamped tf_shoulder2ee;
    pinocchio::SE3 T_shoulder_hand, T_shoulder_hand_ref;
    pinocchio::FrameIndex hand_idx, sh_idx;

    // operational variables
    std::string method_;
    bool print_error_before_loop, print_error_after_loop, print_error_in_loop, print_jacobian;
    float convergence_threshold, damping_factor;
    int max_iterations;

    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {   
        // update the robot state based on the received joint states

        if (msg->name.size() != model_.nq)
        {
            RCLCPP_ERROR(this->get_logger(), "Received joint states size (%zu) does not match model DOF (%d)", msg->name.size(), model_.nq);
            return;
        }
        
        Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
        for (size_t i = 0; i < joint_names_.size(); ++i)
        {
            auto it = std::find(msg->name.begin(), msg->name.end(), joint_names_[i]);
            if (it != msg->name.end())
                q[i] = msg->position[it - msg->name.begin()];
        }


        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);


        geometry_msgs::msg::TransformStamped tf;
        try
        {
            tf_shoulder2ee = tf_buffer_.lookupTransform("RightShoulder","lbr_link_ee", tf2::TimePointZero);
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
        T_shoulder_hand = data_.oMf[hand_idx];
        // check if there is nan in the current estimate
        if (!T_shoulder_hand.rotation().allFinite() ||
            !T_shoulder_hand.translation().allFinite()) {
            RCLCPP_ERROR(this->get_logger(), "Invalid T_shoulder_hand transform (NaNs detected)");
            return;
        }

        // calculate the error between the current estimate and the reference
        pinocchio::SE3 delta_T = T_shoulder_hand_ref.inverse() * T_shoulder_hand;
        pinocchio::Motion error_twist = pinocchio::log6(delta_T);


        // publish ground truth transform from shoulder to hand
        geometry_msgs::msg::TransformStamped gt_transform;
        gt_transform.header.stamp = this->get_clock()->now();
        gt_transform.header.frame_id = "RightShoulder";
        gt_transform.child_frame_id = "RightHand (Ground Truth)";
        gt_transform.transform.translation.x = shoulder_to_hand_ref.translation().x();
        gt_transform.transform.translation.y = shoulder_to_hand_ref.translation().y();
        gt_transform.transform.translation.z = shoulder_to_hand_ref.translation().z();
        Eigen::Quaterniond q_gt(shoulder_to_hand_ref.rotation());
        gt_transform.transform.rotation.w = q_gt.w();
        gt_transform.transform.rotation.x = q_gt.x();
        gt_transform.transform.rotation.y = q_gt.y();
        gt_transform.transform.rotation.z = q_gt.z();
        // RCLCPP_INFO(this->get_logger(), "Publishing ground truth transform from RightShoulder to RightHand");
        tf_broadcaster_->sendTransform(gt_transform);

        //correct the shoulder to hand transform using the error
        if (print_error_before_loop)
        {
            RCLCPP_INFO(this->get_logger(), "Initial error norm: %.6f", error_twist.toVector().norm());
        }

        if (method_ == "SVD")
        {
            // Apply SVD optimization as 100 iterations
            for (int i = 0; i < max_iterations; ++i)
            {
                pinocchio::forwardKinematics(model_, data_, q);
                pinocchio::updateFramePlacements(model_, data_);
                // calculate the Jacobian for the hand frame
                pinocchio::computeJointJacobians(model_, data_, q);
                pinocchio::framesForwardKinematics(model_, data_, q);

                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, model_.nq);

                pinocchio::getFrameJacobian(model_, data_, hand_idx, pinocchio::LOCAL_WORLD_ALIGNED, J);


                // current shoulder to hand transform
                T_shoulder_hand = data_.oMf[hand_idx];
                // check if there is nan in the current estimate
                if (!T_shoulder_hand.rotation().allFinite() ||
                    !T_shoulder_hand.translation().allFinite()) {
                    RCLCPP_ERROR(this->get_logger(), "Invalid T_shoulder_hand transform (NaNs detected)");
                    return;
                }

                // turn reference shoulder to hand transform into Pinocchio SE3
                pinocchio::SE3 delta_T = T_shoulder_hand_ref.inverse() * T_shoulder_hand;
                pinocchio::Motion error_twist = pinocchio::log6(delta_T);

                Eigen::VectorXd error_vector(6);
                error_vector = error_twist.toVector();
                if (error_vector.norm() < convergence_threshold) // convergence criterion
                {
                    RCLCPP_INFO(this->get_logger(), "Converged after %d iterations", i);
                    break;
                }

                if (print_error_in_loop)
                {
                    RCLCPP_INFO(this->get_logger(), "Iteration %d: Error norm = %.6f", i, error_vector.norm());
                }

    
                const auto &hand_frame = model_.frames[hand_idx];
                // log frame type
                if (print_jacobian)
                {
                    RCLCPP_INFO(this->get_logger(), "Calculating Jacobian for frame: %s", hand_frame.name.c_str());
                    RCLCPP_INFO(this->get_logger(), "Frame type: %d", static_cast<int>(hand_frame.type));
                    RCLCPP_INFO(this->get_logger(), "Jacobian size: %d x %d", J.rows(), J.cols());
                    // Print the Jacobian if required
                    for (int j = 0; j < model_.nq; ++j) {
                        double col_norm = J.col(j).norm();
                        RCLCPP_INFO(this->get_logger(), "Joint %s: Jacobian column norm = %.6f", joint_names_[j].c_str(), col_norm);
                    }
                }

                // damped least squares inverse
                Eigen::MatrixXd JJt = J* J.transpose();
                float damping_factor = error_vector.norm() * damping_factor; // damping factor based on error norm
                Eigen::MatrixXd damping_matrix = damping_factor * Eigen::MatrixXd::Identity(6,6);
                Eigen::MatrixXd J_pinv = J.transpose() * (JJt + damping_matrix).inverse();

                // joint update
                Eigen::VectorXd dq = J_pinv * error_vector;
                q = pinocchio::integrate(model_, q, dq);
            }

            if (print_error_after_loop)
            {
                pinocchio::forwardKinematics(model_, data_, q);
                pinocchio::updateFramePlacements(model_, data_);
                T_shoulder_hand = data_.oMf[hand_idx];
                pinocchio::SE3 delta_T = T_shoulder_hand_ref.inverse() * T_shoulder_hand;
                pinocchio::Motion final_error_twist = pinocchio::log6(delta_T);
                RCLCPP_INFO(this->get_logger(), "Final error norm: %.6f", final_error_twist.toVector().norm());
            }

            // RCLCPP_INFO(this->get_logger(), "SVD optimization completed");
            // publish the optimized joint states
            sensor_msgs::msg::JointState optimized_joint_state;
            optimized_joint_state.header.stamp = this->get_clock()->now();
            optimized_joint_state.name = joint_names_;
            optimized_joint_state.position.resize(model_.nq);
            for (size_t i = 0; i < model_.nq; ++i)
            {
                optimized_joint_state.position[i] = q[i];
            }
            joint_state_publisher_->publish(optimized_joint_state);

        }
        
    }


};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseOptimizer>());
    rclcpp::shutdown();
    return 0;
}
