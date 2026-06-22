#pragma once

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <Eigen/Dense>

#include <array>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

class ArmPosePfNode : public rclcpp::Node
{
public:
    ArmPosePfNode();

private:
    static constexpr int kDof = 7;
    static constexpr int kStateSize = 14;

    struct Pf
    {
        struct Particle
        {
            Eigen::Matrix<double, kStateSize, 1> x = Eigen::Matrix<double, kStateSize, 1>::Zero();
            double log_weight = 0.0;
        };

        std::vector<Particle> particles;
        Eigen::Matrix<double, kStateSize, 1> x = Eigen::Matrix<double, kStateSize, 1>::Zero();
        Eigen::Matrix<double, kStateSize, kStateSize> P = Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
        std::mt19937 rng;
        double resample_effective_ratio = 0.5;
        double roughening_stddev = 1.0e-4;

        void init(const Eigen::Matrix<double, kDof, 1>& q0,
                  const Eigen::Matrix<double, kDof, 1>& dq0,
                  const Eigen::Matrix<double, kDof, 1>& q_stddev,
                  double dq_variance,
                  int particle_count,
                  unsigned int seed,
                  double resample_ratio,
                  double roughening_stddev);
        void predict(double dt, double acceleration_variance, double velocity_decay);
        void addLogLikelihood(std::size_t particle_index, double log_likelihood);
        bool normalizeWeights();
        double effectiveSampleSize() const;
        bool resampleIfNeeded();
        void refreshEstimateFromParticles();
        Eigen::Matrix<double, kDof, 1> q() const;
        Eigen::Matrix<double, kDof, 1> dq() const;
    };

    // Data acquisition
    void robotJointCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void forwardSolverJointCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    // PF pipeline — initialization
    bool initializeParticlesFromForwardSolver(const std_msgs::msg::Float32MultiArray& msg);

    // PF pipeline — predict
    void predictTo(const rclcpp::Time& stamp);

    // PF pipeline — measurement correction
    void correctHandPoseAndTwist(const sensor_msgs::msg::JointState& robot_msg);
    // Thread-safe per-particle hand pose/twist likelihood. Runs FK/Jacobian on the
    // supplied (thread-local) Data so it can be called concurrently across particles.
    double handMeasurementLikelihood(
        const Eigen::Matrix<double, kStateSize, 1>& state,
        pinocchio::Data& data,
        const pinocchio::SE3& T_shoulder_hand_measured,
        const Eigen::Matrix<double, 6, 1>& hand_twist_shoulder,
        const Eigen::MatrixXd& covariance,
        int measurement_size,
        bool need_twist) const;
    // Optional robust vision likelihood on the forward-solver q (Student-t + gate).
    void correctVisionQ();
    void updateWeightsFromLikelihoods(const std::vector<double>& likelihoods, const char* source);
    void updateWeightsFromLogLikelihoods(const std::vector<double>& log_likelihoods,
                                         const char* source);

    // PF pipeline — resampling and constraints
    void injectForwardSolverProposalParticles();
    void loadStateConstraintsFromArmModel();
    Eigen::Matrix<double, kDof, 1> jointLimitQStddev(
        double fraction, const Eigen::Matrix<double, kDof, 1>& fallback) const;
    void applyStateConstraints(const char* source);
    void applyStateConstraintsToParticle(
        Eigen::Matrix<double, kStateSize, 1>& state, bool warn, const char* source);

    // Kinematics
    Eigen::VectorXd modelConfigurationFromOpt(const double* values) const;
    Eigen::VectorXd modelConfigurationFromOpt(const Eigen::Matrix<double, kDof, 1>& values) const;
    Eigen::Matrix<double, 6, Eigen::Dynamic> independentJacobian(
        const Eigen::Matrix<double, 6, Eigen::Dynamic>& full_jacobian) const;
    void updateArmKinematics(const Eigen::Matrix<double, kDof, 1>& q,
                              bool compute_jacobians = true);
    pinocchio::SE3 robotEndEffectorPose(const sensor_msgs::msg::JointState& msg,
                                        Eigen::Matrix<double, 6, 1>& ee_twist_world,
                                        bool compute_twist);

    // TF lookups
    pinocchio::SE3 lookupShoulderToEndEffector(rclcpp::Time& tf_stamp) const;
    pinocchio::SE3 lookupWorldToShoulder(const rclcpp::Time& stamp) const;

    // Publishing
    void publishState(const rclcpp::Time& stamp);
    void publishGroundTruthHandTransform(const pinocchio::SE3& T_shoulder_hand,
                                         const rclcpp::Time& stamp) const;

    // Math utilities
    static double gaussianLikelihood(const Eigen::VectorXd& residual,
                                     const Eigen::MatrixXd& covariance);

    std::string robot_prefix_;
    std::string robot_color_;
    std::string world_frame_;
    std::string shoulder_frame_name_;
    std::string hand_frame_name_;
    std::string ee_frame_name_;

    pinocchio::Model robot_model_;
    pinocchio::Data robot_data_;
    pinocchio::FrameIndex robot_ee_frame_id_;

    pinocchio::Model arm_model_;
    pinocchio::Data arm_data_;
    // One Data per OpenMP thread so per-particle FK can run concurrently. arm_model_
    // is read-only after construction and safe to share; each Data is written by a
    // single thread only.
    std::vector<pinocchio::Data> arm_data_pool_;
    std::array<pinocchio::JointIndex, kDof> arm_joint_ids_;
    std::array<double, kDof> arm_joint_multipliers_;
    Eigen::Matrix<double, kDof, 1> q_lower_limits_ =
        Eigen::Matrix<double, kDof, 1>::Constant(-std::numeric_limits<double>::infinity());
    Eigen::Matrix<double, kDof, 1> q_upper_limits_ =
        Eigen::Matrix<double, kDof, 1>::Constant(std::numeric_limits<double>::infinity());
    Eigen::Matrix<double, kDof, 1> dq_velocity_limits_ =
        Eigen::Matrix<double, kDof, 1>::Constant(std::numeric_limits<double>::infinity());
    pinocchio::JointIndex mimic_joint_id_;
    double mimic_joint_multiplier_ = -1.0;
    std::string ground_truth_hand_frame_;
    pinocchio::FrameIndex hand_frame_id_;
    std::vector<std::string> joint_names_;

    pinocchio::SE3 T_ee_hand_ = pinocchio::SE3::Identity();

    Pf pf_;
    bool initialized_ = false;
    rclcpp::Time last_predict_stamp_;
    rclcpp::Time last_processed_robot_stamp_;

    double initial_q_variance_ = 1.0;
    double initial_dq_variance_ = 1.0;
    double joint_limit_initial_q_stddev_fraction_ = 0.2;
    double joint_limit_proposal_q_stddev_fraction_ = 0.2;
    double forward_solver_initial_q_variance_ = 0.05;
    double forward_solver_initial_dq_variance_ = 0.01;
    double forward_solver_proposal_resampling_fraction_ = 0.1;
    Eigen::Matrix<double, kDof, 1> forward_solver_proposal_q_stddev_ =
        Eigen::Matrix<double, kDof, 1>::Constant(0.1);
    Eigen::Matrix<double, kDof, 1> forward_solver_proposal_dq_stddev_ =
        Eigen::Matrix<double, kDof, 1>::Constant(0.3);
    Eigen::Matrix<double, kDof, 1> latest_forward_solver_q_ =
        Eigen::Matrix<double, kDof, 1>::Zero();
    rclcpp::Time latest_forward_solver_q_stamp_;
    int particle_count_ = 500;
    unsigned int particle_seed_ = 1;
    double particle_resample_effective_ratio_ = 0.5;
    double particle_roughening_stddev_ = 1.0e-4;
    double process_acceleration_variance_ = 1.0e-3;
    double velocity_decay_ = 1.0;
    double hand_position_sigma_ = 0.01;
    double hand_rotation_sigma_ = 0.05;
    double hand_linear_velocity_sigma_ = 0.03;
    double hand_angular_velocity_sigma_ = 0.1;
    bool use_hand_pose_ = true;
    bool use_hand_pose_position_ = true;
    bool use_hand_pose_rotation_ = false;
    bool use_hand_twist_ = true;
    bool use_linearized_fk_ = false;
    // Vision (forward-solver q) likelihood
    bool use_vision_measurement_ = false;
    Eigen::Matrix<double, kDof, 1> vision_q_sigma_ =
        Eigen::Matrix<double, kDof, 1>::Constant(0.25);
    double vision_student_t_dof_ = 3.0;
    double vision_max_age_sec_ = 0.3;
    bool use_vision_gating_ = true;
    double vision_nis_threshold_ = 18.48;
    bool initialize_from_forward_solver_ = true;
    bool particles_initialized_from_forward_solver_ = false;
    bool forward_solver_proposal_enabled_ = true;
    bool has_latest_forward_solver_q_ = false;
    bool enforce_joint_limits_ = true;
    bool zero_velocity_at_limits_ = true;
    bool use_latest_shoulder_tf_ = true;
    bool fallback_to_latest_tf_ = true;
    double tf_lookup_timeout_sec_ = 0.05;
    double max_predict_dt_sec_ = 1.0;
    double fallback_predict_dt_sec_ = 1.0 / 250.0;
    double max_update_rate_hz_ = 100.0;
    int robot_joint_queue_depth_ = 1;
    int forward_solver_queue_depth_ = 1;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr robot_joint_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr forward_solver_joint_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
