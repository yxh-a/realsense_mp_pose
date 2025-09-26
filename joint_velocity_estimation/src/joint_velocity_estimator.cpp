#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "std_msgs/msg/float32_multi_array.hpp"


#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <chrono>

using namespace std::chrono_literals;


class EndEffectorVelocityNode : public rclcpp::Node
{
public:
  EndEffectorVelocityNode() : Node("ee_velocity_node")
  {
    // --------------- Robot Model Initialization ---------------
    RCLCPP_INFO(this->get_logger(), "Initializing robot model...");
    // Load URDF and build the model
    std::string urdf_path = ament_index_cpp::get_package_share_directory("lbr_description") + "/urdf/iiwa7/iiwa7.urdf";
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);

    q_ = Eigen::VectorXd::Zero(model_.nq);
    dq_ = Eigen::VectorXd::Zero(model_.nv);
    pinocchio::framesForwardKinematics(model_, data_, q_);

    RCLCPP_INFO(this->get_logger(), "Model loaded: %s", model_.name.c_str());

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/lbr/joint_states", 10,
      std::bind(&EndEffectorVelocityNode::jointCallback, this, std::placeholders::_1));

    twist_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/lbr/ee_velocity", 10);

    // set the frame name of end effector
    ee_frame_name_ = "lbr_link_ee";  // change to your robot's actual end effector frame
    ee_frame_id_ = model_.getFrameId(ee_frame_name_);

    // ---------------- Human Arm Model Initialization ----------------
    RCLCPP_INFO(this->get_logger(), "Initializing human arm model...");
    std::string human_arm_urdf_path = ament_index_cpp::get_package_share_directory("image_pose_tracking") + "/config/right_arm_opt.urdf";
    pinocchio::urdf::buildModel(human_arm_urdf_path, arm_model_);
    arm_data_ = pinocchio::Data(arm_model_);

    RCLCPP_INFO(this->get_logger(), "Human arm model initialized.");
    hand_frame_name_ = "opt_RightHandCOM";  // change to your robot's actual hand frame
    hand_frame_id_ = arm_model_.getFrameId(hand_frame_name_);
    RCLCPP_INFO(this->get_logger(), "Human arm model loaded: %s", arm_model_.name.c_str());
    
    // subscribe to the human arm joint states
    q_arm_ = Eigen::VectorXd::Zero(arm_model_.nq);
    dq_arm_ = Eigen::VectorXd::Zero(arm_model_.nv);
    dq_vis_ = Eigen::VectorXd::Zero(arm_model_.nv);
    RCLCPP_INFO(this->get_logger(), "nq and nv: %d, %d", arm_model_.nq, arm_model_.nv);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_arm_);

    arm_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/optimized_arm/joint_states", 10,
      std::bind(&EndEffectorVelocityNode::jointCallback_arm, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to human arm joint states.");
    updated_arm_joint_states_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/updated_arm/joint_states", 10);

    bad_arm_joint_states_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
      "/bad_arm/joint_states", 10);
  

    // Initialize static transformations
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
    T_eehand_ = pinocchio::SE3::Identity();
    Eigen::Quaterniond q (rotation[3], rotation[0], rotation[1], rotation[2]);
    Eigen::Matrix3d R = q.toRotationMatrix();
    T_eehand_.rotation() = R;
    T_eehand_.translation() = translation;

    T_handee_ = T_eehand_.inverse();
    
    // initialize Kalman filter
    RCLCPP_INFO(this->get_logger(), "Initializing Kalman filter...");
    std::string kf_config_path = ament_index_cpp::get_package_share_directory("joint_velocity_estimation") + "/config/joint_velocity_config.yaml";
    YAML::Node kf_config = YAML::LoadFile(kf_config_path);
    if (!kf_config["parameters"])
    {
      RCLCPP_ERROR(this->get_logger(), "parameters configuration not found in %s", kf_config_path);
      return;
    }
    if (kf_config["parameters"]["kf"])
    {
      q_a_ = kf_config["parameters"]["kf"]["q_a"].as<double>(1.0e-1);
      rate_hz_ = kf_config["parameters"]["rates"]["control_rate_hz"].as<double>(200.0);
      sigma_task_ = kf_config["parameters"]["kf"]["sigma_task"].as<double>(0.5);
      sigma_null_ = kf_config["parameters"]["kf"]["sigma_null"].as<double>(0.1);
      r_sigma_lin_ = kf_config["parameters"]["kf"]["r_sigma_lin"].as<double>(0.01);
      r_sigma_rot_ = kf_config["parameters"]["kf"]["r_sigma_rot"].as<double>(0.01);
      sigma_q_ = kf_config["parameters"]["kf"]["sigma_q"].as<double>(0.1);
      sigma_dq_ = kf_config["parameters"]["kf"]["sigma_dq"].as<double>(0.1);
      acc_cap_ = kf_config["parameters"]["kf"]["acc_cap"].as<double>(10.0);
    }
    
    if (kf_config["parameters"]["output"])
    {
      print_velocity_ = kf_config["parameters"]["output"]["print_velocity"].as<bool>(true);
      print_sigmas_ = kf_config["parameters"]["output"]["print_sigmas"].as<bool>(true);
      RCLCPP_INFO(this->get_logger(), "Output configuration: print_velocity = %s", print_velocity_ ? "true" : "false");
      RCLCPP_INFO(this->get_logger(), "Output configuration: print_sigmas = %s", print_sigmas_ ? "true" : "false");
    }
    
    // TF Robot to Camera transform
    T_worldcamera_ = pinocchio::SE3::Identity();
    if (kf_config["parameters"]["visualizer"]["static_transformation"]) {
      const auto& trans = kf_config["parameters"]["visualizer"]["static_transformation"];
      if (trans.size() == 7) {
        T_worldcamera_.translation() <<
            trans[0].as<double>(), trans[1].as<double>(), trans[2].as<double>();
        Eigen::Quaterniond q(
            trans[6].as<double>(), trans[3].as<double>(),
            trans[4].as<double>(), trans[5].as<double>()
        );
        q.normalize();
        T_worldcamera_.rotation() = q.toRotationMatrix();
      }
    }

    RCLCPP_INFO(this->get_logger(), "Robot to Camera transform initialized");

    // TF Camera to Shoulder transform can be read from arm_model_
    T_camerashoulder_ = arm_data_.oMf[arm_model_.getFrameId("RightShoulder")];
    if (!T_camerashoulder_.rotation().allFinite() || !T_camerashoulder_.translation().allFinite())
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid Camera to Shoulder transform (NaNs detected)");
      return;
    }
    T_worldshoulder_ = T_worldcamera_ * T_camerashoulder_;

    // set initial state of Kalman filter
    kf_.init(model_.nv, 1.0/200.0, 1e-3, 1e-2);

  }

private:
  void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->position.size() != model_.nq || msg->velocity.size() != model_.nv)
    {
      RCLCPP_WARN(this->get_logger(), "Joint state size mismatch with model.");
      return;
    }

    Eigen::VectorXd q = Eigen::VectorXd::Map(msg->position.data(), model_.nq);
    Eigen::VectorXd dq = Eigen::VectorXd::Map(msg->velocity.data(), model_.nv);

    // Compute Jacobian
    pinocchio::computeJointJacobians(model_, data_, q);
    pinocchio::framesForwardKinematics(model_, data_, q);
    pinocchio::updateFramePlacements(model_, data_);

    T_worldee_ = data_.oMf[ee_frame_id_];
    if (!isValidTransform(T_worldee_))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid end effector transform (NaNs detected)");
      return;
    }

    Eigen::Matrix<double, 6, Eigen::Dynamic> J;
    J.setZero(6, model_.nv);
    J = pinocchio::getFrameJacobian(model_, data_, ee_frame_id_, pinocchio::WORLD);

    v_ee_ = J * dq;

    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header.stamp = msg->header.stamp;
    twist_msg.header.frame_id = "lbr_link_ee";  // change to your robot's actual end effector frame

    twist_msg.twist.linear.x = v_ee_[0];
    twist_msg.twist.linear.y = v_ee_[1];
    twist_msg.twist.linear.z = v_ee_[2];
    twist_msg.twist.angular.x = v_ee_[3];
    twist_msg.twist.angular.y = v_ee_[4];
    twist_msg.twist.angular.z = v_ee_[5];

    twist_pub_->publish(twist_msg);

    // transform the twist of EE in world frame to twist of ee in camera frame
    Ad_T = T_worldcamera_.inverse().toActionMatrix();
    if (!isValidMatrix(Ad_T))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid action matrix (NaNs detected)");
      return;
    }
    Eigen::VectorXd v_ee_in_camera_frame_;
    v_ee_in_camera_frame_ = Ad_T * v_ee_;

    // transform the twist of ee in camera frame to twist of hand in camera frame
    Ad_T = T_eehand_.toActionMatrix();
    if (!isValidMatrix(Ad_T))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid action matrix (NaNs detected)");
      return;
    }
    // v_hand_ = Ad_T * v_ee_in_camera_frame_;
    v_hand_ = v_ee_in_camera_frame_;
    // RCLCPP_INFO(this->get_logger(), "Hand linear velocity (camera frame): %.3f, %.3f, %.3f",
    //             v_hand_[0], v_hand_[1], v_hand_[2]);

    // get arm jacboian (7DoF)
    pinocchio::computeJointJacobians(arm_model_, arm_data_, q_arm_);
    pinocchio::framesForwardKinematics(arm_model_, arm_data_, q_arm_);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J_arm;
    J_arm.setZero(6, arm_model_.nv);
    J_arm = pinocchio::getFrameJacobian(arm_model_, arm_data_,hand_frame_id_, pinocchio::WORLD);
    if (!isValidMatrix(J_arm))
    {
      RCLCPP_ERROR(this->get_logger(), "Invalid arm Jacobian (NaNs detected)");
      return;
    }

    kf_.predict(q_a_);
    now_ = this->now().seconds();
    double latency = now_ - t_vis_;
    double scale = std::max(1.0, latency / 0.05); // scale the measurement noise based on latency (assuming 20Hz camera)

    kf_.correct(J_arm, v_hand_, q_arm_, dq_vis_, r_sigma_lin_, r_sigma_rot_, sigma_q_*scale, sigma_dq_*scale);

    dq_arm_ = kf_.dq();

    // // cap the acceleration
    // if (!first_dq_) {
    //   dq_arm_prev_ = dq_arm_;
    //   first_dq_ = true;
    // }
    // if (first_dq_) {
    //   Eigen::VectorXd ddq = (dq_arm_ - dq_arm_prev_)/kf_.dtc;
    //   for (int i=0; i<ddq.size(); i++) {
    //     if (ddq[i] > acc_cap_) {
    //       dq_arm_[i] = dq_arm_prev_[i] + acc_cap_ * kf_.dtc;
    //     }
    //     else if (ddq[i] < -acc_cap_) {
    //       dq_arm_[i] = dq_arm_prev_[i] - acc_cap_ * kf_.dtc;
    //     }
    //   }
    //   dq_arm_prev_ = dq_arm_;
    // }


    if (print_velocity_)
    {
      RCLCPP_INFO(this->get_logger(), "Hand velocity: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
                  v_hand_[0], v_hand_[1], v_hand_[2], v_hand_[3], v_hand_[4], v_hand_[5]);
      RCLCPP_INFO(this->get_logger(), "Estimated arm joint velocities (deg): %f, %f, %f, %f, %f, %f, %f",
                dq_arm_[0]*180.0/M_PI, dq_arm_[1]*180.0/M_PI, dq_arm_[2]*180.0/M_PI,
                dq_arm_[3]*180.0/M_PI, dq_arm_[4]*180.0/M_PI, dq_arm_[5]*180.0/M_PI, dq_arm_[6]*180.0/M_PI);
    }
  
    // RCLCPP_INFO(this->get_logger(), "Estimated arm joint velocities (deg): %f, %f, %f, %f, %f, %f, %f",
    //           dq_arm_[0]*180.0/M_PI, dq_arm_[1]*180.0/M_PI, dq_arm_[2]*180.0/M_PI,
    //           dq_arm_[3]*180.0/M_PI, dq_arm_[4]*180.0/M_PI, dq_arm_[5]*180.0/M_PI, dq_arm_[6]*180.0/M_PI);

    sensor_msgs::msg::JointState updated_arm_joint_state;
    updated_arm_joint_state.header.stamp = this->now();
    updated_arm_joint_state.name = joint_names_;
    Eigen::VectorXd q_arm_updated = kf_.q();
    updated_arm_joint_state.position = std::vector<double>(q_arm_updated.data(), q_arm_updated.data() + q_arm_updated.size());
    updated_arm_joint_state.velocity = std::vector<double>(dq_arm_.data(), dq_arm_.data() + dq_arm_.size());
    
    updated_arm_joint_states_pub_->publish(updated_arm_joint_state);

    Eigen::MatrixXd J_arm_pinv = J_arm.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::VectorXd dq_bad = J_arm_pinv * v_hand_;
    sensor_msgs::msg::JointState bad_arm_joint_state;
    bad_arm_joint_state.header.stamp = this->now();
    bad_arm_joint_state.name = arm_model_.names;
    bad_arm_joint_state.position = std::vector<double>(q_arm_.data(), q_arm_.data() + q_arm_.size());
    bad_arm_joint_state.velocity = std::vector<double>(dq_bad.data(), dq_bad.data() + dq_bad.size());
    bad_arm_joint_states_pub_->publish(bad_arm_joint_state);
    

  }

  void jointCallback_arm(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->name.size() != arm_model_.nq || msg->position.size() != arm_model_.nq)
    {
      RCLCPP_WARN(this->get_logger(), "Arm joint state size mismatch with model.");
      return;
    }

    q_arm_ = Eigen::VectorXd::Map(msg->position.data(), arm_model_.nq);
    dq_vis_ = Eigen::VectorXd::Map(msg->velocity.data(), arm_model_.nv);
    t_vis_ = this->now().seconds();
    // RCLCPP_INFO(this->get_logger(), "nq and nv: %d, %d", arm_model_.nq, arm_model_.nv);
    // kf_.set_state(q_arm_, dq_vis_);
    pinocchio::forwardKinematics(arm_model_, arm_data_, q_arm_);
    pinocchio::updateFramePlacements(arm_model_, arm_data_);
    if (!first_vis_) {
      first_vis_ = true;
      kf_.set_state(q_arm_, dq_vis_);
      return;
    }
  }

  bool isValidTransform(const pinocchio::SE3 &T)
  {
    return T.rotation().allFinite() && T.translation().allFinite();
  }

  bool isValidMatrix(const Eigen::MatrixXd &M)
  {
    return M.allFinite();
  }


  pinocchio::Model model_, arm_model_;
  pinocchio::Data data_, arm_data_;
  std::string ee_frame_name_, hand_frame_name_;
  pinocchio::FrameIndex ee_frame_id_, hand_frame_id_;
  pinocchio::SE3 T_eehand_, T_handee_, T_worldcamera_,T_camerashoulder_,T_worldee_,T_worldhand_,T_shoulderhand_,T_worldshoulder_;
  Eigen::VectorXd q_arm_, dq_arm_, q_, dq_, dq_arm_prev_;
  Eigen::VectorXd v_hand_, v_ee_;
  Eigen::Matrix<double, 6, 6> Ad_T;

  // Kalman filter parameters
  double q_q_, q_dq_, q_a_; // process noise
  double rate_hz_ = 200.0; // control rate in Hz
  double sigma_task_ = 0.5; // task measurement noise
  double sigma_null_ = 2.0; // null-space process noise
  double r_sigma_lin_ = 0.02; // measurement noise
  double r_sigma_rot_ = 0.05; // measurement noise
  double sigma_q_ = 0.01; // vision joint position measurement noise
  double sigma_dq_ = 0.1; // vision joint velocity measurement noise
  double acc_cap_ = 10.0; // cap on max acceleration (rad/s^2)

  bool print_velocity_ = true; // whether to print joint velocities to console
  bool print_sigmas_ = true; // whether to print measurement and task sigmas to console

  bool first_dq_ = false;

  std::vector<std::string> joint_names_ = {
      "upt_jRightShoulder_rotx",
      "upt_jRightShoulder_rotz",
      "upt_jRightShoulder_roty",
      "upt_jRightElbow_rotz",
      "upt_jRightElbow_roty",
      "upt_jRightWrist_rotx",
      "upt_jRightWrist_rotz"
  };

  double t_vis_,now_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_; 
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr arm_joint_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr updated_arm_joint_states_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr bad_arm_joint_states_pub_;

  struct DQKalman{
    int n{0}; 
    double dtc{0.0};
    // State: x = [q; dq], size 2*n
    Eigen::VectorXd x;
    Eigen::MatrixXd P, I;
    bool initialized{false};
    // (2n x 2n)
    void init(int n_, double dtc_, double Pq0 = 1e-2, double Pdq0 = 1e-1) {
      n   = n_;
      dtc = dtc_;
      x   = Eigen::VectorXd::Zero(2*n);
      P   = Eigen::MatrixXd::Zero(2*n, 2*n);
      I   = Eigen::MatrixXd::Identity(2*n, 2*n);
      P.block(0,   0,   n, n).setIdentity();  P.block(0,   0,   n, n) *= Pq0;
      P.block(n,   n,   n, n).setIdentity();  P.block(n,   n,   n, n) *= Pdq0;
      initialized = true;
    }

    void set_state(const Eigen::VectorXd& q0, const Eigen::VectorXd& dq0) {
      x.head(n) = q0;
      x.tail(n) = dq0;
    }

    void predict(double qa) {
      if (!initialized) return;

      // Constant-velocity model
      Eigen::MatrixXd F = Eigen::MatrixXd::Identity(2*n, 2*n);
      F.block(0, n, n, n) = dtc * Eigen::MatrixXd::Identity(n, n);

      const double dt   = dtc;
      const double dt2  = dt * dt;
      const double dt3  = dt2 * dt;


      // Process noise
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
    void correct(const Eigen::Matrix<double,6,Eigen::Dynamic>& J,
                 const Eigen::Matrix<double,6,1>& v_meas, const Eigen::VectorXd& q_meas, const Eigen::VectorXd& dq_meas,
                 double r_sigma_lin, double r_sigma_rot, double sigma_q, double sigma_dq)
    {
      if (!initialized) return;

      // correct based on end-effector twist measurement
      Eigen::MatrixXd H_v = Eigen::MatrixXd::Zero(6, 2*n);
      H_v.block(0, n, 6, n) = J;                 // H = [0  J]
      Eigen::VectorXd z_pred = J * x.tail(n);  // damped pseudo-inverse
      Eigen::VectorXd r_v      = v_meas - z_pred;
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

      // --- 4) Stack ALL rows vertically
      const int rows = 6 + n + n;
      Eigen::MatrixXd H(rows, 2*n);
      Eigen::VectorXd r(rows);
      Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, rows);

      int o = 0;
      H.block(o, 0, 6, 2*n) = H_v;               r.segment(o, 6) = r_v;   R.block(o, o, 6, 6) = R_v;  o += 6;
      H.block(o, 0, n, 2*n) = H_q;               r.segment(o, n) = r_q;   R.block(o, o, n, n) = R_q;  o += n;
      H.block(o, 0, n, 2*n) = H_dq;              r.segment(o, n) = r_dq;  R.block(o, o, n, n) = R_dq; o += n;

      // --- 5) Kalman update (this updates BOTH q and dq)
      Eigen::MatrixXd S = H * P * H.transpose() + R;
      Eigen::MatrixXd K = P * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(rows, rows));

      Eigen::VectorXd dx = K * r;   // dx has size 2n → first n affect q, last n affect dq
      x += dx;
      P  = (I - K * H) * P;
    }
    // Accessors
    Eigen::VectorXd q()  const { return x.head(n); }
    Eigen::VectorXd dq() const { return x.tail(n); }
  } kf_;

  Eigen::VectorXd dq_vis_;
  rclcpp::Time last_vis_stamp_;
  bool first_vis_ = false;

};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<EndEffectorVelocityNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

