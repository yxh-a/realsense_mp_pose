import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
import os
from sensor_msgs.msg import JointState
from tf2_geometry_msgs import TransformStamped
# from urdfpy import URDF


# this node offers simple analytical solution of a human right arm given the 3d postion of shoulder, elbow and wrist in cartesian space
def normalize(v):
    return v / np.linalg.norm(v)

def rotation_from_vector_to_vector(a, b):
    """Return rotation matrix that rotates vector a to vector b."""
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.allclose(v, 0):  # vectors are parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        else:
            # 180-degree rotation around axis orthogonal to a
            axis = normalize(np.cross(a, np.array([1, 0, 0]) if abs(a[0]) < 0.99 else np.array([0, 1, 0])))
            return R.from_rotvec(np.pi * axis).as_matrix()
    s = np.linalg.norm(v)
    kmat = np.array([[  0, -v[2],  v[1]],
                     [ v[2],   0, -v[0]],
                     [-v[1], v[0],   0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


class Arm_Solver_Node(Node):
    def __init__(self):
        super().__init__('forward_solver_node')
        self.pose_subscriber = self.create_subscription(
            PoseArray,
            '/joint_positions',
            self.pose_callback,
            10
        )
    
        self.upper_arm_length = 0.32
        self.lower_arm_length = 0.25

        self.wrist_translation = np.array([0.0, 0.0, 0.0])
        self.elbow_translation = np.array([0.0, 0.0, 0.0])
        self.shoulder_translation = np.array([0.0, 0.0, 0.0])

        self.R_cam2shoulder = np.array([
        [ 0,  0, -1],  # x_shoulder = -z_camera
        [ 0, -1,  0],  # y_shoulder = -y_camera
        [-1,  0,  0]   # z_shoulder = -x_camera
        ])

        self.joint_state_publisher = self.create_publisher(JointState, 'arm/joint_states', 10)

        # implement median filter for joint angles 4 x window size
        self.joint_angles_buffer = {
            'theta_shoulder_x': [],
            'theta_shoulder_y': [],
            'theta_shoulder_z': [],
            'theta_elbow': [],
            'theta_wrist_x': [],
            'theta_wrist_y': [],
            'theta_wrist_z': []
        }
        self.joint_angles_buffer_size = 5  # size of the buffer for median filter

        self.old_4dof_angles = np.zeros(4)  # to store the previous 4 DoF angles

        # setup phyiscal constraints
        self.physical_constraints = {
            # Shoulder (3 DoF)
            'theta_shoulder_x': (-2.35619,  1.5708),   # (-135°,   90°)  jRightShoulder_rotx
            'theta_shoulder_z': (-0.785398, 3.14159),  # ( -45°,  180°)  jRightShoulder_rotz
            'theta_shoulder_y': (-1.5708,   1.5708),   # ( -90°,   90°)  jRightShoulder_roty

            # Elbow (2 DoF)
            'theta_elbow_z':    (0.0,       2.53073),  # (   0°,  145°)  jRightElbow_rotz
            'theta_elbow_y':    (-1.5708,   1.48353),  # ( -90°,   85°)  jRightElbow_roty

            # Wrist (2 DoF)
            'theta_wrist_x':    (-0.872665, 1.0472),   # ( -50°,   60°)  jRightWrist_rotx
            'theta_wrist_z':    (-0.523599, 0.349066), # ( -30°,   20°)  jRightWrist_rotz

        }
        self.joint_velocity_limits = {
            # Shoulder: keep around ≈ 170–200°/s to match fast ADLs / isokinetic tests, well below sport extremes
            'theta_shoulder_x': 3.0,   # ≈ 172°/s
            'theta_shoulder_z': 3.5,   # ≈ 201°/s (IR/ER tends to be quickest) 
            'theta_shoulder_y': 3.0,   # ≈ 172°/s

            # Elbow: flex/extend can move quickly in tasks; set a bit higher
            'theta_elbow_z':    4.0,   # ≈ 229°/s
            'theta_elbow_y':    3.0,   # ≈ 172°/s

            # Wrist: often tested at 150–210°/s in isokinetic protocols — keep similar/under
            'theta_wrist_x':    3.5,   # ≈ 201°/s (flex/extend)
            'theta_wrist_z':    2.0,   # ≈ 115°/s (deviation/axial at wrist—more conservative)
        }
        
        # setup timer
        self.time = self.get_clock().now()
        self.dt = 0.0

        

        self.get_logger().info("URDF file loaded successfully.")
        self.get_logger().info("IK Solver Node Initialized")
    
    def transform_to_shoulder_frame(self,p_camera):
        return self.R_cam2shoulder @ p_camera
    
    def publish_joint_states(self, theta_shoulder_x, theta_shoulder_y, theta_shoulder_z, theta_elbow, theta_wrist_x, theta_wrist_y, theta_wrist_z):
        # Apply median filter to joint angles
        self.joint_angles_buffer['theta_shoulder_x'].append(theta_shoulder_x)
        self.joint_angles_buffer['theta_shoulder_y'].append(theta_shoulder_y)
        self.joint_angles_buffer['theta_shoulder_z'].append(theta_shoulder_z)
        self.joint_angles_buffer['theta_elbow'].append(theta_elbow)
        self.joint_angles_buffer['theta_wrist_x'].append(theta_wrist_x)
        self.joint_angles_buffer['theta_wrist_y'].append(theta_wrist_y)
        self.joint_angles_buffer['theta_wrist_z'].append(theta_wrist_z)

        if len(self.joint_angles_buffer['theta_shoulder_x']) > self.joint_angles_buffer_size:
            self.joint_angles_buffer['theta_shoulder_x'].pop(0)
            self.joint_angles_buffer['theta_shoulder_y'].pop(0)
            self.joint_angles_buffer['theta_shoulder_z'].pop(0)
            self.joint_angles_buffer['theta_elbow'].pop(0)
            self.joint_angles_buffer['theta_wrist_x'].pop(0)
            self.joint_angles_buffer['theta_wrist_y'].pop(0)
            self.joint_angles_buffer['theta_wrist_z'].pop(0)

        theta_shoulder_x = np.mean(self.joint_angles_buffer['theta_shoulder_x'])
        theta_shoulder_y = np.mean(self.joint_angles_buffer['theta_shoulder_y'])
        theta_shoulder_z = np.mean(self.joint_angles_buffer['theta_shoulder_z'])
        theta_elbow = np.mean(self.joint_angles_buffer['theta_elbow'])
        theta_wrist_x = np.mean(self.joint_angles_buffer['theta_wrist_x'])
        theta_wrist_y = np.mean(self.joint_angles_buffer['theta_wrist_y'])
        theta_wrist_z = np.mean(self.joint_angles_buffer['theta_wrist_z'])
        
    
        # pbulish joint states
        joint_states = JointState()
        joint_states.header.stamp = self.get_clock().now().to_msg()
        joint_states.name = [
            'jRightShoulder_rotx',
            'jRightShoulder_rotz',
            'jRightShoulder_roty',
            'jRightElbow_rotz',
            'jRightElbow_roty',
            'jRightWrist_rotx',
            'jRightWrist_rotz'
        ]
        joint_states.position = [
            np.radians(theta_shoulder_x),  # Shoulder x rotation
            np.radians(theta_shoulder_z),  # Shoulder z rotation
            np.radians(theta_shoulder_y),  # Shoulder y rotation
            np.radians(theta_elbow),        # Elbow rotation
            np.radians(theta_wrist_y),  # Wrist y rotation
            np.radians(theta_wrist_x),  # Wrist x rotation
            np.radians(theta_wrist_z)   # Wrist z rotation
        ]
      
        # self.get_logger().info(f"Publishing joint states: {joint_states.position}")
        self.joint_state_publisher.publish(joint_states)

    def get_4DOF_joint_angles(self, u, v):

        # check if u and v are valid vectors
        if u is None or v is None or np.linalg.norm(u) < 1e-6 or np.linalg.norm(v) < 1e-6:
            self.get_logger().error("Invalid vectors for joint angle calculation.")
            return 0, 0, 0, 0
        # get elbow angle
        theta_elbow = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        theta_elbow_deg = np.degrees(theta_elbow)
        # self.get_logger().info(f"Elbow angle: {theta_elbow_deg:.2f} degrees, upper arm length: {upper_arm_length:.2f}, lower arm length: {lower_arm_length:.2f}")

        #  get 3dof shoulder angle
        u_normalized = normalize(u)
        u0 = np.array([0,-1,0])  # shoulder frame y-axis

        Rxz_shoulder = rotation_from_vector_to_vector(u0, u_normalized)
        
        # check if Rxz_shoulder is a valid rotation matrix
        if not np.allclose(np.linalg.det(Rxz_shoulder), 1.0):
            self.get_logger().error("Invalid rotation matrix from shoulder frame y-axis to upper arm vector.")
            return
        else:
            # self.get_logger().info(f"Rotation matrix from shoulder frame y-axis to upper arm vector:\n{Rxz_shoulder}")
            rxz = R.from_matrix(Rxz_shoulder)
            theta_shoulder_x, theta_y_temp, theta_shoulder_z = rxz.as_euler('xzy', degrees=True)
            # self.get_logger().info(f"Shoulder angles: x: {theta_shoulder_x:.2f} degrees, z: {theta_shoulder_z:.2f} degrees")

        #  calculate shoulder y rotation based on direction of the elbow
        v_normalized = normalize(v)
        # Project v onto plane orthogonal to u
        v_proj = v_normalized - np.dot(v_normalized, u_normalized) * u_normalized
        
        if np.linalg.norm(v_proj) < 1e-6:
            self.get_logger().warn("Elbow is aligned with the upper arm, cannot determine shoulder y angle.")
            theta_shoulder_y = 0.0
            
        else:
            v_proj_normalized = normalize(v_proj)
            
            # reference Rxz with y = 0
            Rxz_shoulder_ref = R.from_euler('xzy', [theta_shoulder_x, theta_shoulder_z, 0.0], degrees=True).as_matrix()

            x_elbow_ref = Rxz_shoulder_ref @ np.array([1, 0, 0])  # x-axis in the shoulder frame

            angle = np.arccos(np.clip(np.dot(x_elbow_ref, v_proj_normalized), -1.0, 1.0))
            if np.dot(v_proj_normalized, u0) < 0:
                angle = -angle
            theta_shoulder_y = np.degrees(angle)

        return theta_shoulder_x, theta_shoulder_y, theta_shoulder_z, theta_elbow_deg    
    
    def physical_constraints(self, theta_shoulder_x, theta_shoulder_y, theta_shoulder_z, theta_elbow_deg):
        # this function checks if the calculated joint angles are within the physical constraints of the robot arm,
        # nmaely 1. if the angles are within the range of motion of the joints
        # 2. if the motion is continuous, i.e. limit the change of angles between frames
        return

    def pose_callback(self, msg: PoseArray):
        
        self.dt = (self.time - self.get_clock().now()).nanoseconds * 1e-9  # convert to seconds
        self.time = self.get_clock().now()

        if len(msg.poses) < 3:
            # self.get_logger().warn("Received insufficient poses for IK calculation.")
            return
        
        # Extract the positions of the wrist, elbow, and shoulder from the PoseArray
        wrist_pose = msg.poses[0]
        self.wrist_translation = np.array([
            wrist_pose.position.x,
            wrist_pose.position.y,
            wrist_pose.position.z
        ])

        elbow_pose = msg.poses[1]
        self.elbow_translation = np.array([
            elbow_pose.position.x,
            elbow_pose.position.y,
            elbow_pose.position.z
        ])

        shoulder_pose = msg.poses[2]
        self.shoulder_translation = np.array([
            shoulder_pose.position.x,
            shoulder_pose.position.y,
            shoulder_pose.position.z
        ])
        self.get_logger().info(f"Received shoulder: {self.shoulder_translation}")
        # Transform the wrist, elbow, and shoulder positions to the shoulder frame
        self.wrist_translation = self.transform_to_shoulder_frame(self.wrist_translation)
        self.elbow_translation = self.transform_to_shoulder_frame(self.elbow_translation)
        self.shoulder_translation = self.transform_to_shoulder_frame(self.shoulder_translation)

        # Calculate the upper arm vector u and lower arm vector v
        u = self.elbow_translation - self.shoulder_translation
        v = self.wrist_translation - self.elbow_translation


        theta_shoulder_x, theta_shoulder_y, theta_shoulder_z, theta_elbow_deg = self.get_4DOF_joint_angles(u, v)

        # self.get_logger().info(f"palm reach length: {np.linalg.norm(w):.2f} m")
        # solve last three degreee of freedom with w and v
        # theta_wrist_x, theta_wrist_y, theta_wrist_z = self.get_3DOF_joint_angles(v, w)
        theta_wrist_x, theta_wrist_y, theta_wrist_z = 0.0, 0.0, 0.0
        # Publish the joint states
        self.publish_joint_states(theta_shoulder_x, theta_shoulder_y, theta_shoulder_z, theta_elbow_deg, theta_wrist_x, theta_wrist_y, theta_wrist_z)




def main(args=None):
    rclpy.init(args=args)
    arm_solver_node = Arm_Solver_Node()
    rclpy.spin(arm_solver_node)
    arm_solver_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()