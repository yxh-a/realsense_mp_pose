# flake8: noqa
"""Right-arm inverse kinematics from 3D shoulder, elbow, and wrist points.

Input topic:
    /joint_positions (geometry_msgs/PoseArray)
    Pose order is [right_wrist, right_elbow, right_shoulder, left_shoulder].

Output topic:
    arm/joint_states (std_msgs/Float32MultiArray)

Publishes radians in the order expected by right_arm_osim_shoulder.urdf.xacro:
    [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion,
     forearm_prosup, wrist_x, wrist_z, too_close]
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray
from tf2_geometry_msgs import TransformStamped
from tf2_ros import TransformBroadcaster


POSE_INDEX_WRIST = 0
POSE_INDEX_ELBOW = 1
POSE_INDEX_RIGHT_SHOULDER = 2
POSE_INDEX_LEFT_SHOULDER = 3
MIN_TRACKED_SEGMENT_LENGTH = 0.15

CAMERA_TO_SHOULDER_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0],   # x_shoulder = -z_camera
        [0.0, -1.0, 0.0],   # y_shoulder = -y_camera
        [-1.0, 0.0, 0.0],   # z_shoulder = -x_camera
    ]
)


@dataclass(frozen=True)
class OsimShoulderModel:
    """Axes and limits copied from the OSIM shoulder representation."""

    shoulder_axis: np.ndarray
    elevation_axis: np.ndarray
    upper_arm_reference: np.ndarray
    elbow_axis: np.ndarray
    limits: Dict[str, Tuple[float, float]]


OSIM_SHOULDER = OsimShoulderModel(
    shoulder_axis=np.array([0.0048, 0.99908918, 0.04240001]),
    elevation_axis=np.array([-0.99826136, 0.0023, 0.05889802]),
    upper_arm_reference=np.array([0.0, -1.0, 0.0]),
    elbow_axis=np.array([0.0, 0.0, 1.0]),
    limits={
        "elv_angle": (-1.65806279, 2.26892803),
        "shoulder_elv": (0.0, np.pi),
        "shoulder_rot": (-1.57079633, 2.09439510),
        "elbow_flexion": (0.0, 2.53073),
        "prosup": (-1.5708, 1.48353),
        "wrist_x": (-0.872665, 1.0472),
        "wrist_z": (-0.523599, 0.349066),
    },
)

ANGLE_FILTER_KEYS = (
    "shoulder_0",
    "shoulder_1",
    "shoulder_2",
    "elbow",
    "prosup",
    "wrist_x",
    "wrist_z",
)


def normalize(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return a unit vector and raise for near-zero input."""
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def safe_normalize(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray | None:
    """Return a unit vector, or None when the input is too small."""
    norm = np.linalg.norm(vector)
    if norm < eps:
        return None
    return vector / norm


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for rotation around an arbitrary axis."""
    return R.from_rotvec(angle * normalize(np.asarray(axis, dtype=float))).as_matrix()


def pose_to_array(poses, index: int) -> np.ndarray:
    pose = poses[index]
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)


class Arm_Solver_Node(Node):
    """ROS node that converts tracked arm keypoints into URDF joint values."""

    def __init__(self):
        super().__init__("forward_solver_node")

        self.pose_subscriber = self.create_subscription(
            PoseArray,
            "/joint_positions",
            self.pose_callback,
            10,
        )
        self.joint_state_publisher = self.create_publisher(Float32MultiArray, "arm/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.angle_filter_size = 5
        self.angle_buffers = {key: [] for key in ANGLE_FILTER_KEYS}
        self.shoulder_transform_buffer: List[Tuple[np.ndarray, float]] = []
        self.shoulder_transform_buffer_size = 10

        self.old_osim_solution = np.zeros(4)
        self.points_too_close = False
        self.previous_time = self.get_clock().now()
        self.dt = 0.0
        self.last_short_segment_warn_ns = -1
        self.last_short_shoulder_warn_ns = -1

        self.get_logger().info("OSIM IK Solver Node initialized.")

    # -------------------------------------------------------------------------
    # ROS and frame handling
    # -------------------------------------------------------------------------

    def transform_to_shoulder_frame(self, point_camera: np.ndarray) -> np.ndarray:
        return CAMERA_TO_SHOULDER_ROTATION @ point_camera

    def warn_throttled(self, attr_name: str, message: str, period_sec: float = 2.0) -> None:
        """Log a recurring warning at most once per period."""
        now_ns = self.get_clock().now().nanoseconds
        last_ns = getattr(self, attr_name)
        if last_ns < 0 or (now_ns - last_ns) * 1e-9 >= period_sec:
            self.get_logger().warn(message)
            setattr(self, attr_name, now_ns)

    def pose_callback(self, msg: PoseArray) -> None:
        """Handle one tracked arm pose message."""
        self._update_dt()

        if len(msg.poses) < 4:
            self.get_logger().warn("Received insufficient poses for IK calculation.")
            return

        wrist_camera = pose_to_array(msg.poses, POSE_INDEX_WRIST)
        elbow_camera = pose_to_array(msg.poses, POSE_INDEX_ELBOW)
        right_shoulder_camera = pose_to_array(msg.poses, POSE_INDEX_RIGHT_SHOULDER)
        left_shoulder_camera = pose_to_array(msg.poses, POSE_INDEX_LEFT_SHOULDER)

        self.publish_shoulder_transform(right_shoulder_camera, left_shoulder_camera)

        wrist = self.transform_to_shoulder_frame(wrist_camera)
        elbow = self.transform_to_shoulder_frame(elbow_camera)
        right_shoulder = self.transform_to_shoulder_frame(right_shoulder_camera)
        left_shoulder = self.transform_to_shoulder_frame(left_shoulder_camera)

        upper_arm = elbow - right_shoulder
        forearm = wrist - elbow
        shoulder_line = left_shoulder - right_shoulder

        if self._segments_too_short(upper_arm, forearm, shoulder_line):
            self.warn_throttled(
                'last_short_segment_warn_ns',
                'One or more vectors for joint angle calculation are too short; '
                'publishing too_close and skipping IK for this frame.',
            )
            self.points_too_close = True
            self.publish_current_or_neutral_state()
            return

        self._solve_and_publish(upper_arm, forearm)

    def _update_dt(self) -> None:
        now = self.get_clock().now()
        self.dt = (now - self.previous_time).nanoseconds * 1e-9
        self.previous_time = now

    def _segments_too_short(self, *segments: Iterable[np.ndarray]) -> bool:
        return any(np.linalg.norm(segment) < MIN_TRACKED_SEGMENT_LENGTH for segment in segments)

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def _solve_and_publish(self, upper_arm: np.ndarray, forearm: np.ndarray) -> None:
        prosup = 0.0
        wrist_x = 0.0
        wrist_z = 0.0

        elv_angle, shoulder_elv, shoulder_rot, elbow_flexion = self.solve_osim_arm(upper_arm, forearm)
        self.publish_osim_joint_states(
            elv_angle,
            shoulder_elv,
            shoulder_rot,
            elbow_flexion,
            prosup,
            wrist_x,
            wrist_z,
        )

    def publish_osim_joint_states(
        self,
        elv_angle: float,
        shoulder_elv: float,
        shoulder_rot: float,
        elbow_flexion: float,
        prosup: float,
        wrist_x: float,
        wrist_z: float,
    ) -> None:
        """Publish OSIM shoulder values in radians."""
        values = self._filtered_values(
            [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion, prosup, wrist_x, wrist_z]
        )
        self._publish_joint_array(values)

    def _filtered_values(self, values: Iterable[float]) -> List[float]:
        filtered = []
        for key, value in zip(ANGLE_FILTER_KEYS, values):
            buffer = self.angle_buffers[key]
            buffer.append(float(value))
            if len(buffer) > self.angle_filter_size:
                buffer.pop(0)
            filtered.append(float(np.mean(buffer)))
        return filtered

    def _publish_joint_array(self, joint_values: List[float]) -> None:
        too_close = 1.0 if self.points_too_close else 0.0
        self.points_too_close = False

        msg = Float32MultiArray()
        msg.data = joint_values + [too_close]
        self.joint_state_publisher.publish(msg)

    def publish_current_or_neutral_state(self) -> None:
        """Publish the last filtered values, or neutral zeros before first solve."""
        if all(self.angle_buffers[key] for key in ANGLE_FILTER_KEYS):
            values = [
                float(np.mean(self.angle_buffers[key]))
                for key in ANGLE_FILTER_KEYS
            ]
        else:
            values = [0.0] * len(ANGLE_FILTER_KEYS)
        self._publish_joint_array(values)

    # -------------------------------------------------------------------------
    # OSIM shoulder IK
    # -------------------------------------------------------------------------

    def osim_shoulder_rotation(self, elv_angle: float, shoulder_elv: float, shoulder_rot: float) -> np.ndarray:
        """Forward shoulder rotation for right_arm_osim_shoulder.urdf.xacro."""
        return (
            axis_angle_rotation(OSIM_SHOULDER.shoulder_axis, elv_angle)
            @ axis_angle_rotation(OSIM_SHOULDER.elevation_axis, shoulder_elv)
            @ axis_angle_rotation(OSIM_SHOULDER.shoulder_axis, -elv_angle)
            @ axis_angle_rotation(OSIM_SHOULDER.shoulder_axis, shoulder_rot)
        )

    def solve_osim_arm(self, upper_arm: np.ndarray, forearm: np.ndarray) -> np.ndarray:
        """Solve [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion].

        The upper-arm vector alone only constrains the humerus direction. The axial
        shoulder_rot is resolved by also fitting the observed forearm direction.
        """
        upper_dir = safe_normalize(upper_arm)
        forearm_dir = safe_normalize(forearm)
        if upper_dir is None or forearm_dir is None:
            self.get_logger().error("Invalid vectors for OSIM joint angle calculation.")
            return self.old_osim_solution

        elbow_guess = self._angle_between(upper_dir, forearm_dir)
        lower, upper = self._osim_solver_bounds()
        initial_guess = self._osim_initial_guess(upper_dir, elbow_guess, lower, upper)

        result = least_squares(
            lambda q: self._osim_residual(q, upper_dir, forearm_dir, elbow_guess),
            initial_guess,
            bounds=(lower, upper),
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
            max_nfev=60,
        )
        if not result.success:
            self.get_logger().warn(f"OSIM shoulder solve did not fully converge: {result.message}")

        self.get_logger().info(
            "OSIM initial guess deg: "
            f"{np.degrees(initial_guess[:4]).round(2).tolist()}, "
            "final result deg: "
            f"{np.degrees(result.x[:4]).round(2).tolist()}"
        )

        self.old_osim_solution = result.x
        return result.x

    def _osim_solver_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        names = ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion")
        lower = np.array([OSIM_SHOULDER.limits[name][0] for name in names])
        upper = np.array([OSIM_SHOULDER.limits[name][1] for name in names])
        return lower, upper

    def _osim_initial_guess(
        self,
        upper_dir: np.ndarray,
        elbow_guess: float,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        initial_guess = self.old_osim_solution.copy()
        # initial_guess[0] = np.arctan2(upper_dir[0], upper_dir[2])
        initial_guess[0] = np.sign(upper_dir[0]) * np.arctan2(np.abs(upper_dir[0]), np.abs(upper_dir[2]))
        # initial_guess[1] = -np.arctan2(upper_dir[2], -upper_dir[1])
        if upper_dir[1] < 0.0:
            initial_guess[1] = np.arctan2(upper_dir[2], -upper_dir[1])
        else:
            initial_guess[1] = (np.pi - np.arctan2(upper_dir[2], upper_dir[1]))

        initial_guess[3] = elbow_guess
        return np.clip(initial_guess, lower, upper)

    def _osim_residual(
        self,
        q: np.ndarray,
        upper_dir: np.ndarray,
        forearm_dir: np.ndarray,
        elbow_guess: float,
    ) -> np.ndarray:
        elv_angle, shoulder_elv, shoulder_rot, elbow_flexion = q
        shoulder_rotation = self.osim_shoulder_rotation(elv_angle, shoulder_elv, shoulder_rot)
        elbow_rotation = axis_angle_rotation(OSIM_SHOULDER.elbow_axis, elbow_flexion)

        upper_pred = normalize(shoulder_rotation @ OSIM_SHOULDER.upper_arm_reference)
        forearm_pred = normalize(shoulder_rotation @ elbow_rotation @ OSIM_SHOULDER.upper_arm_reference)

        return np.concatenate(
            (
                3.0 * (upper_pred - upper_dir),
                forearm_pred - forearm_dir,
                [0.15 * (elbow_flexion - elbow_guess)],
            )
        )

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.arccos(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0)))

    def get_osim_4DOF_joint_angles(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.solve_osim_arm(u, v)

    # -------------------------------------------------------------------------
    # Shoulder frame TF
    # -------------------------------------------------------------------------

    def make_tf(self, parent: str, child: str, xyz: np.ndarray, quat_xyzw: np.ndarray) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = float(xyz[0])
        transform.transform.translation.y = float(xyz[1])
        transform.transform.translation.z = float(xyz[2])
        transform.transform.rotation.x = float(quat_xyzw[0])
        transform.transform.rotation.y = float(quat_xyzw[1])
        transform.transform.rotation.z = float(quat_xyzw[2])
        transform.transform.rotation.w = float(quat_xyzw[3])
        return transform

    def publish_shoulder_transform(self, right_shoulder_camera: np.ndarray, left_shoulder_camera: np.ndarray) -> None:
        """Publish the right-shoulder frame used by the URDF visualization."""
        shoulder_line = left_shoulder_camera - right_shoulder_camera
        shoulder_line_unit = safe_normalize(shoulder_line)
        if shoulder_line_unit is None:
            self.warn_throttled(
                'last_short_shoulder_warn_ns',
                'Shoulder line is too short; not updating shoulder TF.',
            )
            return

        shoulder_line_xz = safe_normalize(np.array([shoulder_line_unit[0], 0.0, shoulder_line_unit[2]]))
        if shoulder_line_xz is None:
            yaw_correction = 0.0
        else:
            yaw_correction = -np.arctan2(shoulder_line_xz[2], shoulder_line_xz[0])
            if shoulder_line_unit[2] < 0.0:
                yaw_correction = -yaw_correction

        self.shoulder_transform_buffer.append((right_shoulder_camera, yaw_correction))
        if len(self.shoulder_transform_buffer) > self.shoulder_transform_buffer_size:
            self.shoulder_transform_buffer.pop(0)

        avg_translation = np.mean([entry[0] for entry in self.shoulder_transform_buffer], axis=0)
        avg_yaw_correction = np.mean([entry[1] for entry in self.shoulder_transform_buffer])
        avg_quat = R.from_euler("xyz", [3.14, 1.57 + avg_yaw_correction, 0.0]).as_quat()

        transforms = [
            self.make_tf("camera_depth_optical_frame", child, avg_translation, avg_quat)
            # for child in ("RightShoulder", "gt_RightShoulder", "upt_RightShoulder")
            # for child in ("RightShoulder", "upt_RightShoulder")
            for child in ("RightShoulder",)
        ]
        self.tf_broadcaster.sendTransform(transforms)

    def publish_shoudler_transform(self) -> None:
        """Deprecated typo-preserving wrapper kept for older local calls."""
        self.get_logger().warn("publish_shoudler_transform() is deprecated; use publish_shoulder_transform().")


def main(args=None):
    rclpy.init(args=args)
    arm_solver_node = Arm_Solver_Node()
    rclpy.spin(arm_solver_node)
    arm_solver_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
