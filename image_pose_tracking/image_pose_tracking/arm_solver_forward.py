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
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray, Int16MultiArray
from tf2_geometry_msgs import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


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

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('keypoints_topic', '/keypoints_2d')
        self.declare_parameter('point_cloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('joint_positions_topic', '/joint_positions')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('sample_radius_px', 2)
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('min_valid_samples', 3)
        self.declare_parameter('joint_center_offset_x', 0.0)
        self.declare_parameter('joint_center_offset_y', 0.0)
        self.declare_parameter('joint_center_offset_z', 0.05)
        self.declare_parameter('keypoint_offsets.wrist', [0.0, 0.0, 0.0])
        self.declare_parameter('keypoint_offsets.elbow', [0.0, 0.0, 0.05])
        self.declare_parameter('keypoint_offsets.shoulder', [0.05, 0.0, 0.0])
        self.declare_parameter('lateral_variance_floor', 1.0e-4)
        self.declare_parameter('depth_variance_floor', 4.0e-4)
        self.declare_parameter('confidence_variance_scale', 2.0)
        self.declare_parameter('orientation_variance', 1.0e6)
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('marker_sigma_scale', 2.0)
        self.declare_parameter('marker_min_radius', 0.01)
        self.declare_parameter('marker_max_radius', 0.30)
        self.declare_parameter('shoulder_transform_buffer_size', 10)
        self.declare_parameter('shoulder_translation_gate.enabled', True)
        self.declare_parameter('shoulder_translation_gate.calibration_duration_sec', 3.0)
        self.declare_parameter('shoulder_translation_gate.reject_distance_m', 0.20)
        self.declare_parameter('shoulder_translation_gate.max_calibration_std_m', 0.05)
        self.declare_parameter('shoulder_translation_gate.min_calibration_samples', 10)
        self.declare_parameter('keypoints_queue_depth', 1)
        self.declare_parameter('point_cloud_queue_depth', 1)
        self.declare_parameter('joint_positions_queue_depth', 10)
        self.declare_parameter('max_processing_rate_hz', 15.0)
        self.declare_parameter('correct_shoulder_yaw', False)

        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        keypoints_topic = self.get_parameter('keypoints_topic').get_parameter_value().string_value
        point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value
        joint_positions_topic = self.get_parameter('joint_positions_topic').get_parameter_value().string_value
        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value
        self.sample_radius_px = self.get_parameter('sample_radius_px').get_parameter_value().integer_value
        self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().double_value
        self.min_valid_samples = self.get_parameter('min_valid_samples').get_parameter_value().integer_value
        shared_offset = np.array(
            [
                self.get_parameter('joint_center_offset_x').get_parameter_value().double_value,
                self.get_parameter('joint_center_offset_y').get_parameter_value().double_value,
                self.get_parameter('joint_center_offset_z').get_parameter_value().double_value,
            ],
            dtype=float,
        )
        wrist_offset = self.vector3_parameter('keypoint_offsets.wrist', shared_offset)
        elbow_offset = self.vector3_parameter('keypoint_offsets.elbow', shared_offset)
        shoulder_offset = self.vector3_parameter('keypoint_offsets.shoulder', shared_offset)
        self.keypoint_offsets = {
            'right_wrist': wrist_offset,
            'right_elbow': elbow_offset,
            'right_shoulder': shoulder_offset,
            'left_shoulder': shoulder_offset,
        }
        self.lateral_variance_floor = self.get_parameter('lateral_variance_floor').get_parameter_value().double_value
        self.depth_variance_floor = self.get_parameter('depth_variance_floor').get_parameter_value().double_value
        self.confidence_variance_scale = self.get_parameter('confidence_variance_scale').get_parameter_value().double_value
        self.orientation_variance = self.get_parameter('orientation_variance').get_parameter_value().double_value
        self.publish_markers = self.get_parameter('publish_markers').get_parameter_value().bool_value
        self.marker_sigma_scale = self.get_parameter('marker_sigma_scale').get_parameter_value().double_value
        self.marker_min_radius = self.get_parameter('marker_min_radius').get_parameter_value().double_value
        self.marker_max_radius = self.get_parameter('marker_max_radius').get_parameter_value().double_value
        self.shoulder_transform_buffer_size = (
            self.get_parameter('shoulder_transform_buffer_size').get_parameter_value().integer_value
        )
        self.shoulder_translation_gate_enabled = (
            self.get_parameter('shoulder_translation_gate.enabled').get_parameter_value().bool_value
        )
        self.shoulder_gate_calibration_duration_sec = (
            self.get_parameter('shoulder_translation_gate.calibration_duration_sec').get_parameter_value().double_value
        )
        self.shoulder_gate_reject_distance_m = (
            self.get_parameter('shoulder_translation_gate.reject_distance_m').get_parameter_value().double_value
        )
        self.shoulder_gate_max_calibration_std_m = (
            self.get_parameter('shoulder_translation_gate.max_calibration_std_m').get_parameter_value().double_value
        )
        self.shoulder_gate_min_calibration_samples = (
            self.get_parameter('shoulder_translation_gate.min_calibration_samples').get_parameter_value().integer_value
        )
        keypoints_queue_depth = self.get_parameter('keypoints_queue_depth').get_parameter_value().integer_value
        point_cloud_queue_depth = self.get_parameter('point_cloud_queue_depth').get_parameter_value().integer_value
        joint_positions_queue_depth = self.get_parameter('joint_positions_queue_depth').get_parameter_value().integer_value
        self.max_processing_rate_hz = (
            self.get_parameter('max_processing_rate_hz').get_parameter_value().double_value
        )
        self.correct_shoulder_yaw = self.get_parameter('correct_shoulder_yaw').get_parameter_value().bool_value

        self.pose_subscriber = self.create_subscription(
            PoseArray,
            joint_positions_topic,
            self.pose_callback,
            max(1, joint_positions_queue_depth),
        )
        self.kp_subscriber = self.create_subscription(
            Int16MultiArray,
            keypoints_topic,
            self.keypoints_callback,
            max(1, keypoints_queue_depth),
        )
        self.point_cloud_subscriber = self.create_subscription(
            PointCloud2,
            point_cloud_topic,
            self.point_cloud_callback,
            max(1, point_cloud_queue_depth),
        )
        self.joint_state_publisher = self.create_publisher(Float32MultiArray, "arm/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.distribution_publishers = {
            name: self.create_publisher(PoseWithCovarianceStamped, f'/keypoint_distributions/{name}', 10)
            for name in ('right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder')
        }
        self.marker_pub = self.create_publisher(MarkerArray, '/keypoint_distribution_markers', 10)

        self.angle_filter_size = 5
        self.angle_buffers = {key: [] for key in ANGLE_FILTER_KEYS}
        self.shoulder_transform_buffer: List[Tuple[np.ndarray, float]] = []

        self.keypoint_names = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder']
        self.keypoints = [None] * len(self.keypoint_names)
        self.point_cloud = None
        self.shoulder_gate_start_ns = None
        self.shoulder_gate_samples = []
        self.shoulder_gate_mean = None
        self.shoulder_gate_calibrated = False
        self.shoulder_gate_stable = False
        self.old_osim_solution = np.zeros(4)
        self.points_too_close = False
        self.previous_time = self.get_clock().now()
        self.dt = 0.0
        self.last_short_segment_warn_ns = -1
        self.last_short_shoulder_warn_ns = -1
        self.last_missing_keypoints_warn_ns = -1
        self.last_shoulder_gate_outlier_warn_ns = -1
        self.last_shoulder_gate_time_reset_warn_ns = -1
        self.last_processed_cloud_stamp_ns = None

        self.get_logger().info(
            f"OSIM IK solver initialized. Publishing distributions from {point_cloud_topic} "
            f"and rough q0-q4 estimates on arm/joint_states."
        )

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

    def keypoints_callback(self, msg: Int16MultiArray) -> None:
        expected_values = len(self.keypoint_names) * 3
        if len(msg.data) < expected_values:
            self.get_logger().warn(
                f"Received incomplete keypoint message with {len(msg.data)} values; expected {expected_values}."
            )
            return

        for i in range(len(self.keypoint_names)):
            self.keypoints[i] = np.array(
                [
                    msg.data[i * 3],
                    msg.data[i * 3 + 1],
                    msg.data[i * 3 + 2] / 100.0,
                ],
                dtype=float,
            )

    def point_cloud_callback(self, msg: PointCloud2) -> None:
        if not self.should_process_cloud(msg.header.stamp):
            return

        if any(kp is None for kp in self.keypoints):
            self.warn_throttled(
                'last_missing_keypoints_warn_ns',
                'Keypoints are not available or incomplete.',
            )
            return

        self._update_dt()
        self.point_cloud = msg

        marker_array = MarkerArray()
        distributions = {}
        for i, (name, keypoint) in enumerate(zip(self.keypoint_names, self.keypoints)):
            distribution = self.distribution_from_keypoint(name, keypoint)
            if distribution is None:
                continue
            mean, covariance = distribution
            distributions[name] = distribution
            self.publish_distribution(name, msg.header.stamp, mean, covariance)
            if self.publish_markers:
                marker_array.markers.append(self.make_distribution_marker(i, name, msg.header.stamp, mean, covariance))

        if self.publish_markers:
            self.marker_pub.publish(marker_array)

        required = ('right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder')
        if not all(name in distributions for name in required):
            return

        wrist_camera = distributions['right_wrist'][0]
        elbow_camera = distributions['right_elbow'][0]
        right_shoulder_camera = distributions['right_shoulder'][0]
        left_shoulder_camera = distributions['left_shoulder'][0]

        self.publish_shoulder_transform(right_shoulder_camera, left_shoulder_camera, msg.header.stamp)
        self.solve_from_camera_points(
            wrist_camera,
            elbow_camera,
            right_shoulder_camera,
            left_shoulder_camera,
        )

    def solve_from_camera_points(
        self,
        wrist_camera: np.ndarray,
        elbow_camera: np.ndarray,
        right_shoulder_camera: np.ndarray,
        left_shoulder_camera: np.ndarray,
    ) -> None:
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

        self.publish_shoulder_transform(right_shoulder_camera, left_shoulder_camera, msg.header.stamp)
        self.solve_from_camera_points(
            wrist_camera,
            elbow_camera,
            right_shoulder_camera,
            left_shoulder_camera,
        )

    def _update_dt(self) -> None:
        now = self.get_clock().now()
        self.dt = (now - self.previous_time).nanoseconds * 1e-9
        self.previous_time = now

    def _segments_too_short(self, *segments: Iterable[np.ndarray]) -> bool:
        return any(np.linalg.norm(segment) < MIN_TRACKED_SEGMENT_LENGTH for segment in segments)

    def should_process_cloud(self, stamp) -> bool:
        if self.max_processing_rate_hz <= 0.0:
            return True

        stamp_ns = self.stamp_to_ns(stamp)
        if stamp_ns <= 0:
            stamp_ns = self.get_clock().now().nanoseconds

        if self.last_processed_cloud_stamp_ns is None:
            self.last_processed_cloud_stamp_ns = stamp_ns
            return True

        min_period_ns = int(1_000_000_000.0 / self.max_processing_rate_hz)
        if stamp_ns - self.last_processed_cloud_stamp_ns < min_period_ns:
            return False

        self.last_processed_cloud_stamp_ns = stamp_ns
        return True

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
    # Keypoint distributions from 2D MediaPipe pixels and depth point cloud
    # -------------------------------------------------------------------------

    def distribution_from_keypoint(self, name: str, keypoint: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
        u = int(round(keypoint[0]))
        v = int(round(keypoint[1]))
        confidence = float(keypoint[2])

        if confidence < self.min_confidence:
            return None

        samples = self.samples_from_uv(u, v)
        if samples.shape[0] < self.min_valid_samples:
            self.get_logger().warn(
                f"Only {samples.shape[0]} valid depth samples near keypoint ({u}, {v}); skipping."
            )
            return None

        mean = np.mean(samples, axis=0)
        mean += self.keypoint_offsets.get(name, np.zeros(3, dtype=float))

        covariance = np.cov(samples, rowvar=False)
        if covariance.ndim == 0:
            covariance = np.zeros((3, 3), dtype=float)

        covariance[0, 0] = max(float(covariance[0, 0]), self.lateral_variance_floor)
        covariance[1, 1] = max(float(covariance[1, 1]), self.lateral_variance_floor)
        covariance[2, 2] = max(float(covariance[2, 2]), self.depth_variance_floor)

        confidence_inflation = 1.0 + self.confidence_variance_scale * (1.0 - confidence)
        covariance *= confidence_inflation
        return mean, covariance

    def samples_from_uv(self, u: int, v: int) -> np.ndarray:
        uvs = []
        for du in range(-self.sample_radius_px, self.sample_radius_px + 1):
            for dv in range(-self.sample_radius_px, self.sample_radius_px + 1):
                sample_u = u + du
                sample_v = v + dv
                if 0 <= sample_u < self.image_width and 0 <= sample_v < self.image_height:
                    uvs.append(sample_u + sample_v * self.image_width)

        points = pc2.read_points(
            self.point_cloud,
            field_names=('x', 'y', 'z'),
            uvs=uvs,
            skip_nans=True,
        )
        return np.array(
            [
                [point[0], point[1], point[2]]
                for point in points
                if self.is_valid_point(point)
            ],
            dtype=float,
        )

    def publish_distribution(self, name, stamp, mean, covariance) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.pose.pose.position.x = float(mean[0])
        msg.pose.pose.position.y = float(mean[1])
        msg.pose.pose.position.z = float(mean[2])
        msg.pose.pose.orientation.w = 1.0

        full_covariance = np.zeros((6, 6), dtype=float)
        full_covariance[:3, :3] = covariance
        full_covariance[3, 3] = self.orientation_variance
        full_covariance[4, 4] = self.orientation_variance
        full_covariance[5, 5] = self.orientation_variance
        msg.pose.covariance = full_covariance.reshape(-1).tolist()

        self.distribution_publishers[name].publish(msg)

    def make_distribution_marker(self, marker_id, name, stamp, mean, covariance) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.camera_frame
        marker.ns = 'keypoint_distribution'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(mean[0])
        marker.pose.position.y = float(mean[1])
        marker.pose.position.z = float(mean[2])
        marker.pose.orientation.w = 1.0

        radius = self.marker_radius_from_covariance(covariance)
        diameter = 2.0 * radius
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter

        marker.color.a = 0.35
        marker.color.r = 1.0
        marker.color.g = 0.65
        marker.color.b = 0.0
        marker.text = name
        return marker

    def marker_radius_from_covariance(self, covariance: np.ndarray) -> float:
        max_variance = float(np.max(np.linalg.eigvalsh(covariance[:3, :3])))
        radius = self.marker_sigma_scale * np.sqrt(max(max_variance, 0.0))
        return float(np.clip(radius, self.marker_min_radius, self.marker_max_radius))

    def vector3_parameter(self, name: str, fallback: np.ndarray) -> np.ndarray:
        values = list(self.get_parameter(name).value)
        if len(values) != 3:
            self.get_logger().warn(
                f"Parameter {name} must contain exactly three values; using fallback {fallback.tolist()}."
            )
            return np.array(fallback, dtype=float)
        return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)

    @staticmethod
    def is_valid_point(point) -> bool:
        return (
            np.isfinite(point[0])
            and np.isfinite(point[1])
            and np.isfinite(point[2])
            and not (point[0] == 0.0 and point[1] == 0.0 and point[2] == 0.0)
        )

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

        # self.get_logger().info(
        #     "OSIM initial guess deg: "
        #     f"{np.degrees(initial_guess[:4]).round(2).tolist()}, "
        #     "final result deg: "
        #     f"{np.degrees(result.x[:4]).round(2).tolist()}"
        # )

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

    def make_tf(self, parent: str, child: str, xyz: np.ndarray, quat_xyzw: np.ndarray, stamp=None) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
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

    def publish_shoulder_transform(self, right_shoulder_camera: np.ndarray, left_shoulder_camera: np.ndarray, stamp=None) -> None:
        """Publish the right-shoulder frame used by the URDF visualization."""
        right_shoulder_camera = self.filtered_shoulder_translation(right_shoulder_camera, stamp)

        if not self.correct_shoulder_yaw:
            yaw_correction = 0.0
        else:
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
            self.make_tf(self.camera_frame, child, avg_translation, avg_quat, stamp)
            for child in ("RightShoulder", "gt_RightShoulder", "upt_RightShoulder")
        ]
        self.tf_broadcaster.sendTransform(transforms)

    def filtered_shoulder_translation(self, right_shoulder_camera: np.ndarray, stamp) -> np.ndarray:
        if not self.shoulder_translation_gate_enabled:
            return right_shoulder_camera

        stamp_ns = self.stamp_to_ns(stamp)
        if stamp_ns <= 0:
            stamp_ns = self.get_clock().now().nanoseconds

        if self.shoulder_gate_start_ns is None:
            self.shoulder_gate_start_ns = stamp_ns

        elapsed_sec = (stamp_ns - self.shoulder_gate_start_ns) * 1e-9
        if elapsed_sec < -0.5:
            self.warn_throttled(
                'last_shoulder_gate_time_reset_warn_ns',
                'Shoulder translation gate saw time move backwards; restarting initial calibration.',
            )
            self.reset_shoulder_translation_gate(stamp_ns)
            elapsed_sec = 0.0

        if not self.shoulder_gate_calibrated:
            self.shoulder_gate_samples.append(np.array(right_shoulder_camera, dtype=float))
            if elapsed_sec >= self.shoulder_gate_calibration_duration_sec:
                self.finish_shoulder_translation_gate_calibration()
            return right_shoulder_camera

        if not self.shoulder_gate_stable or self.shoulder_gate_mean is None:
            return right_shoulder_camera

        distance_from_initial_mean = float(np.linalg.norm(right_shoulder_camera - self.shoulder_gate_mean))
        if distance_from_initial_mean > self.shoulder_gate_reject_distance_m:
            self.warn_throttled(
                'last_shoulder_gate_outlier_warn_ns',
                (
                    'Rejecting right shoulder translation for TF: '
                    f'{distance_from_initial_mean:.3f} m from initial mean '
                    f'(limit {self.shoulder_gate_reject_distance_m:.3f} m).'
                ),
            )
            return self.shoulder_gate_mean.copy()

        return right_shoulder_camera

    def finish_shoulder_translation_gate_calibration(self) -> None:
        sample_count = len(self.shoulder_gate_samples)
        if sample_count < self.shoulder_gate_min_calibration_samples:
            self.get_logger().warn(
                'Shoulder translation gate did not receive enough initial samples '
                f'({sample_count}/{self.shoulder_gate_min_calibration_samples}); leaving gate disabled.'
            )
            self.shoulder_gate_calibrated = True
            self.shoulder_gate_stable = False
            return

        samples = np.vstack(self.shoulder_gate_samples)
        self.shoulder_gate_mean = np.mean(samples, axis=0)
        axis_std = np.std(samples, axis=0)
        max_axis_std = float(np.max(axis_std))
        self.shoulder_gate_stable = max_axis_std <= self.shoulder_gate_max_calibration_std_m
        self.shoulder_gate_calibrated = True

        if self.shoulder_gate_stable:
            self.get_logger().info(
                'Shoulder translation gate calibrated: '
                f'mean=[{self.shoulder_gate_mean[0]:.3f}, '
                f'{self.shoulder_gate_mean[1]:.3f}, '
                f'{self.shoulder_gate_mean[2]:.3f}] m, '
                f'std=[{axis_std[0]:.3f}, {axis_std[1]:.3f}, {axis_std[2]:.3f}] m.'
            )
        else:
            self.get_logger().warn(
                'Initial shoulder translation was not stable enough for outlier rejection: '
                f'max std {max_axis_std:.3f} m exceeds '
                f'{self.shoulder_gate_max_calibration_std_m:.3f} m. Gate disabled.'
            )

    def reset_shoulder_translation_gate(self, start_ns: int) -> None:
        self.shoulder_gate_start_ns = start_ns
        self.shoulder_gate_samples = []
        self.shoulder_gate_mean = None
        self.shoulder_gate_calibrated = False
        self.shoulder_gate_stable = False

    @staticmethod
    def stamp_to_ns(stamp) -> int:
        if stamp is None:
            return 0
        return stamp.sec * 1_000_000_000 + stamp.nanosec

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
