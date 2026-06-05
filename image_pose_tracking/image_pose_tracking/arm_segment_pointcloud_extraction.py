import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header, Int16MultiArray
from scipy.spatial.transform import Rotation as R
from tf2_geometry_msgs import TransformStamped
from tf2_ros import TransformBroadcaster


class ArmSegmentPointCloudExtractionNode(Node):
    """Publish right-arm point-cloud segments in the right-shoulder frame.

    The node uses MediaPipe pixel keypoints to seed depth-cloud clusters:

        upper arm: right_shoulder -> right_elbow
        forearm:   right_elbow    -> right_wrist

    The default segmentation first converts shoulder, elbow, and wrist pixels to
    3D points using the same local depth sampling pattern as the older keypoint
    nodes. It then clips the original PointCloud2 records by 3D distance to each
    anatomical segment, transforms only x/y/z into the shoulder frame, and keeps
    every other field such as rgb/intensity unchanged.
    """

    def __init__(self):
        super().__init__('arm_segment_pointcloud_extraction_node')

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('shoulder_frame', 'upt_RightShoulder')
        self.declare_parameter('shoulder_tf_children', ['RightShoulder', 'gt_RightShoulder', 'upt_RightShoulder'])
        self.declare_parameter('keypoints_topic', '/keypoints_2d')
        self.declare_parameter('point_cloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('keypoint_image_width', 640)
        self.declare_parameter('keypoint_image_height', 480)
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('joint_center_offset_x', 0.0)
        self.declare_parameter('joint_center_offset_y', 0.0)
        self.declare_parameter('joint_center_offset_z', 0.0)
        self.declare_parameter('keypoint_offsets.wrist', [0.0, 0.0, 0.0])
        self.declare_parameter('keypoint_offsets.elbow', [0.0, 0.0, 0.0])
        self.declare_parameter('keypoint_offsets.shoulder', [0.0, 0.0, 0.0])
        self.declare_parameter('joint_sample_radius_px', 2)
        self.declare_parameter('cluster_max_distance_to_segment_m', 0.12)
        self.declare_parameter('cluster_endpoint_margin_m', 0.08)
        self.declare_parameter('full_cloud_stride', 1)
        self.declare_parameter('log_segment_counts', True)
        self.declare_parameter('max_points_per_segment', 3000)
        self.declare_parameter('shoulder_transform_buffer_size', 10)
        self.declare_parameter('max_processing_rate_hz', 15.0)
        self.declare_parameter('keypoints_queue_depth', 1)
        self.declare_parameter('point_cloud_queue_depth', 1)
        self.declare_parameter('log_keypoint_updates', False)

        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.shoulder_frame = self.get_parameter('shoulder_frame').get_parameter_value().string_value
        self.shoulder_tf_children = list(self.get_parameter('shoulder_tf_children').value)
        keypoints_topic = self.get_parameter('keypoints_topic').get_parameter_value().string_value
        point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value
        self.keypoint_image_width = self.get_parameter('keypoint_image_width').get_parameter_value().integer_value
        self.keypoint_image_height = self.get_parameter('keypoint_image_height').get_parameter_value().integer_value
        self.min_confidence = self.get_parameter('min_confidence').get_parameter_value().double_value
        self.joint_sample_radius_px = self.get_parameter('joint_sample_radius_px').get_parameter_value().integer_value
        self.cluster_max_distance_to_segment_m = (
            self.get_parameter('cluster_max_distance_to_segment_m').get_parameter_value().double_value
        )
        self.cluster_endpoint_margin_m = (
            self.get_parameter('cluster_endpoint_margin_m').get_parameter_value().double_value
        )
        self.full_cloud_stride = self.get_parameter('full_cloud_stride').get_parameter_value().integer_value
        self.log_segment_counts = self.get_parameter('log_segment_counts').get_parameter_value().bool_value
        self.max_points_per_segment = self.get_parameter('max_points_per_segment').get_parameter_value().integer_value
        self.shoulder_transform_buffer_size = (
            self.get_parameter('shoulder_transform_buffer_size').get_parameter_value().integer_value
        )
        self.max_processing_rate_hz = self.get_parameter('max_processing_rate_hz').get_parameter_value().double_value
        self.log_keypoint_updates = self.get_parameter('log_keypoint_updates').get_parameter_value().bool_value

        shared_offset = np.array(
            [
                self.get_parameter('joint_center_offset_x').get_parameter_value().double_value,
                self.get_parameter('joint_center_offset_y').get_parameter_value().double_value,
                self.get_parameter('joint_center_offset_z').get_parameter_value().double_value,
            ],
            dtype=float,
        )
        self.keypoint_offsets = {
            'right_wrist': self.vector3_parameter('keypoint_offsets.wrist', shared_offset),
            'right_elbow': self.vector3_parameter('keypoint_offsets.elbow', shared_offset),
            'right_shoulder': self.vector3_parameter('keypoint_offsets.shoulder', shared_offset),
            'left_shoulder': self.vector3_parameter('keypoint_offsets.shoulder', shared_offset),
        }

        self.keypoint_names = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder']
        self.keypoints = [None] * len(self.keypoint_names)
        self.point_cloud = None
        self.full_cloud_records = None
        self.cloud_width = self.keypoint_image_width
        self.cloud_height = self.keypoint_image_height
        self.shoulder_transform_buffer = []
        self.last_processed_cloud_stamp_ns = None
        self.last_missing_keypoints_warn_ns = -1
        self.last_short_shoulder_warn_ns = -1
        self.last_empty_segment_warn_ns = -1
        self.last_segment_count_log_ns = -1

        keypoints_queue_depth = self.get_parameter('keypoints_queue_depth').get_parameter_value().integer_value
        point_cloud_queue_depth = self.get_parameter('point_cloud_queue_depth').get_parameter_value().integer_value
        self.kp_sub = self.create_subscription(
            Int16MultiArray, keypoints_topic, self.keypoints_callback, max(1, keypoints_queue_depth)
        )
        self.point_cloud_sub = self.create_subscription(
            PointCloud2, point_cloud_topic, self.point_cloud_callback, max(1, point_cloud_queue_depth)
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.upper_arm_pub = self.create_publisher(PointCloud2, '/arm_segments/upper_arm', 10)
        self.forearm_pub = self.create_publisher(PointCloud2, '/arm_segments/forearm', 10)
        self.full_arm_pub = self.create_publisher(PointCloud2, '/arm_segments/right_arm', 10)

        self.get_logger().info(
            f'Arm segment pointcloud node publishing shoulder-frame segments from {point_cloud_topic}'
        )

    def keypoints_callback(self, msg):
        expected_values = len(self.keypoint_names) * 3
        if len(msg.data) < expected_values:
            self.get_logger().warn(
                f'Received incomplete keypoint message with {len(msg.data)} values; expected {expected_values}.'
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
        if self.log_keypoint_updates:
            self.get_logger().info(f'Updated keypoints: {self.keypoints}')

    def point_cloud_callback(self, msg):
        if not self.should_process_cloud(msg.header.stamp):
            return

        if any(kp is None for kp in self.keypoints):
            self.warn_throttled(
                'last_missing_keypoints_warn_ns',
                'Keypoints are not available or incomplete.',
            )
            return

        self.point_cloud = msg
        self.cloud_width = int(msg.width)
        self.cloud_height = int(msg.height)
        self.full_cloud_records = None

        keypoint_points = self.keypoint_points_from_cloud()
        if keypoint_points is None:
            return

        shoulder_transform = self.publish_shoulder_transform(
            keypoint_points['right_shoulder'],
            keypoint_points['left_shoulder'],
            msg.header.stamp,
        )
        if shoulder_transform is None:
            return

        upper_records = self.clip_segment_records_in_shoulder_frame(
            keypoint_points['right_shoulder'],
            keypoint_points['right_elbow'],
            shoulder_transform,
        )
        forearm_records = self.clip_segment_records_in_shoulder_frame(
            keypoint_points['right_elbow'],
            keypoint_points['right_wrist'],
            shoulder_transform,
        )

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = self.shoulder_frame
        upper_records = self.records_with_cloud_dtype(upper_records)
        forearm_records = self.records_with_cloud_dtype(forearm_records)
        full_records = self.records_with_cloud_dtype(np.concatenate([upper_records, forearm_records]))
        self.upper_arm_pub.publish(pc2.create_cloud(header, msg.fields, upper_records, point_step=msg.point_step))
        self.forearm_pub.publish(pc2.create_cloud(header, msg.fields, forearm_records, point_step=msg.point_step))
        self.full_arm_pub.publish(pc2.create_cloud(header, msg.fields, full_records, point_step=msg.point_step))
        if self.log_segment_counts:
            self.log_segment_counts_throttled(upper_records.shape[0], forearm_records.shape[0])

    def keypoint_points_from_cloud(self):
        keypoint_points = {}
        for name, keypoint in zip(self.keypoint_names, self.keypoints):
            if float(keypoint[2]) < self.min_confidence:
                self.warn_throttled(
                    'last_missing_keypoints_warn_ns',
                    f'{name} confidence is below {self.min_confidence:.2f}; skipping segment cloud update.',
                )
                return None

            point = self.mean_point_near_keypoint(keypoint)
            if point is None:
                self.warn_throttled(
                    'last_missing_keypoints_warn_ns',
                    f'No valid depth samples near {name}; skipping segment cloud update.',
                )
                return None
            keypoint_points[name] = point + self.keypoint_offsets.get(name, np.zeros(3, dtype=float))
        return keypoint_points

    def mean_point_near_keypoint(self, keypoint):
        samples = self.samples_from_keypoint_uv(keypoint)
        if samples.shape[0] == 0:
            return None
        return np.mean(samples, axis=0)

    def samples_from_keypoint_uv(self, keypoint):
        u, v = self.keypoint_to_cloud_uv(keypoint)
        uvs = []
        for du in range(-self.joint_sample_radius_px, self.joint_sample_radius_px + 1):
            for dv in range(-self.joint_sample_radius_px, self.joint_sample_radius_px + 1):
                sample_u = u + du
                sample_v = v + dv
                if 0 <= sample_u < self.cloud_width and 0 <= sample_v < self.cloud_height:
                    uvs.append(sample_u + sample_v * self.cloud_width)

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


    def clip_segment_records_in_shoulder_frame(self, start_point, end_point, shoulder_transform):
        records = self.segment_candidate_records_3d(start_point, end_point)
        if records.shape[0] == 0:
            self.warn_throttled(
                'last_empty_segment_warn_ns',
                'No valid point-cloud records near the detected 3D arm segment.',
            )
            return self.empty_cloud_records()

        if self.max_points_per_segment > 0 and records.shape[0] > self.max_points_per_segment:
            stride = int(np.ceil(records.shape[0] / self.max_points_per_segment))
            records = records[::stride]

        transformed_records = records.copy()
        xyz_camera = self.xyz_from_records(transformed_records)
        translation_camera_shoulder, rotation_camera_shoulder = shoulder_transform
        xyz_shoulder = (rotation_camera_shoulder.T @ (xyz_camera - translation_camera_shoulder).T).T
        transformed_records['x'] = xyz_shoulder[:, 0]
        transformed_records['y'] = xyz_shoulder[:, 1]
        transformed_records['z'] = xyz_shoulder[:, 2]
        return transformed_records

    def segment_candidate_records_3d(self, start_point, end_point):
        records = self.valid_full_cloud_records()
        if records.shape[0] == 0:
            return self.empty_cloud_records()

        segment = end_point - start_point
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-6:
            return self.empty_cloud_records()

        xyz = self.xyz_from_records(records)
        relative_points = xyz - start_point
        projection = (relative_points @ segment) / (segment_length * segment_length)
        min_projection = -self.cluster_endpoint_margin_m / segment_length
        max_projection = 1.0 + self.cluster_endpoint_margin_m / segment_length
        projection_mask = (projection >= min_projection) & (projection <= max_projection)
        if not np.any(projection_mask):
            return self.empty_cloud_records()

        candidate_records = records[projection_mask]
        candidate_xyz = xyz[projection_mask]
        projection = np.clip(projection[projection_mask], 0.0, 1.0)
        closest_points = start_point + projection[:, None] * segment
        distances = np.linalg.norm(candidate_xyz - closest_points, axis=1)
        distance_mask = distances <= self.cluster_max_distance_to_segment_m
        return candidate_records[distance_mask]

    def valid_full_cloud_records(self):
        if self.full_cloud_records is not None:
            return self.full_cloud_records

        records = pc2.read_points(
            self.point_cloud,
            field_names=None,
            skip_nans=True,
        )
        stride = max(1, int(self.full_cloud_stride))
        if stride > 1:
            records = records[::stride]

        if records.shape[0] == 0:
            self.full_cloud_records = records
            return self.full_cloud_records

        xyz = self.xyz_from_records(records)
        valid_mask = np.isfinite(xyz).all(axis=1) & ~np.all(xyz == 0.0, axis=1)
        self.full_cloud_records = records[valid_mask]
        return self.full_cloud_records

    def empty_cloud_records(self):
        dtype = pc2.dtype_from_fields(self.point_cloud.fields, point_step=self.point_cloud.point_step)
        return np.empty((0,), dtype=dtype)

    def records_with_cloud_dtype(self, records):
        dtype = pc2.dtype_from_fields(self.point_cloud.fields, point_step=self.point_cloud.point_step)
        if records.dtype == dtype:
            return records
        normalized_records = np.empty(records.shape, dtype=dtype)
        for name in normalized_records.dtype.names:
            normalized_records[name] = records[name]
        return normalized_records

    @staticmethod
    def xyz_from_records(records):
        return np.column_stack(
            [
                np.asarray(records['x'], dtype=float),
                np.asarray(records['y'], dtype=float),
                np.asarray(records['z'], dtype=float),
            ]
        )

    def keypoint_to_cloud_uv(self, keypoint):
        return tuple(np.rint(self.keypoint_uv_to_cloud_uv(np.array([keypoint[0], keypoint[1]], dtype=float))).astype(int))

    def keypoint_uv_to_cloud_uv(self, keypoint_uv):
        keypoint_width = max(1, int(self.keypoint_image_width))
        keypoint_height = max(1, int(self.keypoint_image_height))
        scale = np.array(
            [
                max(1, self.cloud_width) / keypoint_width,
                max(1, self.cloud_height) / keypoint_height,
            ],
            dtype=float,
        )
        cloud_uv = keypoint_uv * scale
        cloud_uv[0] = np.clip(cloud_uv[0], 0, max(0, self.cloud_width - 1))
        cloud_uv[1] = np.clip(cloud_uv[1], 0, max(0, self.cloud_height - 1))
        return cloud_uv

    def publish_shoulder_transform(self, right_shoulder_camera, left_shoulder_camera, stamp):
        """Publish the camera -> right-shoulder TF used by keypoint_distribution_extraction.py.

        The right-shoulder frame is anchored at the detected right shoulder. Its
        orientation follows the same empirical camera-to-shoulder convention as
        keypoint_distribution_extraction.py, with only a yaw correction from the
        detected shoulder line. Keeping this function equivalent avoids creating
        a second shoulder-frame definition for segmented point clouds.
        """
        shoulder_line = left_shoulder_camera - right_shoulder_camera
        shoulder_line_unit = self.safe_normalize(shoulder_line)
        if shoulder_line_unit is None:
            self.warn_throttled(
                'last_short_shoulder_warn_ns',
                'Shoulder line is too short; not updating shoulder TF.',
            )
            return None

        shoulder_line_xz = self.safe_normalize(np.array([shoulder_line_unit[0], 0.0, shoulder_line_unit[2]]))
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
        avg_quat = R.from_euler('xyz', [3.14, 1.57 + avg_yaw_correction, 0.0]).as_quat()
        rotation_camera_shoulder = R.from_quat(avg_quat).as_matrix()

        transforms = [
            self.make_tf(self.camera_frame, child, avg_translation, avg_quat, stamp)
            for child in self.shoulder_tf_children
        ]
        if self.shoulder_frame not in self.shoulder_tf_children:
            transforms.append(self.make_tf(self.camera_frame, self.shoulder_frame, avg_translation, avg_quat, stamp))
        self.tf_broadcaster.sendTransform(transforms)
        return avg_translation, rotation_camera_shoulder

    def make_tf(self, parent, child, xyz, quat_xyzw, stamp):
        transform = TransformStamped()
        transform.header.stamp = stamp
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

    def should_process_cloud(self, stamp):
        if self.max_processing_rate_hz <= 0.0:
            return True

        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
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

    def vector3_parameter(self, name, fallback):
        values = list(self.get_parameter(name).value)
        if len(values) != 3:
            self.get_logger().warn(
                f'Parameter {name} must contain exactly three values; using fallback {fallback.tolist()}.'
            )
            return np.array(fallback, dtype=float)
        return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)

    def warn_throttled(self, attr_name, message, period_sec=2.0):
        now_ns = self.get_clock().now().nanoseconds
        last_ns = getattr(self, attr_name)
        if last_ns < 0 or (now_ns - last_ns) * 1e-9 >= period_sec:
            self.get_logger().warn(message)
            setattr(self, attr_name, now_ns)

    def log_segment_counts_throttled(self, upper_count, forearm_count, period_sec=1.0):
        now_ns = self.get_clock().now().nanoseconds
        if self.last_segment_count_log_ns < 0 or (now_ns - self.last_segment_count_log_ns) * 1e-9 >= period_sec:
            self.get_logger().info(
                f'Published shoulder-frame segment clouds: upper_arm={upper_count}, forearm={forearm_count}'
            )
            self.last_segment_count_log_ns = now_ns

    @staticmethod
    def is_valid_point(point):
        return (
            np.isfinite(point[0])
            and np.isfinite(point[1])
            and np.isfinite(point[2])
            and not (point[0] == 0.0 and point[1] == 0.0 and point[2] == 0.0)
        )

    @staticmethod
    def safe_normalize(vector, eps=1e-9):
        norm = np.linalg.norm(vector)
        if norm < eps:
            return None
        return vector / norm


def main(args=None):
    rclpy.init(args=args)
    node = ArmSegmentPointCloudExtractionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
