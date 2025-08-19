from turtle import width
import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
import sensor_msgs_py.point_cloud2 as pc2
import sys
import csv
import numpy as np
from geometry_msgs.msg import PoseArray, Point, Pose
from std_msgs.msg import Int16MultiArray


class PointCloudExtractionNode(Node):
    def __init__(self):
        super().__init__('point_cloud_extraction_node')
        
        # Declare parameters
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('joint_center_depth_offset', 0.05)  # Offset for depth adjustment
        self.declare_parameter('marker_size', 0.05)  # Size of the marker for visualization
        self.declare_parameter('filter_window_size', 5)  # Size of the median filter window
        self.declare_parameter('save_joint_positions', False)  # Whether to save joint positions to a csv file

        # Get parameters
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.joint_center_depth_offset = self.get_parameter('joint_center_depth_offset').get_parameter_value().double_value
        self.marker_size = self.get_parameter('marker_size').get_parameter_value().double_value
        self.save_joint_positions = self.get_parameter('save_joint_positions').get_parameter_value().bool_value

        # setup joint pixel variable, as well as joint 3D world coordinate variable in the right arm
        self.right_arm_names = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder']
        empty_point = Point()
        empty_point.x = 0.0
        empty_point.y = 0.0
        empty_point.z = 0.0
        self.right_arm_3D = [empty_point, empty_point, empty_point, empty_point]  # right arm 3D world coordinate
        
        # set up subscribers
        self.keypoints_topic = '/keypoints_2d'
        self.kp_sub = self.create_subscription(Int16MultiArray, self.keypoints_topic, self.keypoints_callback, 10)
        self.keypoints = [None, None, None, None]  # right wrist, right elbow, right shoulder, left shoulder
        # point cloud topic
        self.point_cloud_topic = '/camera/camera/depth/color/points'
        # point cloud subscriber
        self.point_cloud_sub = self.create_subscription(PointCloud2, self.point_cloud_topic, self.point_cloud_callback, 10)
        self.get_logger().info('Pose Detector Node Started')

        # setup median filter for point detection
        self.median_filter_size = self.get_parameter('filter_window_size').get_parameter_value().integer_value  # Size of the median filter

        self.shoulder_point = Point()
        self.elbow_point = Point()
        self.wrist_point = Point()
        self.left_shoulder_point = Point()
        
        # setup publishers
        self.joint_position_pub = self.create_publisher(PoseArray, '/joint_positions', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pose_marker', 10)
        
        self.image_width = 640
        self.image_height = 480

        self.point_cloud = PointCloud2()

        if self.save_joint_positions:
            # Create a CSV file to save joint positions
            self.csv_file = open('joint_positions.csv', 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'wrist_x', 'wrist_y', 'wrist_z',
                                      'elbow_x', 'elbow_y', 'elbow_z',
                                      'shoulder_x', 'shoulder_y', 'shoulder_z',
                                      'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'])
    

    def keypoints_callback(self, msg):
        for i in range(4):
            kp = np.array([
                msg.data[i * 3],  # x
                msg.data[i * 3 + 1],  # y
                msg.data[i * 3 + 2] / 100.0  # confidence, convert from percentage
            ])
            self.keypoints[i] = kp
            # self.get_logger().info(f'Keypoint {self.right_arm_names[i]}: {kp}')

    def point_cloud_callback(self, msg):
        # Store the point cloud data
        self.point_cloud = msg
    
        # Process the point cloud to extract 3D coordinates
        self.process_point_cloud()
        # Publish the joint data
        self.publish_joint_data()


    def point_from_uv(self, u, v):
        this_uvs = []
        # Create a cluster of pixels around the keypoint to account for noise
        for i in range(-2, 2):  # from -2 to 2 inclusive
            for j in range(-2, 2):  # from -2 to 2 inclusive
                new_u = u + i
                new_v = v + j
                if 0 <= new_u < self.image_width and 0 <= new_v < self.image_height:
                    new_uv = new_u + new_v * self.image_width
                    this_uvs.append(new_uv)
    
        # Extract the point cloud data for the corresponding pixel cluster
        cloud_array = pc2.read_points(self.point_cloud, field_names=('x', 'y', 'z'), uvs=this_uvs, skip_nans=True)
        # get rid of 000 points and nans points
        cloud_array = [point for point in cloud_array if not (point[0] == 0.0 and point[1] == 0.0 and point[2] == 0.0) and not (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]))]
        if len(cloud_array) < 1:
            return None
        
        cloud_point = cloud_array[0]  # Take the first valid point
        true_point = Point()
        true_point.x = cloud_point[0]
        true_point.y = cloud_point[1]
        true_point.z = cloud_point[2] + self.joint_center_depth_offset  # Adjust depth with offset
        return true_point

    def process_point_cloud(self):
        # this function use keyppints and point cloud to extract 3D world coordinates of the joints
        if not self.point_cloud.data:
            self.get_logger().warn('No point cloud data available')
            return
        if not self.keypoints or any(kp is None for kp in self.keypoints):
            self.get_logger().warn('Keypoints are not available or incomplete')
            return
        for i in range(4):
            kp = self.keypoints[i]
            # self.get_logger().info(f'Processing keypoint {self.right_arm_names[i]}: {kp}')
            if kp[2] < 0.5:
                self.get_logger().warn('Keypoint confidence is too low, skipping this keypoint and keep the previous value')
                continue
            # Convert keypoint pixel coordinates to 3D point cloud coordinates
            point_joint = self.point_from_uv(kp[0], kp[1])
            if point_joint is None:
                self.get_logger().warn('Failed to convert keypoint to 3D point at '+ self.right_arm_names[i])
                continue
            
            self.right_arm_3D[i] = point_joint
        
    def publish_joint_data(self):
        # this function publish the joint data as PoseArray, as well as visualization markers for rviz
        marker_array = MarkerArray()
        joint_positions = PoseArray()
        joint_positions.header.frame_id = "camera_depth_optical_frame"
        # joint_positions.header.frame_id = "lbr_link_0"
        joint_positions.header.stamp = self.get_clock().now().to_msg()
        for i in range(4):
            point = self.right_arm_3D[i]
            marker = Marker()
            joint_position = Pose()
            marker.header.frame_id = self.camera_frame
            # marker.header.frame_id = "lbr_link_0"
            marker.header.stamp = rclpy.clock.Clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point.x)
            joint_position.position.x = float(point.x)
            marker.pose.position.y = float(point.y)
            joint_position.position.y = float(point.y)
            marker.pose.position.z = float(point.z)
            joint_position.position.z = float(point.z)
            marker.scale.x = self.marker_size
            marker.scale.y = self.marker_size
            marker.scale.z = self.marker_size
            marker.color.a = 1.0
            marker.color.r = 1.0

            # Text marker (caption)
            text_marker = Marker()
            text_marker.header.frame_id = self.camera_frame
            text_marker.header.stamp = rclpy.clock.Clock().now().to_msg()
            text_marker.id = i + 4  # Ensure unique ID for text marker
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(point.x) + 0.05
            text_marker.pose.position.y = float(point.y)
            text_marker.pose.position.z = float(point.z) # put text slightly above sphere

            text_marker.scale.z = 0.04  # height of text
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0

            text_marker.text = self.right_arm_names[marker.id]
            text_marker.ns = "right_arm_text"

            marker_array.markers.append(marker)
            marker_array.markers.append(text_marker)
            joint_positions.poses.append(joint_position)

        self.marker_pub.publish(marker_array)
        self.joint_position_pub.publish(joint_positions)
        
        if self.save_joint_positions:
            # Save joint positions to CSV file
            timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
            self.csv_writer.writerow([
                timestamp,
                self.right_arm_3D[0].x, self.right_arm_3D[0].y, self.right_arm_3D[0].z,  # wrist
                self.right_arm_3D[1].x, self.right_arm_3D[1].y, self.right_arm_3D[1].z,  # elbow
                self.right_arm_3D[2].x, self.right_arm_3D[2].y, self.right_arm_3D[2].z,  # shoulder
                self.right_arm_3D[3].x, self.right_arm_3D[3].y, self.right_arm_3D[3].z   # left shoulder
            ])
            self.csv_file.flush()

            

def main(args=None):
    rclpy.init(args=None)
    pointcloud_extraction_node = PointCloudExtractionNode()
    pointcloud_extraction_node.get_logger().info('Point Cloud Extraction Node is running...')

    rclpy.spin(pointcloud_extraction_node)

    # Clean up
    pointcloud_extraction_node.destroy_node()
    rclpy.shutdown()
    if pointcloud_extraction_node.save_joint_positions:
        pointcloud_extraction_node.csv_file.close()

if __name__ == '__main__':
    main()
