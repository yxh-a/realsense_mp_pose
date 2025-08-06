import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import sensor_msgs_py.point_cloud2 as pc2
import sys
import numpy as np
from geometry_msgs.msg import PoseArray, Point, Pose



class PoseDetectorNode(Node):
    def __init__(self):
        super().__init__('pose_detector')

        # check if venv is aused
        print(f"Using Python: {sys.executable}")

        # read args from launch
        self.declare_parameter('publish_annotated', True)
        self.publish_annotated = self.get_parameter('publish_annotated').value

        self.bridge = CvBridge()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # setup 2d anotated image publisher
        self.image_pub = self.create_publisher(Image, '/pose_annotated', 10)
        self.rgb_image = Image()
        self.annotated_image = Image()
        self.cv_image = None
        self.results = None

        # setup joint pixel variable, as well as joint 3D world coordinate variable in the right arm
        self.right_arm_index = [16,14,12,11] # right wrist, right elbow, right shoulder, left shoulder
        self.right_arm_names = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder']
        empty_point = Point()
        empty_point.x = 0.0
        empty_point.y = 0.0
        empty_point.z = 0.0
        self.right_arm_3D = [empty_point, empty_point, empty_point, empty_point]  # right arm 3D world coordinate
        self.point_initialized = [False, False, False, False]  # flag to check if the point is initialized

        
        # setup rviz marker list publisher
        self.joint_position_pub = self.create_publisher(PoseArray, '/joint_positions', 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/pose_marker', 10)
        
        self.image_width = 640
        self.image_height = 480

        self.point_cloud = PointCloud2()


        

        # set up subscirbers
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        
        # point cloud topic
        self.point_cloud_topic = '/camera/camera/depth/color/points'
        # point cloud subscriber
        self.point_cloud_sub = self.create_subscription(
            PointCloud2, self.point_cloud_topic, self.point_cloud_callback, 10)

        # set up a time recorder to check frequencies on the image and point cloud
        self.image_last_time = self.get_clock().now()
        self.point_cloud_last_time = self.get_clock().now()
        self.image_hz = 0
        self.point_cloud_hz = 0
        self.get_logger().info('Pose Detector Node Started')

        # setup median filter for point detection
        self.median_filter_size = 5  # Size of the median filter
        self.shoulder_median_filter = np.zeros((self.median_filter_size, 3))  # Initialize median filter for shoulder
        self.elbow_median_filter = np.zeros((self.median_filter_size, 3))  # Initialize median filter for elbow
        self.wrist_median_filter = np.zeros((self.median_filter_size, 3))  # Initialize median filter for wrist
        self.left_shoulder_median_filter = np.zeros((self.median_filter_size, 3))  # Initialize median filter for left shoulder

        self.shoulder_point = Point()
        self.elbow_point = Point()
        self.wrist_point = Point()
        self.left_shoulder_point = Point()


    def publish_image(self):
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(self.cv_image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        self.annotated_image = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        self.image_pub.publish(self.annotated_image)

    def validate_point(self, index,cloud_point):
        if index < 0 or index > 3:
            self.get_logger().error('Index out of range for right arm points')
            return cloud_point
        
        old_point = self.right_arm_3D[index]

        if cloud_point is None:
            # self.get_logger().warn(f'Point at index {index} is None, retaining old point')
            return old_point

        if cloud_point.x == 0.0 and cloud_point.y == 0.0 and cloud_point.z == 0.0:
            # self.get_logger().warn(f'Point at index {index} is (0,0,0), retaining old point')
            return old_point
        
        # check if the point is initialized
        # if not self.point_initialized[index]:
        #     if float(cloud_point.x) != 0.0 and float(cloud_point.y) != 0.0 and float(cloud_point.z) != 0.0:
        #         # if the point is not initialized, set it to the new point
        #         self.get_logger().info(f'Point at index {index} is not initialized, setting to new point')
        #         self.point_initialized[index] = True
        #         return cloud_point
        
        # # check if the new point is too far from the old point
        # distance = np.sqrt((cloud_point.x - old_point.x) ** 2 + (cloud_point.y - old_point.y) ** 2 + (cloud_point.z - old_point.z) ** 2)
        # if distance > 0.5:  # threshold of 0.5 meters
        #     # self.get_logger().warn(f'Point at index {index} is too far from old point, retaining old point')
        #     return old_point
        
        # if the point is valid, return the new point
        # self.get_logger().info(f'Point at index {index} is valid, updating point [{cloud_point.x}, {cloud_point.y}, {cloud_point.z}]')
        return cloud_point

    def point_from_uv(self, u, v):
        # this function returns the point from the point cloud given the pixel coordinate (u, v)
        if self.point_cloud is None:
            self.get_logger().error('Point Cloud is not initialized')
            return None
        
        width = self.point_cloud.width

        # generate a patch of pixels around the pixel coordinate (u, v) to get a list of points (+-1 pixels)
        this_uvs = []
        for i in range(-2, 3):  # from -2 to 2 inclusive
            for j in range(-2, 3):  # from -2 to 2 inclusive
                new_u = u + i
                new_v = v + j
                if 0 <= new_u < self.image_width and 0 <= new_v < self.image_height:
                    new_uv = new_u + new_v * width
                    this_uvs.append(new_uv)
        
        # read the list of points from the point cloud, get rid of anomoly, and return median of 
        cloud_array = pc2.read_points_list(self.point_cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=this_uvs)
        
        # self.get_logger().info(f'Found {len(cloud_array)} points at pixel coordinate (u, v): ({u}, {v})')
        # get rid of points that are (0,0,0)
        cloud_array = [point for point in cloud_array if not (point.x == 0.0 and point.y == 0.0 and point.z == 0.0)]
        # self.get_logger().info(f'After filtering, get {len(cloud_array)} valid points')
        # if the point is not found, return None
        if len(cloud_array) < 1:
            # self.get_logger().warn(f'No point found at pixel coordinate (u, v): ({u}, {v})')
            return None
        
        #return median of the points
        cloud_point = cloud_array[0]
        # add a small offset to the z coordinate to compensate for surface point
        true_point = Point()
        true_point.x = cloud_point[0]
        true_point.y = cloud_point[1]
        true_point.z = cloud_point[2]+ 0.05  # add a small offset to the z coordinate to compensate for surface point
        # self.get_logger().info(f'Found {len(cloud_array)} points at pixel coordinate (u, v): ({u}, {v})')

        return true_point

    def publish_markers(self):
    # this function publish the markers for the right arm, as well as their positions to /joint_positions
        uvs = [] # iterable
        width = self.point_cloud.width

        # get the pixel coordinate of the right arm
    
        for i in range(len(self.right_arm_index)):

            index = self.right_arm_index[i]
            u = int(self.results.pose_landmarks.landmark[index].x * self.image_width)
            v = int(self.results.pose_landmarks.landmark[index].y * self.image_height)
            
            cloud_point = self.point_from_uv(u, v)
            validated_point = self.validate_point(i, cloud_point)

            self.right_arm_3D[i] = validated_point

        # create marker array
        self.paint_markers()


    def point_cloud_callback(self, msg):
        self.point_cloud_hz = 1.0 / (self.get_clock().now() - self.point_cloud_last_time).nanoseconds*1e9
        self.point_cloud_last_time = self.get_clock().now()
        # self.get_logger().info('Point Cloud Hz: {}'.format(self.point_cloud_hz))
        self.point_cloud = msg
        # # Publish 3D World Coordinate
        try:
            self.publish_markers()
        except Exception as e:
            self.get_logger().error('Error Processing Point Cloud: {}'.format(e))
    
    def process_point_cloud(self):
        # this function will be called in the image callback, it grabs the latest point cloud in memory and get pose data
        # self.get_logger().info('Processing Point Cloud')
        # get the latest point cloud on self.point_cloud_topic
        self.point_cloud = rclpy.wait_for_message(self.point_cloud_topic, PointCloud2, timeout_sec=0.1)
        if self.point_cloud is None:
            self.get_logger().error('No Point Cloud Received')
            return
        try:
            self.publish_markers()
        except Exception as e:
            self.get_logger().error('Error Processing Point Cloud: {}'.format(e))

    def paint_markers(self):
        marker_array = MarkerArray()
        joint_positions = PoseArray()
        joint_positions.header.frame_id = "camera_depth_optical_frame"
        # joint_positions.header.frame_id = "lbr_link_0"
        joint_positions.header.stamp = self.get_clock().now().to_msg()
        marker_id = 0

        if len(self.right_arm_3D) < 4:
            self.get_logger().warn('Right arm 3D points are not initialized, skipping marker publishing')
            return
        
        for point in self.right_arm_3D:
            # print("point ", marker_id, " : ", point)
            marker = Marker()
            joint_position = Pose()
            marker.header.frame_id = "camera_depth_optical_frame"
            # marker.header.frame_id = "lbr_link_0"
            marker.header.stamp = rclpy.clock.Clock().now().to_msg()
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point.x)
            joint_position.position.x = float(point.x)
            marker.pose.position.y = float(point.y)
            joint_position.position.y = float(point.y)
            marker.pose.position.z = float(point.z)
            joint_position.position.z = float(point.z)
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0

            # Text marker (caption)
            text_marker = Marker()
            text_marker.header.frame_id = "camera_depth_optical_frame"
            text_marker.header.stamp = rclpy.clock.Clock().now().to_msg()
            text_marker.id = marker_id + 1000  # offset to avoid ID collision
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


            # Draw line strip connecting all right arm points
            line_marker = Marker()
            line_marker.header.frame_id = "camera_depth_optical_frame"
            # line_marker.header.frame_id = "lbr_link_0"
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.id = marker_id
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.02  # thickness of the line
            line_marker.color.a = 1.0
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.ns = "right_arm_line"
            line_marker.points.append(self.wrist_point)
            line_marker.points.append(self.elbow_point)
            line_marker.points.append(self.shoulder_point)

            marker_array.markers.append(line_marker)

            self.marker_pub.publish(marker_array)
            self.joint_position_pub.publish(joint_positions)

    def image_callback(self, msg):
        self.image_hz = 1.0 / (self.get_clock().now() - self.image_last_time).nanoseconds*1e9
        self.image_last_time = self.get_clock().now()
        # self.get_logger().info('Image Hz: {}'.format(self.image_hz))
        try:
            # get the image from the message
            # self.get_logger().info('Received Image')

            # Convert ROS Image to OpenCV format
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # process the image with mediapipe
            self.rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            
            # Process Image with MediaPipe Pose Detection
            self.results = self.pose.process(self.rgb_image)
            
            # Publish Annotated Image
            if self.publish_annotated:
                self.publish_image()

            
            
        except Exception as e:
            self.get_logger().error('Error Processing Image: {}'.format(e))
        
        


def main(args=None):

    print(sys.executable)
    print(sys.path)

    rclpy.init(args=args)
    pose_detector_node = PoseDetectorNode()

    pose_detector_node.get_logger().info('Pose Detector Node is running...')
    rclpy.spin(pose_detector_node)

    # Clean up
    pose_detector_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
