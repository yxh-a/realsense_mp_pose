import rclpy
from rclpy.node import Node
import rclpy.wait_for_message
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import sensor_msgs_py.point_cloud2 as pc2
import sys
import numpy as np
from geometry_msgs.msg import Point



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
        self.right_arm_3D = [] # list of points in 3D world coordinate
        
        # setup rviz marker list publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/pose_marker', 10)
        self.marker_array = MarkerArray()
        
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
       


    def publish_image(self):
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(self.cv_image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        self.annotated_image = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        self.image_pub.publish(self.annotated_image)

    def get_3d_points(self,this_uvs):

        if self.point_cloud is None:
            return None
        # get the pixel coordinate of the right arm
        # print(self.point_cloud)

        cloud_array = pc2.read_points_list(self.point_cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=this_uvs)
        points3d = []
        for i in range(len(cloud_array)):\
            # check if the point is not (0,0,0)
            cloud_point = cloud_array[i]
            if cloud_point != (0.0, 0.0, 0.0):
                this_uv = this_uvs[i]
                # substitue with increments of one pixel until  it is not (0,0,0)
                while cloud_point == (0.0, 0.0, 0.0):
                    this_uv = (this_uv[0]+1, this_uv[1])
                    cloud_point = pc2.read_points_list(self.point_cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=this_uv)[0]
                points3d.append(cloud_point)
        return points3d

    def publish_markers(self):
           

            uvs = [] # iterable
            width = self.point_cloud.width

            # get the pixel coordinate of the right arm
            for i in self.right_arm_index:
                u = int(self.results.pose_landmarks.landmark[i].x * self.image_width)
                v = int(self.results.pose_landmarks.landmark[i].y * self.image_height)
                uvs.append(u+v*width)

            # get the 3D world coordinate of the right arm
            # uvs_ = [(u,v) for u,v in uvs]
            self.right_arm_3D = self.get_3d_points(uvs)
            
            num_points = len(self.right_arm_3D)
            # self.get_logger().info('Number of points in right arm: {}'.format(num_points))
            #  check if all points are floats
            # print(self.right_arm_3D)
            # create marker for each point in the right arm
            if num_points < 4:
                self.get_logger().info('Not enough points in right arm')
            else:
                  # create marker array
                self.marker_array = MarkerArray()
                marker_id = 0
                for point in self.right_arm_3D:
                    # print("point ", marker_id, " : ", point)
                    marker = Marker()
                    marker.header.frame_id = "camera_depth_optical_frame"
                    marker.header.stamp = rclpy.clock.Clock().now().to_msg()
                    marker.id = marker_id
                    marker_id += 1
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    print(float(point.x), float(point.y), float(point.z))
                    marker.pose.position.x = float(point.x)
                    marker.pose.position.y = float(point.y)
                    marker.pose.position.z = float(point.z)
                    marker.scale.x = 0.05
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05
                    marker.color.a = 1.0
                    marker.color.r = 1.0

                    # add names
                    marker.ns = self.right_arm_names[marker.id]
                    marker.text = self.right_arm_names[marker.id]

                    self.marker_array.markers.append(marker)

                # Draw line strip connecting all right arm points
                line_marker = Marker()
                line_marker.header.frame_id = "camera_frame"
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

                # Add points to the line
                for pt in self.right_arm_3D:
                    p = Point()
                    p.x = float(pt.x)
                    p.y = float(pt.y)
                    p.z = float(pt.z)
                    line_marker.points.append(p)

                self.marker_array.markers.append(line_marker)

            
            self.marker_pub.publish(self.marker_array)


    def point_cloud_callback(self, msg):
        self.point_cloud_hz = 1.0 / (self.get_clock().now() - self.point_cloud_last_time).nanoseconds*1e9
        self.point_cloud_last_time = self.get_clock().now()
        self.get_logger().info('Point Cloud Hz: {}'.format(self.point_cloud_hz))
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
        
        
        # self.process_point_cloud()

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
