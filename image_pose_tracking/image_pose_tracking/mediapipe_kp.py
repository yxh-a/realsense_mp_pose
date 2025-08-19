import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int16MultiArray
from cv_bridge import CvBridge
import mediapipe as mp

class MediaPipePoseNode(Node):
    def __init__(self):
        super().__init__('mediapipe_pose_node')

        # Params
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('model_complexity', 1)  # 0,1,2
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('enable_gpu', True)  # if MP build supports it
        self.declare_parameter('publish_annotated', True)

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        model_complexity = int(self.get_parameter('model_complexity').value)
        min_det = float(self.get_parameter('min_detection_confidence').value)
        min_trk = float(self.get_parameter('min_tracking_confidence').value)
        enable_gpu = bool(self.get_parameter('enable_gpu').value)
        self.publish_annotated = self.get_parameter('publish_annotated').value

        # setup 2d anotated image publisher
       
        self.mp_drawing = mp.solutions.drawing_utils

        self.bridge = CvBridge()

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        # Note: MediaPipe uses BGR input sizes up to 2560x1440 comfortably; resize upstream if needed
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )

        # Subscriber
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.on_image, 10)

        # Publisher
        self.kp_pub = self.create_publisher(Int16MultiArray, '/keypoints_2d', 10)
        self.image_pub = self.create_publisher(Image, '/pose_annotated', 10)
        
        self.right_arm_index = [16,14,12,11] # right wrist, right elbow, right shoulder, left shoulder
        self.right_arm_names = ['right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder']

        # Preallocate structure to avoid per‑frame allocations, pose is always 4 landmarks
        self._poses_msg = Int16MultiArray()
        self._poses_msg.data = [0] * (len(self.right_arm_index) * 3)  # 3 values per landmark: x, y, confidence

        self.get_logger().info(f"MediaPipe Pose running on {rgb_topic} (model_complexity={model_complexity})")

    def on_image(self, msg: Image):
        # Convert to OpenCV BGR image. Avoid copying where possible.
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        if not res.pose_landmarks:
            self.get_logger().warn('No pose landmarks detected')
            return
        
        h, w = img.shape[:2]
        landmarks = res.pose_landmarks.landmark  # full list of landmarks
        pa = self._poses_msg
        # Fill 2D keypoints (pixels) + confidence
        # Orientation fields unused except .w to carry confidence cheaply
        i = 0
        for index in self.right_arm_index:
            
            lm = landmarks[index]
            # Clamp to image bounds to avoid downstream array indexing issues
            u = float(np.clip(lm.x * w, 0, w - 1))
            v = float(np.clip(lm.y * h, 0, h - 1))
            c = float(np.clip(lm.visibility if lm.visibility > 0 else lm.visibility, 0.0, 1.0))  # MP "visibility" in [0,1]
            # self.get_logger().debug(f"Keypoint {self.right_arm_names[i]}: ({u}, {v}), confidence: {c}")
            pa.data[i * 3] = int(u)  # x
            pa.data[i * 3 + 1] = int(v)  # y
            pa.data[i * 3 + 2] = int(c * 100)
            i += 1

        self.kp_pub.publish(pa)

        if self.publish_annotated:
            # Draw landmarks on the image
            if res.pose_landmarks:
                annotated_img = img_rgb.copy()
                self.mp_drawing.draw_landmarks(annotated_img, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # Convert back to ROS Image message
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
                annotated_msg.header = msg.header
                annotated_msg.header.frame_id = self.camera_frame
                self.image_pub.publish(annotated_msg)


def main():
    rclpy.init(args=None)
    node = MediaPipePoseNode()
    node.get_logger().info('MediaPipe Pose Node is running...')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
