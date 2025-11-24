import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Pose2D, BoundingBox2D

from task_5.utils.detect import find_object_hsv, find_object_hsv_triangle


class ObjectDetector(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('object_detector')

        # Subscribe to a topic
        self.subscription = self.create_subscription(
            Image,
            'video_data',
            self.listener_callback,
            20
        )
        self.subscription  # prevent unused variable warning

        # Bridge to convert image type between CV and ROS
        self.bridge = CvBridge()

        # Parameter to show video in this node
        self.show_video = True

        # Mode of detection
        self.mode = 'hsv_triangle'  # hsv_triangle or hsv

        # Publisher on /bbox
        self.bbox_pub = self.create_publisher(BoundingBox2D, '/bbox', 10)

    def _detect(self, frame):
        if self.mode == 'hsv_triangle':
            return find_object_hsv_triangle(frame)
        return find_object_hsv(frame)
    
    def listener_callback(self, msg):
        # Logs the received messages from a topic
        # self.get_logger().info(f'Message Type: {type(msg)}')

        # Convert ros to cv image type
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        result = self._detect(frame)
        if result is not None:
            cx, cy, w, h, (x, y, bw, bh) = result

            # Logging pixels from top-left origin
            self.get_logger().info(f'centroid=({cx:.1f},{cy:.1f}) size=({w:.0f},{h:.0f})')

            # Publish BoundingBox2D
            bb = BoundingBox2D()
            bb.center.position.x = float(cx)
            bb.center.position.y = float(cy)
            bb.center.theta = 0.0
            bb.size_x = float(w)
            bb.size_y = float(h)
            self.bbox_pub.publish(bb)

            # Draw bbox + centroid
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

        # Show or display frames on a window
        if self.show_video:
            cv.imshow('Object Detector: Frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.show_video = False
                cv.destroyAllWindows()


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'ObjectDetector' class node
    object_detector = ObjectDetector()

    # Spin up the created node
    rclpy.spin(object_detector)

    # Explicit declaration to destroy the node object and shutdown rclpy
    object_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
