import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


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

    def listener_callback(self, msg):
        # Logs the received messages from a topic
        self.get_logger().info(f'Message Type: {type(msg)}')

        # Convert ros to cv image type
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Show or display frames on a window
        if self.show_video:
            cv.imshow('Frame', frame)
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
