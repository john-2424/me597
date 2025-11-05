import os
import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory

from task_5.utils.video import Video

class ImagePublisher(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('image_publisher')
        
        # Topic publisher, with 'Image' message type, 20 queue size
        self.publisher_ = self.create_publisher(Image, 'video_data', 20)
        
        # Topic timer to publish messages in an interval
        timer_period = 0.05  # in seconds, for 20Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Bridge to convert image type between CV and ROS
        self.bridge = CvBridge()

        # Reading a video using open cv
        pkg_share = get_package_share_directory('my_package')
        video_file_path = os.path.join(pkg_share, 'resource', 'lab4_video.avi')
        self.vid = Video(video_file_path)

        # Parameter to show video in this node
        self.show_video = True

    def timer_callback(self):
        try:
            # Get next frame to publish
            frame = next(self.vid)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Show or display frames on a window
            if self.show_video:
                cv.imshow('Frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    self.show_video = False
                    cv.destroyAllWindows()
            
            # Create and assign 'Image' message object by converting CV image to ROS image
            msg = self.bridge.cv2_to_imgmsg(frame)
            
            # Publish message to the topic
            self.publisher_.publish(msg)
        except StopIteration:
            self.get_logger.info('End of video stream.')
    
    def __del__(self):
        self.vid.release()


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'ImagePublisher' class node
    image_publisher = ImagePublisher()

    # Spin up the created node
    rclpy.spin(image_publisher)

    # Explicit declaration to destroy the node object and shutdown rclpy
    image_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
