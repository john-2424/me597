import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from task_6.utils.detect import find_object_hsv, find_object_hsv_triangle
from task_6.utils.pid import PID


class RedBallTracker(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('red_ball_tracker')

        # Subscribe to a topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            20
        )
        self.subscription  # prevent unused variable warning

        # Bridge to convert image type between CV and ROS
        self.bridge = CvBridge()

        # Parameter to show video in this node
        self.show_video = True

        # Mode of detection
        self.mode = 'hsv'  # hsv_triangle or hsv

        # Frame info params
        self.first_frame = True
        self.frame_width = None
        self.frame_height = None

        # Tracking Reference params
        self.speed_reference = None
        self.speed_tol = None
        # self.speed_err_max = None
        # self.speed_err_min = None
        self.heading_reference = None
        self.heading_tol = None
        # self.heading_err_max = None
        # self.heading_err_min = None
        self.alpha = 0.3  # error scale down factor

        # PID parameters
        self.prev_sec = None
        self.speed_max = 0.20
        self.heading_max = 1.0
        self.turn_coeff = 1.0    # 0.20 coeff to stop bot by zeroing bot vel, 1.0 to ignore this logic
        # PIDs for speed and yaw
        self.pid_speed = PID(
            kp=0.20, ki=0.01, kd=0.50, 
            i_limit=0.8, 
            out_limit=(-self.speed_max, self.speed_max)
        )   # output = linear.x (m/s)
        self.pid_heading = PID(
            kp=0.20, ki=0.01, kd=0.50, 
            i_limit=0.8, 
            out_limit=(-self.heading_max, self.heading_max)
        )  # output = angular.z (rad/s)

        # Command Velocity Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def _detect(self, frame):
        if self.mode == 'hsv_triangle':
            return find_object_hsv_triangle(frame)
        return find_object_hsv(frame)
    
    def __scale_value(self, x, min1, max1, min2, max2):
        return min2 + ( (x - min1) / (max1 - min1) ) * (max2 - min2)
    
    def _plan(self, cx, bw, bh):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        dt = now_sec - self.prev_sec if self.prev_sec is not None else 0
        self.prev_sec = now_sec

        speed = 0.0
        heading = 0.0

        # reset integrators to avoid "creep"
        # self.pid_speed.reset()
        # self.pid_heading.reset()
        # return 0.0, 0.0

        speed_curr = 0.5 * bw + 0.5 * bh
        speed_err = self.alpha * (self.speed_reference - speed_curr)
        # speed_err = self.__scale_value(speed_err, self.speed_err_min, self.speed_err_max, -1*self.speed_max, self.speed_max)

        heading_curr = cx
        heading_err = self.alpha * (heading_curr - self.heading_reference)
        # heading_err = self.__scale_value(heading_err, self.heading_err_min, self.heading_err_max, -1*self.heading_max, self.heading_max)

        speed = self.pid_speed.step(speed_err, dt)
        heading = self.pid_heading.step(heading_err, dt)

        if abs(heading) > self.turn_coeff*self.heading_max:
            speed = 0.0
        
        if abs(speed_err)  < self.speed_tol: speed = 0.0
        if abs(heading_err) < self.heading_tol: heading = 0.0

        if abs(speed) < 1e-2: speed = 0.0
        if abs(heading) < 1e-3: heading = 0.0

        return speed, heading

    def _follow(self, speed, heading):
        cmd_vel = Twist()

        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def listener_callback(self, msg):
        # Logs the received messages from a topic
        # self.get_logger().info(f'Message Type: {type(msg)}')

        # Convert ros to cv image type
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.first_frame:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.speed_reference = 0.5 * 0.3 * self.frame_width + 0.5 * 0.3 * self.frame_height
            self.heading_reference = 0.5 * self.frame_width
            # self.speed_err_min = self.speed_reference - (0.5 * self.frame_width + 0.5 * self.frame_height)
            # self.speed_err_max = self.speed_reference - 0
            # self.heading_err_min = self.heading_reference - self.frame_width
            # self.heading_err_max = self.heading_reference - 0
            self.first_frame = False

        result = self._detect(frame)
        if result is not None:
            cx, cy, w, h, (x, y, bw, bh) = result
            # Logging pixels from top-left origin
            self.get_logger().info(f'centroid=({cx:.1f},{cy:.1f}) size=({w:.0f},{h:.0f})')

            # Draw bbox + centroid
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

            speed, heading = self._plan(cx, bw, bh)
        else:
            speed, heading = 0.0, 0.0
            self.pid_speed.reset()
            self.pid_heading.reset()

        self.get_logger().info(f'Speed: {speed}; Heading: {heading}')
        self._follow(speed, heading)

        # Show or display frames on a window
        if self.show_video:
            cv.imshow('Red Ball Tracker: Frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.show_video = False
                cv.destroyAllWindows()


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'RedBallTracker' class node
    red_ball_tracker = RedBallTracker()

    # Spin up the created node
    rclpy.spin(red_ball_tracker)

    # Explicit declaration to destroy the node object and shutdown rclpy
    red_ball_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
