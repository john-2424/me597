import math
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from task_6.utils.detect import find_object_hsv, find_object_hsv_triangle, find_object_hsv_circle, LOWER_RED_1, LOWER_RED_2, UPPER_RED_1, UPPER_RED_2
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
        self.detector_mode = 'hsv_circle'  # hsv_circle, hsv_triangle or hsv

        # Runtime options/ROS Params
        self.declare_parameter('controller', 'pid')     # 'pid' or 'stanley'
        self.declare_parameter('search_mode', 'none')   # 'none', 'spiral', or 'hybrid'
        self.controller  = self.get_parameter('controller').get_parameter_value().string_value
        self.search_mode = self.get_parameter('search_mode').get_parameter_value().string_value

        # Frame info params
        self.first_frame = True
        self.frame_width = None
        self.frame_height = None

        # Tracking Reference params
        self.speed_reference = None
        self.speed_tol = 1
        self.heading_reference = None
        self.heading_tol = 1
        self.prev_speed = 0.0
        self.prev_heading = 0.0
        self.log_prev_speed = 0.0
        self.log_prev_heading = 0.0
        
        # PID parameters
        self.prev_sec = None
        self.speed_max = 0.20
        self.heading_max = 1.0
        self.turn_coeff = 1.0    # 0.20 coeff to stop bot by zeroing bot vel, 1.0 to ignore this logic
        self.speed_db   = 0.05      # ~5% size error
        self.heading_db = 0.02      # ~2% of half-frame
        self.max_speed_slew   = 0.05     # m/s per cycle
        self.max_heading_slew = 0.20     # rad/s per cycle
        # PIDs for speed and yaw
        # normalized-error tuning (err in [-1,1])
        self.pid_speed = PID(
            kp=0.18, ki=0.03, kd=0.08,
            i_limit=0.8,
            out_limit=(-self.speed_max, self.speed_max)
        )
        self.pid_heading = PID(
            kp=1.00, ki=0.02, kd=0.25,
            i_limit=0.8,
            out_limit=(-self.heading_max, self.heading_max)
        )

        self.scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        # Command Velocity Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def _scan_cb(self, msg: LaserScan):
        self.scan = msg

    def _red_likelihood(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # simple red mask
        m = cv.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | cv.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        return float(cv.countNonZero(m)) / float(m.size)  # fraction in [0,1]

    def _scan_gaps(self, width_m):
        if self.scan is None: return []
        r = np.array(self.scan.ranges, dtype=np.float32)
        ang0 = self.scan.angle_min
        dang = self.scan.angle_increment
        valid = np.isfinite(r)

        # Consider “open” if valid and not too close; use 1.0 m as a softer floor
        farish = r > 1.0
        open_mask = valid & farish

        gaps = []
        i = 0
        while i < len(open_mask):
            if open_mask[i]:
                j = i
                while j < len(open_mask) and open_mask[j]:
                    j += 1

                seg = r[i:j][np.isfinite(r[i:j])]
                if seg.size == 0:
                    i = j
                    continue

                depth = float(np.quantile(seg, 0.3))  # robust depth
                span  = (j - i) * dang
                gap_width = depth * span

                if gap_width >= width_m:
                    ang_center = ang0 + (i + j - 1) * 0.5 * dang
                    gaps.append((ang_center, gap_width))
                i = j
            else:
                i += 1

        gaps.sort(key=lambda t: t[1], reverse=True)
        return gaps

    def _detect(self, frame):
        if self.detector_mode == 'hsv_circle':
            return find_object_hsv_circle(frame)
        if self.detector_mode == 'hsv_triangle':
            return find_object_hsv_triangle(frame)
        return find_object_hsv(frame)

    def __slew(self, prev, new, max_delta):
        delta = max(-max_delta, min(max_delta, new - prev))
        return prev + delta
    
    def __plan_pid(self, cx, bw, bh, dt):
        # now_sec = self.get_clock().now().nanoseconds * 1e-9
        # dt = now_sec - self.prev_sec if self.prev_sec is not None else 1.0 / 10.0
        # self.prev_sec = now_sec

        speed = 0.0
        heading = 0.0        
            
        # speed_curr = 0.5 * bw + 0.5 * bh
        speed_curr = bw
        speed_err = (self.speed_reference - speed_curr) / max(self.speed_reference, 1e-6)
        speed_err = max(-1.5, min(1.5, speed_err))  # clamp outliers
        self.get_logger().info(f'[Speed] Ref: {self.speed_reference}; Curr: {speed_curr}; Err: {speed_err}')

        heading_curr = cx
        heading_err = (self.heading_reference - heading_curr) / max(heading_curr, 1e-6)
        heading_err = max(-1.5, min(1.5, heading_err))
        self.get_logger().info(f'[Heading] Ref: {self.heading_reference}; Curr: {heading_curr}; Err: {heading_err}')

        speed = self.pid_speed.step(speed_err, dt)
        heading = self.pid_heading.step(heading_err, dt)

        # deadbands (on normalized error)
        if abs(speed_err) < self.speed_db: speed = 0.0
        if abs(heading_err) < self.heading_db: heading = 0.0
        
        if abs(heading) > self.turn_coeff*self.heading_max:
            speed = 0.0
        
        # slew-rate limit commands
        speed   = self.__slew(self.prev_speed, speed, self.max_speed_slew)
        heading = self.__slew(self.prev_heading, heading, self.max_heading_slew)

        self.prev_speed, self.prev_heading = speed, heading

        return speed, heading

    def _plan(self, cx, bw, bh, dt):
        if self.controller == 'pid':
            return self.__plan_pid(cx, bw, bh, dt)

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

        # Compute dt
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        dt = now_sec - self.prev_sec if self.prev_sec is not None else 1.0 / 10.0
        self.prev_sec = now_sec

        if self.first_frame:
            self.frame_height, self.frame_width = frame.shape[:2]
            # self.speed_reference = 0.5 * 0.3 * self.frame_width + 0.5 * 0.3 * self.frame_height
            self.speed_reference = 0.08 * self.frame_width
            self.heading_reference = 0.5 * self.frame_width
            self.first_frame = False
            self.get_logger().info('Frame Parameters are set!')

        result = self._detect(frame)
        if result is not None:
            self.get_logger().info('Red Ball Detected!')
            cx, cy, w, h, (x, y, bw, bh) = result
            # Logging pixels from top-left origin
            self.get_logger().info(f'[Object - Red Ball] centroid=({cx:.1f},{cy:.1f}) size=({w:.0f},{h:.0f})')

            # Draw bbox + centroid
            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

            speed, heading = self._plan(cx, bw, bh, dt)
        else:
            self.get_logger().info('No Red Ball Detected!')
            # Lost target
            if self.controller == 'pid':
                if not self.pid_speed.is_reset():
                    self.pid_speed.reset()
                    self.prev_speed = 0.0
                    self.get_logger().info('Speed PID Reset!')
                if not self.pid_heading.is_reset():
                    self.pid_heading.reset()
                    self.prev_heading = 0.0
                    self.get_logger().info('Heading PID Reset!')

        if self.log_prev_speed != speed and self.log_prev_heading != heading:
            self.get_logger().info(f'[Robot] Speed: {speed}; Heading: {heading}')
            self.log_prev_speed, self.log_prev_heading = speed, heading
        
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
