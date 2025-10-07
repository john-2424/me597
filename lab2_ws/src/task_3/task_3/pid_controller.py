import csv
import math
from datetime import datetime
from statistics import median

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class PID:
    def __init__(self, kp, ki, kd, 
                 i_limit=1.0
                 ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i = 0.0
        self.prev_e = 0.0
        self.i_limit = i_limit
        self.first = True

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def step(self, e, dt):
        # Integral with clamping (anti-windup)
        self.i += e * dt
        self.i = max(-self.i_limit, min(self.i_limit, self.i))

        # Derivative (protect first step)
        if self.first or dt <= 0.0:
            d = 0.0
            self.first = False
        else:
            d = (e - self.prev_e) / dt
        self.prev_e = e

        return self.kp * e + self.ki * self.i + self.kd * d

class PIDSpeedController(Node):
    def __init__(self):
        # Initialize with 'Node' constructor and properties
        super().__init__('pid_speed_controller')
        self.dtl = []
        self.distl = []
        self.el = []
        self.vl = []
        self.last_listener_message = None
        self.last_listener_time = self.get_clock().now()

        # ===== Parameters (declare + defaults) =====
        self.declare_parameter('setpoint_reference', 0.35)          # desired distance [m]
        # self.declare_parameter('tolerance', 0.05)         # stop band [m]
        self.declare_parameter('hz', 10.0)                # control rate [Hz]
        self.declare_parameter('kp', 0.4)
        self.declare_parameter('ki', 0.0006)
        self.declare_parameter('kd', 0.6)
        self.declare_parameter('i_limit', 0.1)            # integral clamp
        self.declare_parameter('max_speed', 0.15)          # m/s forward cap
        # self.declare_parameter('min_speed', 0.05)         # deadband to avoid micro jitter
        self.declare_parameter('sector_deg', 12.0)        # +/- sector around front
        self.declare_parameter('use_median', True)        # median vs min
        self.declare_parameter('reverse_ok', True)       # allow backing up if too close

        # ===== Get params =====
        self.setpoint_reference   = float(self.get_parameter('setpoint_reference').value)
        # self.tolerance  = float(self.get_parameter('tolerance').value)
        self.hz         = float(self.get_parameter('hz').value)
        kp              = float(self.get_parameter('kp').value)
        ki              = float(self.get_parameter('ki').value)
        kd              = float(self.get_parameter('kd').value)
        i_limit         = float(self.get_parameter('i_limit').value)
        self.max_speed  = float(self.get_parameter('max_speed').value)
        # self.min_speed  = float(self.get_parameter('min_speed').value)
        self.sector_deg = float(self.get_parameter('sector_deg').value)
        self.use_median = bool(self.get_parameter('use_median').value)
        self.reverse_ok = bool(self.get_parameter('reverse_ok').value)

        # Subscribe to the /scan topic
        self.scan_listener = self.create_subscription(
            LaserScan,
            '/robot/scan',
            self._scan_subscribe_callback,
            10
        )
        self.scan_listener  # prevent unused variable warning

        # Timer to process the received message in an interval
        timer_period = float(1 / self.hz)
        self.timer = self.create_timer(timer_period, self._process_timer_callback)

        # Publish to /cmd_vel topic
        self.cmd_vel_talker = self.create_publisher(Twist, '/robot/cmd_vel', 10)

        # Initialize PID controller
        self.pid = PID(kp, ki, kd, 
                       i_limit
                    )

        # Dist topic, with 'Float64' message type, 10 queue size
        # self.dist_talker = self.create_publisher(Float64, '/bot_dist', 10)

    def _scan_subscribe_callback(self, msg):
        # Assigns and Logs the received messages from the /scan topic
        self.last_listener_message = msg
        # dist = self._front_distance(self.last_listener_message)
        # self.get_logger().info(f'Listener: [{msg.angle_min}, {msg.angle_max}, {msg.angle_increment}, {msg.range_min}, {msg.range_max}, {len(msg.ranges)}, {dist}]')
        # self.get_logger().info(f'{[(i, i_range) for i, i_range in enumerate(msg.ranges)]}')
        # min_val = min(msg.ranges)
        # min_idx = msg.ranges.index(min_val)
        # self.get_logger().info(f'{min_idx}, {min_val} -> Front angle: {min_idx*msg.angle_increment}')
    
    def _front_distance(self, msg: LaserScan):
        # Compute indices for a symmetric sector around angle = 0 (front)
        total = len(msg.ranges)
        # self.get_logger().info(f'Total Ranges: {total}')
        if total == 0:
            return None

        # # Convert sector to radians and to index range
        # sector_rad = math.radians(self.sector_deg)
        # self.get_logger().info(f'Angle Increment: {msg.angle_increment}')
        # # Index of angle 0:
        # if msg.angle_increment == 0:
        #     return None
        # zero_idx = int(round((0.0 - msg.angle_min) / msg.angle_increment))
        # half_span = int(round(sector_rad / msg.angle_increment))
        # start = max(0, zero_idx - half_span)
        # end   = min(total - 1, zero_idx + half_span)

        front_index = 270
        return msg.ranges[front_index]
        start = front_index - 100
        end = front_index + 100

        window = []
        for i in range(start, end + 1):
            r = msg.ranges[i]
            if math.isfinite(r):
                # clamp to sensor limits
                r = max(msg.range_min, min(msg.range_max, r))
                window.append(r)

        self.get_logger().info(f'Window: {window}')
        if not window:
            return None

        return median(window) if self.use_median else min(window)
    
    def _process_timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_listener_time).nanoseconds / 1e9
        self.last_listener_time = now
        if self.dtl: self.dtl.append(self.dtl[-1] + dt)
        else: self.dtl.append(dt)

        twist = Twist()

        if self.last_listener_message is None or dt <= 0:
            self.cmd_vel_talker.publish(twist)
            return

        dist = self._front_distance(self.last_listener_message)
        # self.get_logger().info(f'**** Distance: {dist}')
        # self.dist_talker.publish(Float64(data=float(dist)))
        if dist is None or dist == 'inf':
            # No valid reading; stop for safety
            self.cmd_vel_talker.publish(twist)
            return

        self.distl.append(dist)
        e = dist - self.setpoint_reference   # positive if bot more than the target distance of the obstacle, negative if less
        # self.get_logger().info(f'**** Error: {e}')
        self.el.append(e)
        # at_goal = abs(e) <= self.tolerance

        # if at_goal:
        #     self.pid.reset()
        #     # Smooth stop
        #     twist.linear.x = 0.0
        #     self.cmd_vel_talker.publish(twist)
        #     return

        u = self.pid.step(e, dt)

        # Map PID output to forward speed
        v = u

        # If reverse is not allowed, block negative speeds
        # if not self.reverse_ok and v < 0.0:
        #     v = 0.0

        # Apply deadband and saturation
        # if abs(v) < self.min_speed:
        #     v = math.copysign(0.0, v)  # snap to 0 within deadband
        v = max(-self.max_speed, min(self.max_speed, v))
        # self.get_logger().info(f'**** Velocity: {v}')
        self.vl.append(v)

        twist.linear.x = v
        twist.angular.z = 0.0  # keep heading; add heading control if needed
        self.cmd_vel_talker.publish(twist)

        # Optional debug
        self.get_logger().info(
            f"dist={dist:.3f} e={e:.3f} v={v:.3f}"
        )

    def save_csv(self):
        filename = f'pid_{self.pid.kp}_{self.pid.ki}_{self.pid.kd}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'dist', 'error', 'velocity'])
            for dt, dist, e, v in zip(self.dtl, self.distl, self.el, self.vl):
                writer.writerow([dt, dist, e, v])
        self.get_logger().info(f"Saved log to {filename}")


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'PIDSpeedController' class node
    pid_speed_controller = PIDSpeedController()

    # Spin up the created node
    try:
        rclpy.spin(pid_speed_controller)
    except KeyboardInterrupt:
        # pid_speed_controller.save_csv()
        pass
    finally:
        # Explicit declaration to destroy the node object and shutdown rclpy
        pid_speed_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
