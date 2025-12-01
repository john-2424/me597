#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped


class Task2(Node):
    """
    Environment localization and navigation task.

    Step 1: ROS I/O and node skeleton.
    - Set up parameters
    - Subscriptions (AMCL pose, goal, LaserScan)
    - Publications (cmd_vel, global/local path, timing)
    - Basic state variables and timer
    """

    def __init__(self):
        super().__init__('task2_node')

        # -------------------------
        # Parameters
        # -------------------------
        # Namespace for topics (can be empty string)
        self.declare_parameter('namespace', '')
        # Map name parameter (will be used later for loading map)
        self.declare_parameter('map', 'map')

        self.ns = self.get_parameter('namespace').get_parameter_value().string_value
        if self.ns and not self.ns.startswith('/'):
            self.ns = '/' + self.ns

        # -------------------------
        # State variables
        # -------------------------
        self.current_pose = None           # PoseWithCovarianceStamped
        self.current_goal = None           # PoseStamped
        self.latest_scan = None            # LaserScan

        # Simple state machine placeholder
        self.state = 'WAIT_FOR_POSE'
        self.goal_active = False

        # Timing (will be used later for scoring)
        self.navigation_start_time = None
        self.navigation_time_pub_msg = Float32()

        # -------------------------
        # Publishers
        # -------------------------
        # Velocity command publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self._ns_topic('cmd_vel'),
            10
        )

        # Global path publisher (for A* path visualization)
        self.global_path_pub = self.create_publisher(
            Path,
            self._ns_topic('global_plan'),
            10
        )

        # Local path publisher (for RRT* detour visualization)
        self.local_path_pub = self.create_publisher(
            Path,
            self._ns_topic('local_plan'),
            10
        )

        # Navigation time publisher (for grading / debugging)
        self.navigation_time_pub = self.create_publisher(
            Float32,
            self._ns_topic('navigation_time'),
            10
        )

        # -------------------------
        # Subscriptions
        # -------------------------
        # AMCL localization pose
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self._ns_topic('amcl_pose'),
            self.amcl_pose_callback,
            10
        )

        # Goal pose from RViz / grader
        # Using relative topic name so launch/namespace can remap if needed
        self.goal_sub = self.create_subscription(
            PoseStamped,
            self._ns_topic('move_base_simple/goal'),
            self.goal_callback,
            10
        )

        # LaserScan for obstacle detection (trash cans / static obstacles)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self._ns_topic('scan'),
            self.scan_callback,
            10
        )

        # -------------------------
        # Timer
        # -------------------------
        # Main control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Task2 node initialized: ROS I/O and skeleton ready.')

    # ----------------------------------------------------------------------
    # Helper to handle namespace in topic names
    # ----------------------------------------------------------------------
    def _ns_topic(self, base_name: str) -> str:
        """
        Prepend namespace (if any) to a topic name, keeping it relative-friendly.

        Example:
        - ns = ''     -> 'cmd_vel'
        - ns = '/tb3' -> '/tb3/cmd_vel'
        """
        if not self.ns:
            return base_name
        # Avoid double slashes
        if base_name.startswith('/'):
            base_name = base_name[1:]
        return f'{self.ns}/{base_name}'

    # ----------------------------------------------------------------------
    # Subscriber callbacks
    # ----------------------------------------------------------------------
    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg
        if self.state == 'WAIT_FOR_POSE':
            self.state = 'WAIT_FOR_GOAL'
            self.get_logger().info('AMCL pose received. Waiting for goal.')

    def goal_callback(self, msg: PoseStamped):
        self.current_goal = msg
        self.goal_active = True
        self.navigation_start_time = self.get_clock().now()
        self.state = 'PLAN_GLOBAL'
        self.get_logger().info(
            f'New goal received at '
            f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}). '
            'Switching to PLAN_GLOBAL state.'
        )

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        # For now we just store it; it will be used later for obstacle handling.

    # ----------------------------------------------------------------------
    # Main timer callback / state machine skeleton
    # ----------------------------------------------------------------------
    def timer_cb(self):
        # High-level state debugging (throttled)
        self.get_logger().info(
            f'Task2 state: {self.state}',
            throttle_duration_sec=1.0
        )

        if self.state == 'WAIT_FOR_POSE':
            # Do nothing until we have a valid AMCL pose
            return

        if self.state == 'WAIT_FOR_GOAL':
            # Robot is localized; wait for a goal from RViz / grader
            return

        if self.state == 'PLAN_GLOBAL':
            # Step 1: we only define skeleton; actual A* planning will be
            # implemented in later steps.
            # For now, just log and transition to FOLLOW_PATH placeholder.
            self.get_logger().info('PLAN_GLOBAL: placeholder (A* not implemented yet).')
            self.state = 'FOLLOW_PATH'
            return

        if self.state == 'FOLLOW_PATH':
            # Placeholder for path following logic.
            # For now, just publish zero cmd_vel to keep robot stationary.
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            return

        if self.state == 'REPLAN_LOCAL':
            # Placeholder for RRT* local replanning.
            # Will be implemented in later steps.
            return

        if self.state == 'GOAL_REACHED':
            # Placeholder for goal reached behavior (timing, reset, etc.)
            return


def main(args=None):
    rclpy.init(args=args)

    task2 = Task2()

    try:
        rclpy.spin(task2)
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
