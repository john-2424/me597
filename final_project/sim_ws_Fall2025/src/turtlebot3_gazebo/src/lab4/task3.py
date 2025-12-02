#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# ROS message types
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Path

# We will add more imports (cv2, numpy, yaml, etc.) in later layers.


class Task3(Node):
    """
    Task 3: Search and Localize colored balls.

    Layer 0: Boilerplate & ROS wiring
    - Set up node
    - Set up publishers/subscribers
    - Basic state machine skeleton and timer
    """

    def __init__(self):
        super().__init__('task3_node')

        # -------------------------
        # Parameters (minimal for now)
        # -------------------------
        self.declare_parameter('namespace', '')
        self.declare_parameter('map', 'map')  # logical map name (map.yaml in maps/)
        
        self.ns = self.get_parameter('namespace').get_parameter_value().string_value
        if self.ns and not self.ns.startswith('/'):
            self.ns = '/' + self.ns

        # -------------------------
        # Core state / member variables
        # -------------------------
        # Map-related (will be filled in Layer 1)
        self.map_loaded = False
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None

        self.static_occupancy = None
        self.inflated_occupancy = None
        self.dynamic_occupancy = None

        # Robot state
        self.current_pose = None          # PoseWithCovarianceStamped
        self.latest_scan = None           # LaserScan
        self.latest_image = None          # Image (OpenCV conversion later)

        # Waypoints & navigation (later layers)
        self.patrol_waypoints = []        # list of (x, y) in map frame
        self.current_waypoint_idx = 0

        self.global_path_points = []      # list[(x, y)]
        self.active_path_points = []      # list[(x, y)]
        self.current_path_index = 0

        # High-level task state machine
        self.state = 'WAIT_FOR_POSE'      # other states will be added later

        # -------------------------
        # Publishers
        # -------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self._ns_topic('cmd_vel'),
            10
        )

        # For visualization/debug (will be used by later layers)
        self.global_path_pub = self.create_publisher(
            Path,
            self._ns_topic('global_plan'),
            10
        )

        self.local_path_pub = self.create_publisher(
            Path,
            self._ns_topic('local_plan'),
            10
        )

        # -------------------------
        # Subscriptions
        # -------------------------
        # AMCL pose (map frame)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self._ns_topic('amcl_pose'),
            self.amcl_pose_callback,
            10
        )

        # LaserScan for obstacles and later for ball range estimation
        self.scan_sub = self.create_subscription(
            LaserScan,
            self._ns_topic('scan'),
            self.scan_callback,
            10
        )

        # Camera image for color detection
        self.image_sub = self.create_subscription(
            Image,
            self._ns_topic('camera/image_raw'),
            self.image_callback,
            10
        )

        # -------------------------
        # Timer / main loop
        # -------------------------
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Task3 node initialized (Layer 0: wiring + skeleton).')

    # ----------------------------------------------------------------------
    # Helper to handle namespace in topic names
    # ----------------------------------------------------------------------
    def _ns_topic(self, base_name: str) -> str:
        """
        Prefix a topic with the namespace (if any).
        """
        if not self.ns:
            return base_name
        if base_name.startswith('/'):
            base_name = base_name[1:]
        return f'{self.ns}/{base_name}'

    # ----------------------------------------------------------------------
    # Subscriber callbacks (minimal for now)
    # ----------------------------------------------------------------------
    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def image_callback(self, msg: Image):
        self.latest_image = msg

    # ----------------------------------------------------------------------
    # Timer callback / state machine skeleton
    # ----------------------------------------------------------------------
    def timer_cb(self):
        """
        Main state machine loop (very minimal in Layer 0).
        Later layers will implement:
        - Map loading & waypoint generation
        - Navigation (A*, RRT*, controllers)
        - Perception & ball localization
        """
        self.get_logger().info(
            f'[Layer 0] State = {self.state}, '
            f'pose_received={self.current_pose is not None}, '
            f'scan_received={self.latest_scan is not None}, '
            f'image_received={self.latest_image is not None}',
            throttle_duration_sec=1.0
        )

        # For now, do nothing but stop the robot.
        self._publish_stop()

    # ----------------------------------------------------------------------
    # Simple helper to stop the robot
    # ----------------------------------------------------------------------
    def _publish_stop(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)

    task3 = Task3()

    try:
        rclpy.spin(task3)
    except KeyboardInterrupt:
        pass
    finally:
        task3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
