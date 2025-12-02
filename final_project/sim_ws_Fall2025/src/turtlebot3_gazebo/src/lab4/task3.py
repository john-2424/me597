#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node

# ROS message types
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Path

# Map / image handling
import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory


class Task3(Node):
    """
    Task 3: Search and Localize colored balls.

    Layers implemented so far:
    - Layer 0: Boilerplate & ROS wiring
    - Layer 1: Map loading & occupancy utilities
    """

    def __init__(self):
        super().__init__('task3_node')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('namespace', '')
        self.declare_parameter('map', 'map')          # logical map name (map.yaml in maps/)
        self.declare_parameter('inflation_kernel', 6) # for static obstacle inflation

        self.ns = self.get_parameter('namespace').get_parameter_value().string_value
        if self.ns and not self.ns.startswith('/'):
            self.ns = '/' + self.ns

        self.map_name = self.get_parameter('map').get_parameter_value().string_value
        self.inflation_kernel = (
            self.get_parameter('inflation_kernel').get_parameter_value().integer_value
        )

        # -------------------------
        # Map-related members (Layer 1)
        # -------------------------
        self.map_loaded = False
        self.map_resolution = None     # meters / cell
        self.map_origin = None         # [x, y, yaw]
        self.map_width = None          # cols
        self.map_height = None         # rows

        self.static_occupancy = None   # 0 free, 1 occupied
        self.inflated_occupancy = None # 0 free, 1 occupied (inflated)
        self.dynamic_occupancy = None  # 0 free, 1 occupied (from LaserScan; Layer 3 will use this)

        # -------------------------
        # Robot state
        # -------------------------
        self.current_pose = None          # PoseWithCovarianceStamped
        self.latest_scan = None           # LaserScan
        self.latest_image = None          # Image (OpenCV conversion later)

        # -------------------------
        # Waypoints & navigation (later layers)
        # -------------------------
        self.patrol_waypoints = []        # list of (x, y) in map frame
        self.current_waypoint_idx = 0

        self.global_path_points = []      # list[(x, y)]
        self.active_path_points = []      # list[(x, y)]
        self.current_path_index = 0

        # -------------------------
        # High-level task state machine
        # -------------------------
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
        # Map loading (Layer 1)
        # -------------------------
        self._load_map_from_yaml()

        # -------------------------
        # Timer / main loop
        # -------------------------
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Task3 node initialized (Layers 0 + 1 ready).')

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
    # Map loading utilities (Layer 1)
    # ----------------------------------------------------------------------
    def _load_map_from_yaml(self):
        """
        Load map YAML and image from the turtlebot3_gazebo package,
        build static and inflated occupancy grids, and initialize dynamic layer.
        """
        try:
            pkg_share = get_package_share_directory('turtlebot3_gazebo')
        except Exception as e:
            self.get_logger().error(
                f'[Layer 1] Failed to get package share directory: {e}'
            )
            return

        # Use map_name parameter, assume <map_name>.yaml in maps/
        map_yaml_path = os.path.join(pkg_share, 'maps', self.map_name + '.yaml')
        if not os.path.exists(map_yaml_path):
            self.get_logger().error(
                f'[Layer 1] Map YAML not found: {map_yaml_path}'
            )
            return

        self.get_logger().info(f'[Layer 1] Loading map YAML: {map_yaml_path}')

        # Parse YAML
        try:
            with open(map_yaml_path, 'r') as f:
                map_yaml = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'[Layer 1] Failed to parse map YAML: {e}')
            return

        image_path = map_yaml.get('image', None)
        resolution = map_yaml.get('resolution', None)
        origin = map_yaml.get('origin', None)
        occupied_thresh = map_yaml.get('occupied_thresh', 0.65)
        free_thresh = map_yaml.get('free_thresh', 0.196)

        if image_path is None or resolution is None or origin is None:
            self.get_logger().error(
                '[Layer 1] Map YAML missing required fields (image, resolution, origin).'
            )
            return

        # If the image path is relative, make it relative to the YAML folder
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

        if not os.path.exists(image_path):
            self.get_logger().error(f'[Layer 1] Map image not found: {image_path}')
            return

        self.get_logger().info(f'[Layer 1] Loading map image: {image_path}')

        # Load grayscale image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error('[Layer 1] Failed to load map image with cv2.')
            return

        # Normalize to [0,1]
        img_normalized = img.astype(np.float32) / 255.0

        # Build static occupancy: 0 = free, 1 = occupied/unknown
        occ = np.ones_like(img_normalized, dtype=np.uint8)  # default occupied
        occ[img_normalized > free_thresh] = 0               # free
        occ[img_normalized < occupied_thresh] = 1           # occupied
        # Unknown remains 1 for safety

        self.static_occupancy = occ
        self.map_height, self.map_width = occ.shape
        self.map_resolution = float(resolution)
        self.map_origin = origin

        self.get_logger().info(
            f'[Layer 1] Map loaded: {self.map_width} x {self.map_height} cells, '
            f'resolution = {self.map_resolution:.3f} m/cell, '
            f'origin = ({self.map_origin[0]:.2f}, '
            f'{self.map_origin[1]:.2f}, {self.map_origin[2]:.2f})'
        )

        # Inflate static obstacles for navigation
        kernel_size = max(1, int(self.inflation_kernel))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inflated = cv2.dilate(self.static_occupancy, kernel, iterations=1)
        self.inflated_occupancy = inflated

        # Initialize dynamic occupancy layer as all free
        self.dynamic_occupancy = np.zeros_like(self.static_occupancy, dtype=np.uint8)

        self.map_loaded = True
        self.get_logger().info(
            f'[Layer 1] Obstacle inflation done with kernel size {kernel_size}. '
            'Dynamic obstacle layer initialized.'
        )

    def world_to_grid(self, x: float, y: float):
        """
        Convert world/map coordinates (x, y) to grid (row, col).
        Returns None if outside the map.
        """
        if not self.map_loaded:
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        dx = x - origin_x
        dy = y - origin_y

        # Column increases with +x
        col = int(dx / res)
        # Row index from top: map image origin is at top-left
        index_from_bottom = int(dy / res)
        row = (self.map_height - 1) - index_from_bottom

        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return None

        return row, col

    def grid_to_world(self, row: int, col: int):
        """
        Convert grid (row, col) to world/map coordinates (x, y),
        using cell centers.
        """
        if not self.map_loaded:
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        # x increases with column index
        x = origin_x + (col + 0.5) * res

        # y increases from bottom; row 0 is top of image
        index_from_bottom = self.map_height - row - 1
        y = origin_y + (index_from_bottom + 0.5) * res

        return x, y

    # ----------------------------------------------------------------------
    # Subscriber callbacks (Layer 0)
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
        Main state machine loop (very minimal so far).
        Later layers will implement:
        - Waypoint generation & route optimization
        - Navigation (A*, RRT*, controllers, obstacle avoidance)
        - Perception & ball localization
        """
        self.get_logger().info(
            f'[Layer 0+1] State = {self.state}, '
            f'map_loaded={self.map_loaded}, '
            f'pose_received={self.current_pose is not None}, '
            f'scan_received={self.latest_scan is not None}, '
            f'image_received={self.latest_image is not None}',
            throttle_duration_sec=1.0
        )

        if self.map_loaded:
            test_xy = (0.0, 0.0)
            idx = self.world_to_grid(*test_xy)
            if idx is not None:
                back_xy = self.grid_to_world(*idx)
                self.get_logger().info(
                    f'Test world->grid->world: {test_xy} -> {idx} -> {back_xy}',
                    throttle_duration_sec=5.0
                )
        
        if self.map_loaded and self.static_occupancy is not None:
            self.get_logger().info(
                f'static_occupancy shape = {self.static_occupancy.shape}, '
                f'inflated_occupancy shape = {self.inflated_occupancy.shape}, '
                f'dynamic_occupancy shape = {self.dynamic_occupancy.shape}',
                throttle_duration_sec=5.0
            )

        # For now, keep the robot stopped.
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
