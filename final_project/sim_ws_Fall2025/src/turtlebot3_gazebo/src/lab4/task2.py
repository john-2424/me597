#!/usr/bin/env python3

import math
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from ament_index_python.packages import get_package_share_directory
import yaml


class Task2(Node):
    """
    Environment localization and navigation task.

    Step 1: ROS I/O + state machine skeleton
    Step 2: Map loading (Task 1 map) + world <-> grid conversions
    """

    def __init__(self):
        super().__init__('task2_node')

        # -------------------------
        # Parameters
        # -------------------------
        # Namespace for topics (can be empty string)
        self.declare_parameter('namespace', '')
        # Map name parameter (will be used for loading map)
        self.declare_parameter('map', 'map')
        # Inflation kernel size (in pixels) for obstacle dilation
        self.declare_parameter('inflation_kernel', 5)

        self.ns = self.get_parameter('namespace').get_parameter_value().string_value
        if self.ns and not self.ns.startswith('/'):
            self.ns = '/' + self.ns

        self.map_name = self.get_parameter('map').get_parameter_value().string_value
        self.inflation_kernel = (
            self.get_parameter('inflation_kernel').get_parameter_value().integer_value
        )

        # -------------------------
        # Map-related members
        # -------------------------
        self.map_loaded = False
        self.map_resolution = None     # meters / cell
        self.map_origin = None         # [x, y, yaw] in map frame
        self.map_width = None          # number of columns
        self.map_height = None         # number of rows

        self.static_occupancy = None   # 2D numpy array: 0 free, 1 occupied
        self.inflated_occupancy = None # 2D numpy array: 0 free, 1 occupied (inflated)

        # Load map from Task 1
        self._load_map_from_yaml()

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

        self.get_logger().info('Task2 node initialized: ROS I/O + map loading ready.')

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
        if base_name.startswith('/'):
            base_name = base_name[1:]
        return f'{self.ns}/{base_name}'

    # ----------------------------------------------------------------------
    # Map loading utilities
    # ----------------------------------------------------------------------
    def _load_map_from_yaml(self):
        """
        Load map.yaml + map.pgm from the turtlebot3_gazebo package using relative paths.
        Build a binary occupancy grid and an inflated occupancy grid.
        """
        try:
            pkg_share = get_package_share_directory('turtlebot3_gazebo')
        except Exception as e:
            self.get_logger().error(
                f'Failed to get package share directory for turtlebot3_gazebo: {e}'
            )
            return

        map_yaml_path = os.path.join(pkg_share, 'maps', self.map_name + '.yaml')
        if not os.path.exists(map_yaml_path):
            self.get_logger().error(f'Map YAML not found: {map_yaml_path}')
            return

        self.get_logger().info(f'Loading map YAML: {map_yaml_path}')

        try:
            with open(map_yaml_path, 'r') as f:
                map_yaml = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to parse map YAML: {e}')
            return

        # Extract basic map info
        image_path = map_yaml.get('image', None)
        resolution = map_yaml.get('resolution', None)
        origin = map_yaml.get('origin', None)
        occupied_thresh = map_yaml.get('occupied_thresh', 0.65)
        free_thresh = map_yaml.get('free_thresh', 0.196)

        if image_path is None or resolution is None or origin is None:
            self.get_logger().error('Map YAML missing required fields (image, resolution, origin).')
            return

        # Resolve image path (may be relative to YAML)
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

        if not os.path.exists(image_path):
            self.get_logger().error(f'Map image not found: {image_path}')
            return

        self.get_logger().info(f'Loading map image: {image_path}')

        # Load PGM as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error('Failed to load map image with cv2.')
            return

        # Convert to [0, 1] float
        img_normalized = img.astype(np.float32) / 255.0

        # Build binary occupancy grid: 1 = occupied, 0 = free
        # ROS map convention:
        # - values > free_thresh -> free
        # - values < occupied_thresh -> occupied
        # - otherwise unknown (we treat unknown as occupied for safety)
        occ = np.ones_like(img_normalized, dtype=np.uint8)  # default: occupied
        occ[img_normalized > free_thresh] = 0               # free
        occ[img_normalized < occupied_thresh] = 1           # occupied
        # unknown (between thresholds) stays 1 (occupied)

        self.static_occupancy = occ
        self.map_height, self.map_width = occ.shape
        self.map_resolution = float(resolution)
        self.map_origin = origin  # [x, y, yaw]

        self.get_logger().info(
            f'Map loaded: {self.map_width} x {self.map_height} cells, '
            f'resolution = {self.map_resolution:.3f} m/cell, '
            f'origin = ({self.map_origin[0]:.2f}, {self.map_origin[1]:.2f}, {self.map_origin[2]:.2f})'
        )

        # Inflate obstacles using dilation
        kernel_size = max(1, int(self.inflation_kernel))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inflated = cv2.dilate(self.static_occupancy, kernel, iterations=1)
        self.inflated_occupancy = inflated

        self.map_loaded = True
        self.get_logger().info(
            f'Obstacle inflation done with kernel size {kernel_size}.'
        )

    # ----------------------------------------------------------------------
    # World <-> grid conversion helpers
    # ----------------------------------------------------------------------
    def world_to_grid(self, x: float, y: float):
        """
        Convert world/map frame coordinates (x, y) to grid indices (row, col).
        - row corresponds to image row (0 at top)
        - col corresponds to image column (0 at left)
        """
        if not self.map_loaded:
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        # Position relative to origin (bottom-left of the map)
        dx = x - origin_x
        dy = y - origin_y

        # col: straightforward from dx
        col = int(dx / res)

        # row: image row 0 is top, but origin y is bottom-left
        # index from bottom = dy / res
        # row = (map_height - 1) - index_from_bottom
        index_from_bottom = int(dy / res)
        row = (self.map_height - 1) - index_from_bottom

        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return None

        return row, col

    def grid_to_world(self, row: int, col: int):
        """
        Convert grid indices (row, col) to world/map frame coordinates (x, y).
        Returns the center of the cell.
        """
        if not self.map_loaded:
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        # x is from left to right
        x = origin_x + (col + 0.5) * res

        # y: origin is bottom-left, row=0 is top
        # index_from_bottom = map_height - row - 1
        index_from_bottom = self.map_height - row - 1
        y = origin_y + (index_from_bottom + 0.5) * res

        return x, y

    # ----------------------------------------------------------------------
    # Subscriber callbacks
    # ----------------------------------------------------------------------
    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg
        if self.state == 'WAIT_FOR_POSE':
            self.state = 'WAIT_FOR_GOAL'
            self.get_logger().info('AMCL pose received. Waiting for goal.')

        # Optional debug: show current grid cell
        if self.map_loaded:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            grid_idx = self.world_to_grid(x, y)
            if grid_idx is not None:
                r, c = grid_idx
                self.get_logger().debug(
                    f'AMCL pose in grid: row={r}, col={c}'
                )

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
        # Will be used later for dynamic obstacles.

    # ----------------------------------------------------------------------
    # Main timer callback / state machine skeleton
    # ----------------------------------------------------------------------
    def timer_cb(self):
        self.get_logger().info(
            f'Task2 state: {self.state}',
            throttle_duration_sec=1.0
        )

        if not self.map_loaded:
            self.get_logger().warn(
                'Map not loaded yet; navigation disabled.',
                throttle_duration_sec=5.0
            )
            return

        if self.state == 'WAIT_FOR_POSE':
            # Do nothing until we have a valid AMCL pose
            return

        if self.state == 'WAIT_FOR_GOAL':
            # Robot is localized; wait for a goal from RViz / grader
            return

        if self.state == 'PLAN_GLOBAL':
            # A* planning will be implemented in the next step.
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
