#!/usr/bin/env python3

import math
import os
import heapq

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
    Step 3: Global path planner (A*) using inflated occupancy grid
    """

    def __init__(self):
        super().__init__('task2_node')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('namespace', '')
        self.declare_parameter('map', 'map')
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
        self.map_origin = None         # [x, y, yaw]
        self.map_width = None          # cols
        self.map_height = None         # rows

        self.static_occupancy = None    # 0 free, 1 occupied
        self.inflated_occupancy = None  # 0 free, 1 occupied (inflated)

        # Load map from Task 1
        self._load_map_from_yaml()

        # -------------------------
        # State variables
        # -------------------------
        self.current_pose = None           # PoseWithCovarianceStamped
        self.current_goal = None           # PoseStamped
        self.latest_scan = None            # LaserScan

        self.state = 'WAIT_FOR_POSE'
        self.goal_active = False

        # Global path storage (list of (x, y))
        self.global_path_points = []
        self.current_path_index = 0

        # Timing (will be used later for scoring)
        self.navigation_start_time = None
        self.navigation_time_pub_msg = Float32()

        # -------------------------
        # Publishers
        # -------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self._ns_topic('cmd_vel'),
            10
        )

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

        self.navigation_time_pub = self.create_publisher(
            Float32,
            self._ns_topic('navigation_time'),
            10
        )

        # -------------------------
        # Subscriptions
        # -------------------------
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self._ns_topic('amcl_pose'),
            self.amcl_pose_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            self._ns_topic('move_base_simple/goal'),
            self.goal_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self._ns_topic('scan'),
            self.scan_callback,
            10
        )

        # -------------------------
        # Timer
        # -------------------------
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Task2 node initialized: ROS I/O + map + A* ready.')

    # ----------------------------------------------------------------------
    # Helper to handle namespace in topic names
    # ----------------------------------------------------------------------
    def _ns_topic(self, base_name: str) -> str:
        if not self.ns:
            return base_name
        if base_name.startswith('/'):
            base_name = base_name[1:]
        return f'{self.ns}/{base_name}'

    # ----------------------------------------------------------------------
    # Map loading utilities
    # ----------------------------------------------------------------------
    def _load_map_from_yaml(self):
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

        image_path = map_yaml.get('image', None)
        resolution = map_yaml.get('resolution', None)
        origin = map_yaml.get('origin', None)
        occupied_thresh = map_yaml.get('occupied_thresh', 0.65)
        free_thresh = map_yaml.get('free_thresh', 0.196)

        if image_path is None or resolution is None or origin is None:
            self.get_logger().error('Map YAML missing required fields (image, resolution, origin).')
            return

        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

        if not os.path.exists(image_path):
            self.get_logger().error(f'Map image not found: {image_path}')
            return

        self.get_logger().info(f'Loading map image: {image_path}')

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error('Failed to load map image with cv2.')
            return

        img_normalized = img.astype(np.float32) / 255.0

        occ = np.ones_like(img_normalized, dtype=np.uint8)  # default: occupied
        occ[img_normalized > free_thresh] = 0               # free
        occ[img_normalized < occupied_thresh] = 1           # occupied
        # unknown stays 1 (occupied) for safety

        self.static_occupancy = occ
        self.map_height, self.map_width = occ.shape
        self.map_resolution = float(resolution)
        self.map_origin = origin

        self.get_logger().info(
            f'Map loaded: {self.map_width} x {self.map_height} cells, '
            f'resolution = {self.map_resolution:.3f} m/cell, '
            f'origin = ({self.map_origin[0]:.2f}, {self.map_origin[1]:.2f}, {self.map_origin[2]:.2f})'
        )

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
        if not self.map_loaded:
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        dx = x - origin_x
        dy = y - origin_y

        col = int(dx / res)
        index_from_bottom = int(dy / res)
        row = (self.map_height - 1) - index_from_bottom

        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return None

        return row, col

    def grid_to_world(self, row: int, col: int):
        if not self.map_loaded:
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        x = origin_x + (col + 0.5) * res
        index_from_bottom = self.map_height - row - 1
        y = origin_y + (index_from_bottom + 0.5) * res

        return x, y

    # ----------------------------------------------------------------------
    # A* global path planner
    # ----------------------------------------------------------------------
    def is_cell_free(self, row: int, col: int) -> bool:
        """Check if a cell is free in the inflated occupancy grid."""
        if (
            row < 0 or row >= self.map_height or
            col < 0 or col >= self.map_width
        ):
            return False
        return self.inflated_occupancy[row, col] == 0

    def astar_plan(self, start_world, goal_world):
        """
        Run A* on the inflated occupancy grid.

        :param start_world: (x, y) in world frame
        :param goal_world:  (x, y) in world frame
        :return: list of (x, y) world coordinates representing the path, or None
        """
        if not self.map_loaded:
            self.get_logger().error('Cannot run A*: map not loaded.')
            return None

        start_idx = self.world_to_grid(start_world[0], start_world[1])
        goal_idx = self.world_to_grid(goal_world[0], goal_world[1])

        if start_idx is None or goal_idx is None:
            self.get_logger().warn('Start or goal is outside the map.')
            return None

        s_row, s_col = start_idx
        g_row, g_col = goal_idx

        if not self.is_cell_free(s_row, s_col):
            self.get_logger().warn('Start cell is occupied after inflation.')
            return None

        if not self.is_cell_free(g_row, g_col):
            self.get_logger().warn('Goal cell is occupied after inflation.')
            return None

        # 8-connected neighbors: (dr, dc, cost)
        neighbors = [
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            ( 0, -1, 1.0),
            ( 0,  1, 1.0),
            (-1, -1, math.sqrt(2)),
            (-1,  1, math.sqrt(2)),
            ( 1, -1, math.sqrt(2)),
            ( 1,  1, math.sqrt(2)),
        ]

        def heuristic(r, c):
            return math.hypot(r - g_row, c - g_col)

        open_set = []
        heapq.heappush(open_set, (0.0, (s_row, s_col)))

        came_from = {}          # (row, col) -> (parent_row, parent_col)
        g_score = { (s_row, s_col): 0.0 }

        closed_set = set()

        iterations = 0
        max_iterations = self.map_width * self.map_height  # very loose cap

        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, (cur_row, cur_col) = heapq.heappop(open_set)

            if (cur_row, cur_col) in closed_set:
                continue

            if (cur_row, cur_col) == (g_row, g_col):
                # Reconstruct path
                return self._reconstruct_path(came_from, (cur_row, cur_col))

            closed_set.add((cur_row, cur_col))

            for dr, dc, move_cost in neighbors:
                nr = cur_row + dr
                nc = cur_col + dc

                if not self.is_cell_free(nr, nc):
                    continue

                neighbor = (nr, nc)
                tentative_g = g_score[(cur_row, cur_col)] + move_cost

                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(nr, nc)
                came_from[neighbor] = (cur_row, cur_col)
                heapq.heappush(open_set, (f_score, neighbor))

        self.get_logger().warn('A*: failed to find a path within iteration limit.')
        return None

    def _reconstruct_path(self, came_from, current_cell):
        """
        Reconstruct a world-frame path from A* came_from dict and final cell.
        """
        path_cells = [current_cell]
        while current_cell in came_from:
            current_cell = came_from[current_cell]
            path_cells.append(current_cell)

        path_cells.reverse()

        # Convert grid cells to world coordinates
        path_world = []
        for (r, c) in path_cells:
            world_xy = self.grid_to_world(r, c)
            if world_xy is not None:
                path_world.append(world_xy)

        self.get_logger().info(f'A*: path length (cells) = {len(path_world)}')
        return path_world

    def build_path_msg(self, path_points):
        """
        Build a nav_msgs/Path from a list of (x, y) points in world coordinates.
        """
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for (x, y) in path_points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            # Orientation: we can leave as default; controller will handle yaw.
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        return msg

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
        # Will be used later for dynamic obstacles.

    # ----------------------------------------------------------------------
    # Main timer callback / state machine
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
            return

        if self.state == 'WAIT_FOR_GOAL':
            return

        if self.state == 'PLAN_GLOBAL':
            self._handle_plan_global()
            return

        if self.state == 'FOLLOW_PATH':
            # For now, we only publish zero cmd_vel to keep robot stationary
            # while we focus on global planning. Path following comes later.
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            return

        if self.state == 'REPLAN_LOCAL':
            # RRT* local replanning will go here in a later step.
            return

        if self.state == 'GOAL_REACHED':
            # Goal handling, timing, and reset will be implemented later.
            return

    def _handle_plan_global(self):
        """
        Called in PLAN_GLOBAL state: run A* from current pose to goal, publish Path, then switch to FOLLOW_PATH.
        """
        if self.current_pose is None or self.current_goal is None:
            self.get_logger().warn('PLAN_GLOBAL: missing pose or goal.')
            return

        start_x = self.current_pose.pose.pose.position.x
        start_y = self.current_pose.pose.pose.position.y

        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y

        self.get_logger().info(
            f'PLAN_GLOBAL: running A* from '
            f'({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f}).'
        )

        path_world = self.astar_plan((start_x, start_y), (goal_x, goal_y))

        if path_world is None or len(path_world) < 2:
            self.get_logger().warn('PLAN_GLOBAL: A* failed or path too short. Staying in PLAN_GLOBAL.')
            return

        # Store path and reset index
        self.global_path_points = path_world
        self.current_path_index = 0

        # Publish Path for visualization
        path_msg = self.build_path_msg(path_world)
        self.global_path_pub.publish(path_msg)

        self.get_logger().info('PLAN_GLOBAL: A* path computed and published. Switching to FOLLOW_PATH.')
        self.state = 'FOLLOW_PATH'


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
