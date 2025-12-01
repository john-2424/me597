#!/usr/bin/env python3

import math
import os
import heapq
import random

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
    Step 4: Dynamic obstacle layer from LaserScan (trash cans)
    Step 5: Blocked-path detection & trigger for local replanning
    Step 6: Local RRT* planner for detours around static obstacles
    """

    def __init__(self):
        super().__init__('task2_node')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('namespace', '')
        self.declare_parameter('map', 'map')
        self.declare_parameter('inflation_kernel', 5)
        # How many future waypoints to check for blockage
        self.declare_parameter('block_check_lookahead_points', 30)

        # RRT* parameters
        self.declare_parameter('rrt_max_iterations', 500)
        self.declare_parameter('rrt_step_size', 0.25)         # meters
        self.declare_parameter('rrt_goal_sample_rate', 0.2)   # probability [0,1]
        self.declare_parameter('rrt_neighbor_radius', 0.75)   # meters
        self.declare_parameter('rrt_local_range', 2.5)        # meters (sampling window radius)

        self.ns = self.get_parameter('namespace').get_parameter_value().string_value
        if self.ns and not self.ns.startswith('/'):
            self.ns = '/' + self.ns

        self.map_name = self.get_parameter('map').get_parameter_value().string_value
        self.inflation_kernel = (
            self.get_parameter('inflation_kernel').get_parameter_value().integer_value
        )
        self.block_check_lookahead_points = (
            self.get_parameter('block_check_lookahead_points')
            .get_parameter_value().integer_value
        )

        self.rrt_max_iterations = (
            self.get_parameter('rrt_max_iterations').get_parameter_value().integer_value
        )
        self.rrt_step_size = (
            self.get_parameter('rrt_step_size').get_parameter_value().double_value
        )
        self.rrt_goal_sample_rate = (
            self.get_parameter('rrt_goal_sample_rate').get_parameter_value().double_value
        )
        self.rrt_neighbor_radius = (
            self.get_parameter('rrt_neighbor_radius').get_parameter_value().double_value
        )
        self.rrt_local_range = (
            self.get_parameter('rrt_local_range').get_parameter_value().double_value
        )

        # -------------------------
        # Map-related members
        # -------------------------
        self.map_loaded = False
        self.map_resolution = None     # meters / cell
        self.map_origin = None         # [x, y, yaw]
        self.map_width = None          # cols
        self.map_height = None         # rows

        self.static_occupancy = None     # 0 free, 1 occupied
        self.inflated_occupancy = None   # 0 free, 1 occupied (inflated)
        self.dynamic_occupancy = None    # 0 free, 1 occupied (from LaserScan)

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

        # Global path (pure A*)
        self.global_path_points = []       # list[(x, y)]
        # Active path (global + local detours)
        self.active_path_points = []       # list[(x, y)]
        self.current_path_index = 0        # index into active_path_points

        # For local replanning (RRT*)
        self.local_replan_start_index = None
        self.local_replan_goal_index = None

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

        self.get_logger().info('Task2 node initialized: IO + map + A* + dynamic obstacles + blockage + RRT* ready.')

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

        # Inflate static obstacles
        kernel_size = max(1, int(self.inflation_kernel))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inflated = cv2.dilate(self.static_occupancy, kernel, iterations=1)
        self.inflated_occupancy = inflated

        # Initialize dynamic occupancy layer (all free initially)
        self.dynamic_occupancy = np.zeros_like(self.static_occupancy, dtype=np.uint8)

        self.map_loaded = True
        self.get_logger().info(
            f'Obstacle inflation done with kernel size {kernel_size}. '
            'Dynamic obstacle layer initialized.'
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
    # Pose utilities
    # ----------------------------------------------------------------------
    @staticmethod
    def _quat_to_yaw(q):
        x = q.x
        y = q.y
        z = q.z
        w = q.w

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ----------------------------------------------------------------------
    # Dynamic obstacle layer from LaserScan
    # ----------------------------------------------------------------------
    def update_dynamic_occupancy_from_scan(self):
        if not self.map_loaded or self.latest_scan is None or self.current_pose is None:
            return

        # Reset dynamic obstacles each scan
        self.dynamic_occupancy.fill(0)

        pose = self.current_pose.pose.pose
        rx = pose.position.x
        ry = pose.position.y
        yaw = self._quat_to_yaw(pose.orientation)

        scan = self.latest_scan
        angle = scan.angle_min

        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue
            if r < scan.range_min or r > scan.range_max:
                angle += scan.angle_increment
                continue

            # Beam in robot frame
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            # Transform to world
            wx = rx + math.cos(yaw) * lx - math.sin(yaw) * ly
            wy = ry + math.sin(yaw) * lx + math.cos(yaw) * ly

            idx = self.world_to_grid(wx, wy)
            if idx is not None:
                row, col = idx
                self.dynamic_occupancy[row, col] = 1

            angle += scan.angle_increment

        num_dyn = int(self.dynamic_occupancy.sum())
        self.get_logger().info(
            f'Updated dynamic occupancy. Occupied cells (approx): {num_dyn}',
            throttle_duration_sec=2.0
        )

    # ----------------------------------------------------------------------
    # A* global path planner
    # ----------------------------------------------------------------------
    def is_cell_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free considering both inflated static map and dynamic obstacles.
        """
        if (
            row < 0 or row >= self.map_height or
            col < 0 or col >= self.map_width
        ):
            return False

        if self.inflated_occupancy[row, col] != 0:
            return False

        if self.dynamic_occupancy is not None and self.dynamic_occupancy[row, col] != 0:
            return False

        return True

    def astar_plan(self, start_world, goal_world):
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
            self.get_logger().warn('Start cell is occupied (static or dynamic).')
            return None

        if not self.is_cell_free(g_row, g_col):
            self.get_logger().warn('Goal cell is occupied (static or dynamic).')
            return None

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

        came_from = {}
        g_score = {(s_row, s_col): 0.0}
        closed_set = set()

        iterations = 0
        max_iterations = self.map_width * self.map_height

        while open_set and iterations < max_iterations:
            iterations += 1
            current_f, (cur_row, cur_col) = heapq.heappop(open_set)

            if (cur_row, cur_col) in closed_set:
                continue

            if (cur_row, cur_col) == (g_row, g_col):
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
        path_cells = [current_cell]
        while current_cell in came_from:
            current_cell = came_from[current_cell]
            path_cells.append(current_cell)

        path_cells.reverse()

        path_world = []
        for (r, c) in path_cells:
            world_xy = self.grid_to_world(r, c)
            if world_xy is not None:
                path_world.append(world_xy)

        self.get_logger().info(f'A*: path length (cells) = {len(path_world)}')
        return path_world

    def build_path_msg(self, path_points):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for (x, y) in path_points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        return msg

    # ----------------------------------------------------------------------
    # Path blockage detection helpers
    # ----------------------------------------------------------------------
    def _nearest_path_index(self, x, y, path_points):
        if not path_points:
            return None, float('inf')

        best_idx = 0
        best_dist2 = float('inf')

        for i, (px, py) in enumerate(path_points):
            dx = px - x
            dy = py - y
            d2 = dx * dx + dy * dy
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i

        return best_idx, best_dist2

    def _is_waypoint_blocked(self, x, y) -> bool:
        idx = self.world_to_grid(x, y)
        if idx is None:
            return True
        row, col = idx
        return not self.is_cell_free(row, col)

    def _check_path_blocked_ahead(self):
        if not self.active_path_points or self.current_pose is None:
            return False, None, None

        rx = self.current_pose.pose.pose.position.x
        ry = self.current_pose.pose.pose.position.y

        nearest_idx, _ = self._nearest_path_index(rx, ry, self.active_path_points)
        if nearest_idx is None:
            return False, None, None

        max_idx = min(
            nearest_idx + self.block_check_lookahead_points,
            len(self.active_path_points) - 1
        )

        blocked = False
        first_blocked_idx = None

        for i in range(nearest_idx, max_idx + 1):
            x, y = self.active_path_points[i]
            if self._is_waypoint_blocked(x, y):
                blocked = True
                first_blocked_idx = i
                break

        if not blocked:
            return False, nearest_idx, max_idx

        reconnect_idx = min(first_blocked_idx + 5, len(self.active_path_points) - 1)

        self.get_logger().info(
            f'Path blocked detected: nearest_idx={nearest_idx}, '
            f'first_blocked_idx={first_blocked_idx}, reconnect_idx={reconnect_idx}'
        )

        return True, nearest_idx, reconnect_idx

    # ----------------------------------------------------------------------
    # RRT* helpers
    # ----------------------------------------------------------------------
    class _RRTNode:
        __slots__ = ('x', 'y', 'parent', 'cost')

        def __init__(self, x, y, parent=None, cost=0.0):
            self.x = x
            self.y = y
            self.parent = parent
            self.cost = cost

    def _segment_collision_free(self, x1, y1, x2, y2) -> bool:
        """
        Check if the line segment between (x1,y1) and (x2,y2) is collision-free
        using the static+dynamic occupancy grids.
        """
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0.0:
            idx = self.world_to_grid(x1, y1)
            if idx is None:
                return False
            r, c = idx
            return self.is_cell_free(r, c)

        # Sample along the segment at about half a grid resolution
        step = max(self.map_resolution * 0.5, 0.01)
        n_steps = max(int(dist / step), 1)

        for i in range(n_steps + 1):
            t = i / float(n_steps)
            x = x1 + t * dx
            y = y1 + t * dy
            idx = self.world_to_grid(x, y)
            if idx is None:
                return False
            r, c = idx
            if not self.is_cell_free(r, c):
                return False

        return True

    def _rrt_sample(self, center_x, center_y, goal_x, goal_y):
        """
        Sample a random free point in a local square window around center,
        occasionally sampling the goal directly (goal bias).
        """
        if random.random() < self.rrt_goal_sample_rate:
            return goal_x, goal_y

        for _ in range(100):
            # Uniform sample in local square around robot
            dx = (random.random() * 2.0 - 1.0) * self.rrt_local_range
            dy = (random.random() * 2.0 - 1.0) * self.rrt_local_range
            x = center_x + dx
            y = center_y + dy

            idx = self.world_to_grid(x, y)
            if idx is None:
                continue
            r, c = idx
            if self.is_cell_free(r, c):
                return x, y

        # Fallback
        return goal_x, goal_y

    def _rrt_nearest(self, nodes, x, y):
        best_node = None
        best_dist2 = float('inf')
        for node in nodes:
            dx = node.x - x
            dy = node.y - y
            d2 = dx * dx + dy * dy
            if d2 < best_dist2:
                best_dist2 = d2
                best_node = node
        return best_node

    def _rrt_steer(self, from_node, x, y):
        dx = x - from_node.x
        dy = y - from_node.y
        dist = math.hypot(dx, dy)
        if dist <= self.rrt_step_size:
            return x, y

        scale = self.rrt_step_size / dist
        return from_node.x + dx * scale, from_node.y + dy * scale

    def _rrt_neighbors(self, nodes, new_node):
        """
        Return nodes within neighbor_radius of new_node.
        """
        neighbors = []
        radius2 = self.rrt_neighbor_radius * self.rrt_neighbor_radius
        for node in nodes:
            dx = node.x - new_node.x
            dy = node.y - new_node.y
            if dx * dx + dy * dy <= radius2:
                neighbors.append(node)
        return neighbors

    def plan_rrt_star(self, start_xy, goal_xy, center_xy):
        """
        Plan a local path from start_xy to goal_xy using RRT* within a local window.
        :param start_xy: (x, y) start in world frame (robot or path start)
        :param goal_xy: (x, y) goal in world frame (reconnect point on global path)
        :param center_xy: (x, y) center of local window (robot pose)
        :return: list[(x, y)] from start to goal, or None if failed
        """
        sx, sy = start_xy
        gx, gy = goal_xy
        cx, cy = center_xy

        start_node = self._RRTNode(sx, sy, parent=None, cost=0.0)
        nodes = [start_node]
        goal_node = None

        for it in range(self.rrt_max_iterations):
            # 1) Sample
            sample_x, sample_y = self._rrt_sample(cx, cy, gx, gy)

            # 2) Nearest
            nearest = self._rrt_nearest(nodes, sample_x, sample_y)
            if nearest is None:
                continue

            # 3) Steer
            new_x, new_y = self._rrt_steer(nearest, sample_x, sample_y)

            # 4) Collision check
            if not self._segment_collision_free(nearest.x, nearest.y, new_x, new_y):
                continue

            new_node = self._RRTNode(new_x, new_y, parent=None, cost=float('inf'))

            # 5) Choose best parent among neighbors
            neighbors = self._rrt_neighbors(nodes, new_node)
            best_parent = nearest
            best_cost = nearest.cost + math.hypot(nearest.x - new_x, nearest.y - new_y)

            for nb in neighbors:
                d = math.hypot(nb.x - new_x, nb.y - new_y)
                if nb.cost + d < best_cost and self._segment_collision_free(nb.x, nb.y, new_x, new_y):
                    best_parent = nb
                    best_cost = nb.cost + d

            new_node.parent = best_parent
            new_node.cost = best_cost
            nodes.append(new_node)

            # 6) Rewire neighbors
            for nb in neighbors:
                d = math.hypot(nb.x - new_node.x, nb.y - new_node.y)
                if new_node.cost + d < nb.cost and self._segment_collision_free(new_node.x, new_node.y, nb.x, nb.y):
                    nb.parent = new_node
                    nb.cost = new_node.cost + d

            # 7) Check if we can connect to goal
            dist_to_goal = math.hypot(new_node.x - gx, new_node.y - gy)
            if dist_to_goal <= self.rrt_step_size * 2.0:
                if self._segment_collision_free(new_node.x, new_node.y, gx, gy):
                    goal_node = self._RRTNode(gx, gy, parent=new_node,
                                              cost=new_node.cost + dist_to_goal)
                    self.get_logger().info(f'RRT*: reached goal at iteration {it}.')
                    break

        if goal_node is None:
            self.get_logger().warn('RRT*: failed to find a local path.')
            return None

        # Reconstruct path from goal_node back to start
        path = []
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()

        self.get_logger().info(f'RRT*: local path length = {len(path)} waypoints.')
        return path

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
        self.update_dynamic_occupancy_from_scan()

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
            # Check if path ahead is blocked
            blocked, start_idx, goal_idx = self._check_path_blocked_ahead()
            if blocked:
                self.local_replan_start_index = start_idx
                self.local_replan_goal_index = goal_idx
                self.get_logger().info(
                    f'FOLLOW_PATH: path blocked. Triggering local replanning '
                    f'from index {start_idx} to {goal_idx}.'
                )
                self.state = 'REPLAN_LOCAL'
                return

            # Controller will be added later; keep robot stopped for now.
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            return

        if self.state == 'REPLAN_LOCAL':
            self._handle_replan_local()
            return

        if self.state == 'GOAL_REACHED':
            # Goal handling, timing, and reset will be implemented later.
            return

    def _handle_plan_global(self):
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

        # Store pure global path
        self.global_path_points = path_world

        # Active path initially matches global path (until RRT* detours modify it)
        self.active_path_points = list(path_world)
        self.current_path_index = 0
        self.local_replan_start_index = None
        self.local_replan_goal_index = None

        path_msg = self.build_path_msg(path_world)
        self.global_path_pub.publish(path_msg)

        self.get_logger().info('PLAN_GLOBAL: A* path computed and published. Switching to FOLLOW_PATH.')
        self.state = 'FOLLOW_PATH'

    def _handle_replan_local(self):
        """
        Called in REPLAN_LOCAL state: run RRT* between two points on the active path
        using the robot pose as the sampling center, then splice the result into
        active_path_points and publish local_plan.
        """
        if (
            self.current_pose is None or
            not self.active_path_points or
            self.local_replan_start_index is None or
            self.local_replan_goal_index is None
        ):
            self.get_logger().warn('REPLAN_LOCAL: missing data for replanning. Returning to FOLLOW_PATH.')
            self.state = 'FOLLOW_PATH'
            return

        # Clamp indices
        start_idx = max(0, min(self.local_replan_start_index, len(self.active_path_points) - 1))
        goal_idx = max(0, min(self.local_replan_goal_index, len(self.active_path_points) - 1))

        if start_idx >= goal_idx:
            self.get_logger().warn(
                f'REPLAN_LOCAL: invalid indices start={start_idx}, goal={goal_idx}. '
                'Returning to FOLLOW_PATH.'
            )
            self.state = 'FOLLOW_PATH'
            return

        # Start: use current robot pose (for realism)
        rx = self.current_pose.pose.pose.position.x
        ry = self.current_pose.pose.pose.position.y
        start_xy = (rx, ry)

        # Goal: reconnect point on the original active path
        goal_xy = self.active_path_points[goal_idx]

        center_xy = (rx, ry)

        self.get_logger().info(
            f'REPLAN_LOCAL: running RRT* from ({start_xy[0]:.2f}, {start_xy[1]:.2f}) '
            f'to ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) '
            f'with local_range={self.rrt_local_range:.2f}.'
        )

        local_path = self.plan_rrt_star(start_xy, goal_xy, center_xy)

        if local_path is None or len(local_path) < 2:
            self.get_logger().warn(
                'REPLAN_LOCAL: RRT* failed or local path too short. '
                'Returning to FOLLOW_PATH without modifying path.'
            )
            self.state = 'FOLLOW_PATH'
            return

        # Build Path message for visualization
        local_path_msg = self.build_path_msg(local_path)
        self.local_path_pub.publish(local_path_msg)

        # Splice local RRT* path into active_path_points
        prefix = self.active_path_points[:start_idx]
        suffix = self.active_path_points[goal_idx + 1:]

        # Avoid duplicating the reconnect goal if the last local point is very close
        if math.hypot(local_path[-1][0] - goal_xy[0], local_path[-1][1] - goal_xy[1]) < 0.1:
            local_splice = local_path
        else:
            local_splice = local_path + [goal_xy]

        new_active = prefix + local_splice + suffix
        self.active_path_points = new_active

        # Reset indices & go back to FOLLOW_PATH
        self.current_path_index = 0
        self.local_replan_start_index = None
        self.local_replan_goal_index = None

        self.get_logger().info(
            f'REPLAN_LOCAL: local detour spliced into active path. '
            f'New active path length = {len(self.active_path_points)}. '
            'Returning to FOLLOW_PATH.'
        )

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
