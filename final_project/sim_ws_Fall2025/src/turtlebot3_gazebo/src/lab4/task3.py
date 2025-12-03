#!/usr/bin/env python3

import os
import math
import random  # For RRT* sampling

import rclpy
from rclpy.node import Node

# ROS message types
from geometry_msgs.msg import (
    Twist,
    PoseWithCovarianceStamped,
    PoseStamped,
    Pose,
    PoseArray,
)
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32

# Map / image handling
import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

from cv_bridge import CvBridge, CvBridgeError


class Task3(Node):
    """
    Task 3: Search and Localize colored balls.

    Layers implemented:

    - Layer 0: Boilerplate & ROS wiring
    - Layer 1: Map loading & occupancy utilities
    - Layer 2: Automatic waypoint generation & route optimization
               using distance-transform maxima (geometry-based coverage).
    - Layer 3: Navigation core
               * A* global planner over inflated map
               * Pure-pursuit-style path follower
               * Patrol state machine visiting DT waypoints in an efficient order
               * Simple reactive obstacle avoidance (LaserScan-based)
               * Local RRT* detours for path blockage avoidance (dynamic obstacles)
    - Layer 4: Vision + ball detection + visualization
               * Subscribe to camera
               * Multi-color (red/green/blue) HSV-based blob detection
               * Draw detections and show OpenCV debug window
    - Layer 5: Ball localization & task bookkeeping
               * Fuse image bearing + LaserScan range + AMCL pose -> world (x,y)
               * Track first good localization per color
               * Publish markers and PoseArray for detected balls
               * Track task start/end time and publish completion time
    """

    # Future improvements (not yet implemented):
    # - Better multi-frame filtering / averaging of ball positions.
    # - More precise camera calibration / TF alignment between camera and base_scan.
    # - Dynamic adjustment of HSV ranges / area thresholds for robustness.

    def __init__(self):
        super().__init__('task3_node')
        self.get_logger().info('[Layer 0] Initializing Task3 node...')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('namespace', '')
        self.declare_parameter('map', 'map')          # logical map name (map.yaml in maps/)
        self.declare_parameter('inflation_kernel', 6) # for static obstacle inflation

        # Layer 2 tuning params
        self.declare_parameter('l2_prune_dist', 0.5)          # m, min dist between waypoints
        self.declare_parameter('l2_two_opt_max_iters', 50)    # iterations for 2-opt
        self.declare_parameter('l2_min_dt_cells', 2)          # min distance-to-obstacle in cells

        # Layer 3 navigation params
        self.declare_parameter('max_linear_vel', 0.35)        # m/s
        self.declare_parameter('max_angular_vel', 1.0)        # rad/s
        self.declare_parameter('lookahead_distance', 0.6)     # m
        self.declare_parameter('goal_tolerance', 0.20)        # m
        self.declare_parameter('angular_gain', 2.0)           # P-gain on heading error
        self.declare_parameter('obstacle_avoid_distance', 0.35)  # m, simple stop threshold

        # Reactive avoidance tuning (simple escape maneuver)
        self.declare_parameter('obstacle_avoid_back_time', 1.0)      # s
        self.declare_parameter('obstacle_avoid_turn_time', 1.0)      # s
        self.declare_parameter('obstacle_avoid_forward_time', 1.0)   # s
        self.declare_parameter('obstacle_avoid_linear_vel', 0.10)    # m/s
        self.declare_parameter('obstacle_avoid_angular_vel', 0.80)   # rad/s

        # Dynamic obstacle + RRT* related
        self.declare_parameter('block_check_lookahead_points', 30)
        self.declare_parameter('dynamic_inflation_kernel', 6)

        # RRT* parameters (borrowed from Task2)
        self.declare_parameter('rrt_max_iterations', 1200)
        self.declare_parameter('rrt_step_size', 0.25)         # meters
        self.declare_parameter('rrt_goal_sample_rate', 0.25)  # probability [0,1]
        self.declare_parameter('rrt_neighbor_radius', 0.9)    # meters
        self.declare_parameter('rrt_local_range', 5.0)        # meters (sampling window radius)
        self.declare_parameter('rrt_clearance_cells', 1)

        # Camera model (approximate)
        self.declare_parameter('camera_h_fov_deg', 60.0)      # assumed horizontal FOV
        self.declare_parameter('ball_min_contour_area', 100)  # px, to filter noise

        self.ns = self.get_parameter('namespace').get_parameter_value().string_value
        if self.ns and not self.ns.startswith('/'):
            self.ns = '/' + self.ns

        self.map_name = self.get_parameter('map').get_parameter_value().string_value
        self.inflation_kernel = (
            self.get_parameter('inflation_kernel').get_parameter_value().integer_value
        )

        self.l2_prune_dist = (
            self.get_parameter('l2_prune_dist')
            .get_parameter_value().double_value
        )
        self.l2_two_opt_max_iters = (
            self.get_parameter('l2_two_opt_max_iters')
            .get_parameter_value().integer_value
        )
        self.l2_min_dt_cells = (
            self.get_parameter('l2_min_dt_cells')
            .get_parameter_value().integer_value
        )

        self.max_linear_vel = (
            self.get_parameter('max_linear_vel').get_parameter_value().double_value
        )
        self.max_angular_vel = (
            self.get_parameter('max_angular_vel').get_parameter_value().double_value
        )
        self.lookahead_distance = (
            self.get_parameter('lookahead_distance').get_parameter_value().double_value
        )
        self.goal_tolerance = (
            self.get_parameter('goal_tolerance').get_parameter_value().double_value
        )
        self.angular_gain = (
            self.get_parameter('angular_gain').get_parameter_value().double_value
        )
        self.obstacle_avoid_distance = (
            self.get_parameter('obstacle_avoid_distance').get_parameter_value().double_value
        )
        # Avoidance params
        self.obstacle_avoid_back_time = (
            self.get_parameter('obstacle_avoid_back_time').get_parameter_value().double_value
        )
        self.obstacle_avoid_turn_time = (
            self.get_parameter('obstacle_avoid_turn_time').get_parameter_value().double_value
        )
        self.obstacle_avoid_forward_time = (
            self.get_parameter('obstacle_avoid_forward_time').get_parameter_value().double_value
        )
        self.obstacle_avoid_linear_vel = (
            self.get_parameter('obstacle_avoid_linear_vel').get_parameter_value().double_value
        )
        self.obstacle_avoid_angular_vel = (
            self.get_parameter('obstacle_avoid_angular_vel').get_parameter_value().double_value
        )

        self.block_check_lookahead_points = (
            self.get_parameter('block_check_lookahead_points')
            .get_parameter_value().integer_value
        )
        self.dynamic_inflation_kernel = (
            self.get_parameter('dynamic_inflation_kernel')
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
        self.rrt_clearance_cells = (
            self.get_parameter('rrt_clearance_cells').get_parameter_value().integer_value
        )

        self.camera_h_fov_deg = (
            self.get_parameter('camera_h_fov_deg').get_parameter_value().double_value
        )
        self.ball_min_contour_area = (
            self.get_parameter('ball_min_contour_area').get_parameter_value().integer_value
        )

        self.get_logger().info(
            f'[Layer 0] Parameters: map={self.map_name}, inflation_kernel={self.inflation_kernel}, '
            f'l2_prune_dist={self.l2_prune_dist}, l2_min_dt_cells={self.l2_min_dt_cells}, '
            f'max_linear_vel={self.max_linear_vel}, max_angular_vel={self.max_angular_vel}, '
            f'rrt_max_iters={self.rrt_max_iterations}, rrt_range={self.rrt_local_range}'
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
        self.dynamic_occupancy = None  # 0 free, 1 occupied (from LaserScan; dynamic)

        # -------------------------
        # Robot state
        # -------------------------
        self.current_pose = None          # PoseWithCovarianceStamped
        self.latest_scan = None           # LaserScan
        self.latest_image = None          # Image

        # For image handling
        self.bridge = CvBridge()
        self.show_debug_image = True     # OpenCV debug frames

        # -------------------------
        # Waypoints & navigation (Layer 2 & 3)
        # -------------------------
        self.patrol_waypoints = []        # list of (x, y) in map frame (ordered)
        self.current_waypoint_idx = None
        self.waypoints_generated = False  # flag to avoid regenerating
        self.visited_waypoints = set()

        # Global path currently being followed
        self.global_path_points = []      # list of (x, y)
        # Active path (global + local detours from RRT*)
        self.active_path_points = []      # list of (x, y)
        self.current_path_index = 0

        # For local replanning (RRT*)
        self.local_replan_start_index = None
        self.local_replan_goal_index = None
        self.rrt_fail_count = 0

        # -------------------------
        # High-level task state machine
        # -------------------------
        self.state = 'WAIT_FOR_POSE'

        # Simple reactive avoidance state
        self.avoidance_phase = None               # 'BACK', 'TURN', 'FORWARD'
        self.avoidance_direction = None           # 'LEFT' or 'RIGHT'
        self.avoidance_phase_start_time = None    # rclpy.time.Time

        # -------------------------
        # Layer 5: Ball bookkeeping & timing
        # -------------------------
        self.ball_colors = ['red', 'green', 'blue']
        self.detected_balls = {c: False for c in self.ball_colors}
        self.ball_pixel_obs = {c: None for c in self.ball_colors}   # (cx, cy)
        self.ball_world_est = {c: None for c in self.ball_colors}   # (wx, wy)
        self.task_start_time = None
        self.task_done_time = None

        # -------------------------
        # Publishers
        # -------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self._ns_topic('cmd_vel'),
            10
        )

        # Path for patrol/room-order visualization
        self.patrol_path_pub = self.create_publisher(
            Path,
            self._ns_topic('patrol_plan'),
            10
        )

        # A* global path
        self.global_path_pub = self.create_publisher(
            Path,
            self._ns_topic('global_plan'),
            10
        )

        # Local path (RRT* detours)
        self.local_path_pub = self.create_publisher(
            Path,
            self._ns_topic('local_plan'),
            10
        )

        # Markers for patrol waypoints
        self.patrol_markers_pub = self.create_publisher(
            MarkerArray,
            self._ns_topic('patrol_waypoints_markers'),
            10
        )

        # Ball markers (Layer 5)
        self.ball_markers_pub = self.create_publisher(
            MarkerArray,
            self._ns_topic('ball_markers'),
            10
        )

        # PoseArray for ball positions (Layer 5)
        self.balls_posearray_pub = self.create_publisher(
            PoseArray,
            self._ns_topic('detected_balls'),
            10
        )

        # Completion time publisher (Layer 5)
        self.task_time_pub = self.create_publisher(
            Float32,
            self._ns_topic('task3_completion_time'),
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

        self.scan_sub = self.create_subscription(
            LaserScan,
            self._ns_topic('scan'),
            self.scan_callback,
            10
        )

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

        self.get_logger().info('Task3 node initialized (Layers 0–5 + RRT* local detours).')

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
    # Map loading utilities (Layer 1)
    # ----------------------------------------------------------------------
    def _load_map_from_yaml(self):
        self.get_logger().info('[Layer 1] Attempting to load map from YAML...')
        try:
            pkg_share = get_package_share_directory('turtlebot3_gazebo')
        except Exception as e:
            self.get_logger().error(
                f'[Layer 1] Failed to get package share directory: {e}'
            )
            return

        map_yaml_path = os.path.join(pkg_share, 'maps', self.map_name + '.yaml')
        if not os.path.exists(map_yaml_path):
            self.get_logger().error(
                f'[Layer 1] Map YAML not found: {map_yaml_path}'
            )
            return

        self.get_logger().info(f'[Layer 1] Loading map YAML: {map_yaml_path}')

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

        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

        if not os.path.exists(image_path):
            self.get_logger().error(f'[Layer 1] Map image not found: {image_path}')
            return

        self.get_logger().info(f'[Layer 1] Loading map image: {image_path}')

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.get_logger().error('[Layer 1] Failed to load map image with cv2.')
            return

        img_normalized = img.astype(np.float32) / 255.0

        occ = np.ones_like(img_normalized, dtype=np.uint8)  # default occupied
        occ[img_normalized > free_thresh] = 0               # free
        occ[img_normalized < occupied_thresh] = 1           # occupied

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

        kernel_size = max(1, int(self.inflation_kernel))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inflated = cv2.dilate(self.static_occupancy, kernel, iterations=1)
        self.inflated_occupancy = inflated

        # Dynamic occupancy: all free initially
        self.dynamic_occupancy = np.zeros_like(self.static_occupancy, dtype=np.uint8)

        free_static_count = int((self.static_occupancy == 0).sum())
        free_inflated_count = int((self.inflated_occupancy == 0).sum())
        self.get_logger().info(
            f'[Layer 1] Free cells (static)   = {free_static_count}\n'
            f'[Layer 1] Free cells (inflated) = {free_inflated_count}'
        )

        self.map_loaded = True
        self.get_logger().info(
            f'[Layer 1] Obstacle inflation done with kernel size {kernel_size}. '
            'Dynamic obstacle layer initialized.'
        )

    # Dynamic obstacle layer update (from LaserScan, like Task2)
    def update_dynamic_occupancy_from_scan(self):
        if not self.map_loaded or self.latest_scan is None or self.current_pose is None:
            return

        self.get_logger().info(
            '[Layer 3] Updating dynamic occupancy from LaserScan...',
            throttle_duration_sec=2.0
        )

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

            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            wx = rx + math.cos(yaw) * lx - math.sin(yaw) * ly
            wy = ry + math.sin(yaw) * lx + math.cos(yaw) * ly

            idx = self.world_to_grid(wx, wy)
            if idx is not None:
                row, col = idx
                self.dynamic_occupancy[row, col] = 1

            angle += scan.angle_increment

        if self.dynamic_inflation_kernel > 1:
            k = max(1, int(self.dynamic_inflation_kernel))
            kernel = np.ones((k, k), np.uint8)
            cv2.dilate(
                self.dynamic_occupancy,
                kernel,
                dst=self.dynamic_occupancy,
                iterations=1
            )

        num_dyn = int(self.dynamic_occupancy.sum())
        self.get_logger().info(
            f'[Layer 3] Dynamic occupancy updated. Occupied cells ≈ {num_dyn}',
            throttle_duration_sec=2.0
        )

    def world_to_grid(self, x: float, y: float):
        if not self.map_loaded:
            self.get_logger().warn('[Layer 1] world_to_grid called but map not loaded.')
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        dx = x - origin_x
        dy = y - origin_y

        col = int(dx / res)
        index_from_bottom = int(dy / res)
        row = (self.map_height - 1) - index_from_bottom

        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            self.get_logger().debug(
                f'[Layer 1] world_to_grid: ({x:.2f},{y:.2f}) -> out of bounds '
                f'(row={row}, col={col}).'
            )
            return None

        return row, col

    def grid_to_world(self, row: int, col: int):
        if not self.map_loaded:
            self.get_logger().warn('[Layer 1] grid_to_world called but map not loaded.')
            return None

        origin_x, origin_y, _ = self.map_origin
        res = self.map_resolution

        x = origin_x + (col + 0.5) * res
        index_from_bottom = self.map_height - row - 1
        y = origin_y + (index_from_bottom + 0.5) * res

        return x, y

    # ----------------------------------------------------------------------
    # Layer 2: waypoint generation & route optimization (DT-based)
    # ----------------------------------------------------------------------
    def _generate_patrol_waypoints_if_needed(self):
        if self.waypoints_generated:
            return
        if not self.map_loaded:
            return
        if self.current_pose is None:
            return

        self.get_logger().info('[Layer 2] Generating patrol waypoints from map...')
        self._generate_patrol_waypoints_from_map()
        self.waypoints_generated = True

        self.get_logger().info(
            f'[Layer 2] Patrol waypoints generated: {len(self.patrol_waypoints)} points.'
        )

        self._publish_waypoints_path(self.patrol_waypoints)
        self._publish_waypoints_markers(self.patrol_waypoints)

    def _compute_waypoint_free_mask(self):
        """
        Free mask for waypoint placement:
        - static_occupancy == 0 (free)
        - AND inflated_occupancy == 0 (safely away from walls)
        """
        if self.static_occupancy is None or self.inflated_occupancy is None:
            self.get_logger().warn('[Layer 2] Occupancy grids not ready.')
            return None

        self.get_logger().info('[Layer 2] Computing waypoint_free_mask...')
        free_static = (self.static_occupancy == 0)
        free_inflated = (self.inflated_occupancy == 0)

        waypoint_free = np.logical_and(free_static, free_inflated).astype(np.uint8)

        count_free_static = int(free_static.sum())
        count_free_inflated = int(free_inflated.sum())
        count_waypoint_free = int(waypoint_free.sum())

        self.get_logger().info(
            f'[Layer 2] waypoint_free_mask stats:\n'
            f'  free_static      cells = {count_free_static}\n'
            f'  free_inflated    cells = {count_free_inflated}\n'
            f'  waypoint_free    cells = {count_waypoint_free}'
        )

        return waypoint_free

    def _generate_patrol_waypoints_from_map(self):
        waypoint_free_mask = self._compute_waypoint_free_mask()
        if waypoint_free_mask is None or waypoint_free_mask.sum() == 0:
            self.get_logger().warn(
                '[Layer 2] waypoint_free_mask has no free cells. '
                'No patrol waypoints will be generated.'
            )
            self.patrol_waypoints = []
            return

        dist = cv2.distanceTransform(waypoint_free_mask, cv2.DIST_L2, 3)
        min_dt = float(dist[waypoint_free_mask == 1].min()) if waypoint_free_mask.sum() > 0 else 0.0
        max_dt = float(dist.max())
        self.get_logger().info(
            f'[Layer 2] DistanceTransform stats (cells): '
            f'min={min_dt:.2f}, max={max_dt:.2f}'
        )

        min_dist_cells = max(1, int(self.l2_min_dt_cells))
        candidate_mask = np.logical_and(
            waypoint_free_mask == 1,
            dist >= float(min_dist_cells)
        )

        self.get_logger().info(
            f'[Layer 2] Candidate cells after DT threshold >= {min_dist_cells}: '
            f'{int(candidate_mask.sum())}'
        )

        kernel = np.ones((3, 3), np.uint8)
        dist_dilated = cv2.dilate(dist, kernel)

        local_max_mask = np.logical_and(
            candidate_mask,
            dist >= dist_dilated - 1e-6
        )

        ys, xs = np.where(local_max_mask)
        self.get_logger().info(
            f'[Layer 2] Distance-transform local maxima count (grid): {len(xs)}'
        )

        if len(xs) == 0:
            self.get_logger().warn(
                '[Layer 2] No local maxima found. Consider lowering l2_min_dt_cells.'
            )

        raw_world_points = []
        for r, c in zip(ys, xs):
            world_xy = self.grid_to_world(int(r), int(c))
            if world_xy is not None:
                raw_world_points.append(world_xy)

        self.get_logger().info(
            f'[Layer 2] Raw world waypoints from DT maxima: {len(raw_world_points)}'
        )

        if raw_world_points:
            xs_w = [p[0] for p in raw_world_points]
            ys_w = [p[1] for p in raw_world_points]
            self.get_logger().info(
                f'[Layer 2] Raw waypoint world bounds: '
                f'x in [{min(xs_w):.2f}, {max(xs_w):.2f}], '
                f'y in [{min(ys_w):.2f}, {max(ys_w):.2f}]'
            )

            idx_br = max(
                range(len(raw_world_points)),
                key=lambda i: (raw_world_points[i][0], -raw_world_points[i][1])
            )
            br_x, br_y = raw_world_points[idx_br]
            self.get_logger().info(
                f'[Layer 2] Bottom-right-ish DT candidate: ({br_x:.2f}, {br_y:.2f})'
            )

            sample_n = min(5, len(raw_world_points))
            msg_pts = ', '.join(
                f'({raw_world_points[i][0]:.2f},{raw_world_points[i][1]:.2f})'
                for i in range(sample_n)
            )
            self.get_logger().info(
                f'[Layer 2] Sample raw DT waypoints (first {sample_n}): {msg_pts}'
            )

        pruned_world_points = self._prune_waypoints_by_distance(
            raw_world_points, self.l2_prune_dist
        )

        self.get_logger().info(
            f'[Layer 2] Pruned world waypoints: {len(pruned_world_points)} '
            f'(prune_dist={self.l2_prune_dist:.2f} m)'
        )

        if not pruned_world_points:
            self.get_logger().warn(
                '[Layer 2] No valid waypoints after pruning. '
                'Patrol waypoints will be empty.'
            )
            self.patrol_waypoints = []
            return

        start_x = self.current_pose.pose.pose.position.x
        start_y = self.current_pose.pose.pose.position.y
        start_xy = (start_x, start_y)

        self.get_logger().info(
            f'[Layer 2] Building route from robot start at ({start_x:.2f}, {start_y:.2f}) '
            f'over {len(pruned_world_points)} waypoints.'
        )

        route_indices = self._nearest_neighbor_route(pruned_world_points, start_xy)
        if len(route_indices) > 2 and self.l2_two_opt_max_iters > 0:
            route_indices = self._two_opt_improvement(
                pruned_world_points, route_indices, self.l2_two_opt_max_iters
            )

        self.patrol_waypoints = [pruned_world_points[i] for i in route_indices]

        total_route_len = 0.0
        cur = start_xy
        for idx in route_indices:
            nx, ny = pruned_world_points[idx]
            total_route_len += math.hypot(nx - cur[0], ny - cur[1])
            cur = (nx, ny)

        self.get_logger().info(
            f'[Layer 2] Final patrol route length ≈ {total_route_len:.2f} m, '
            f'waypoints count={len(self.patrol_waypoints)}.'
        )

        lines = []
        for i, (wx, wy) in enumerate(self.patrol_waypoints):
            lines.append(f'  #{i:02d}: ({wx:.2f}, {wy:.2f})')
        self.get_logger().info(
            '[Layer 2] Final ordered patrol waypoints:\n' + '\n'.join(lines)
        )

    def _prune_waypoints_by_distance(self, points, d_min):
        self.get_logger().info(
            f'[Layer 2] Pruning waypoints by distance: d_min={d_min:.2f}, raw={len(points)}'
        )
        pruned = []
        for (x, y) in points:
            keep = True
            for (px, py) in pruned:
                if math.hypot(x - px, y - py) < d_min:
                    keep = False
                    break
            if keep:
                pruned.append((x, y))

        self.get_logger().info(
            f'[Layer 2] Prune complete. Kept {len(pruned)} waypoints.'
        )
        return pruned

    def _nearest_neighbor_route(self, points, start_xy):
        n = len(points)
        if n == 0:
            self.get_logger().warn('[Layer 2] _nearest_neighbor_route: no points.')
            return []

        self.get_logger().info(f'[Layer 2] Running nearest neighbor route for {n} points...')
        unvisited = set(range(n))
        route = []
        cur_xy = start_xy

        while unvisited:
            best_idx = None
            best_dist = float('inf')
            for i in unvisited:
                x, y = points[i]
                d = math.hypot(x - cur_xy[0], y - cur_xy[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            route.append(best_idx)
            unvisited.remove(best_idx)
            cur_xy = points[best_idx]

        self.get_logger().info(
            f'[Layer 2] Nearest neighbor route built. Route length={len(route)}.'
        )
        return route

    def _two_opt_improvement(self, points, route, max_iters):
        self.get_logger().info(
            f'[Layer 2] Starting 2-opt improvement with max_iters={max_iters}...'
        )

        def route_length(rt):
            length = 0.0
            for i in range(len(rt) - 1):
                x1, y1 = points[rt[i]]
                x2, y2 = points[rt[i + 1]]
                length += math.hypot(x2 - x1, y2 - y1)
            return length

        best_route = list(route)
        best_length = route_length(best_route)

        improved = True
        it = 0

        while improved and it < max_iters:
            improved = False
            it += 1
            for i in range(len(best_route) - 2):
                for j in range(i + 2, len(best_route)):
                    if j == i + 1:
                        continue
                    new_route = best_route[:]
                    new_route[i + 1:j + 1] = reversed(new_route[i + 1:j + 1])
                    new_length = route_length(new_route)
                    if new_length + 1e-6 < best_length:
                        best_route = new_route
                        best_length = new_length
                        improved = True
                        break
                if improved:
                    break

        self.get_logger().info(
            f'[Layer 2] 2-opt finished after {it} iterations. '
            f'Improved route length ≈ {best_length:.2f}'
        )

        return best_route

    def _publish_waypoints_path(self, waypoints):
        if not waypoints:
            self.get_logger().info('[Layer 2] No waypoints to publish as Path.')
            return

        self.get_logger().info(
            f'[Layer 2] Publishing patrol_path Path with {len(waypoints)} poses.'
        )

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for (x, y) in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.patrol_path_pub.publish(path_msg)

    def _publish_waypoints_markers(self, waypoints):
        if not waypoints:
            self.get_logger().info('[Layer 2] No waypoints to publish as markers.')
            return

        self.get_logger().info(
            f'[Layer 2] Publishing {len(waypoints)} patrol_waypoints markers.'
        )

        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, (x, y) in enumerate(waypoints):
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = 'map'
            m.ns = 'patrol_waypoints'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            m.color.a = 1.0
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.lifetime.sec = 0  # forever
            ma.markers.append(m)

        self.patrol_markers_pub.publish(ma)

    # ----------------------------------------------------------------------
    # Layer 3: Navigation core (A* + RRT* + path follower + avoidance)
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

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _is_cell_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free considering both inflated static map
        and dynamic obstacles (like Task2.is_cell_free).
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

    def is_cell_free_rrt(self, row: int, col: int) -> bool:
        k = max(0, int(self.rrt_clearance_cells))
        for r in range(row - k, row + k + 1):
            for c in range(col - k, col + k + 1):
                if not self._is_cell_free(r, c):
                    return False
        return True

    def astar_plan(self, start_world, goal_world):
        if not self.map_loaded:
            self.get_logger().error('[Layer 3] Cannot run A*: map not loaded.')
            return None

        self.get_logger().info(
            f'[Layer 3] A*: planning from {start_world} to {goal_world}...'
        )

        start_idx = self.world_to_grid(start_world[0], start_world[1])
        goal_idx = self.world_to_grid(goal_world[0], goal_world[1])

        if start_idx is None or goal_idx is None:
            self.get_logger().warn('[Layer 3] Start or goal is outside the map.')
            return None

        s_row, s_col = start_idx
        g_row, g_col = goal_idx

        if not self._is_cell_free(s_row, s_col):
            self.get_logger().warn(
                '[Layer 3] Start cell is occupied (static/dynamic); '
                'continuing A* planning anyway.'
            )

        if not self._is_cell_free(g_row, g_col):
            self.get_logger().warn('[Layer 3] Goal cell is not free in inflated/dynamic map.')
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
        import heapq
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
                path = self._reconstruct_path(came_from, (cur_row, cur_col))
                self.get_logger().info(
                    f'[Layer 3] A*: SUCCESS in {iterations} iterations. '
                    f'Path len={len(path)}.'
                )
                return path

            closed_set.add((cur_row, cur_col))

            for dr, dc, move_cost in neighbors:
                nr = cur_row + dr
                nc = cur_col + dc

                if not self._is_cell_free(nr, nc):
                    continue

                neighbor = (nr, nc)
                tentative_g = g_score[(cur_row, cur_col)] + move_cost

                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(nr, nc)
                came_from[neighbor] = (cur_row, cur_col)
                heapq.heappush(open_set, (f_score, neighbor))

        self.get_logger().warn('[Layer 3] A*: failed to find a path within iteration limit.')
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

        self.get_logger().info(f'[Layer 3] A*: path length (cells) = {len(path_world)}')
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
            self.get_logger().debug(
                f'[Layer 3] _is_waypoint_blocked: ({x:.2f},{y:.2f}) outside map -> blocked.'
            )
            return True
        row, col = idx
        blocked = not self._is_cell_free(row, col)
        if blocked:
            self.get_logger().debug(
                f'[Layer 3] _is_waypoint_blocked: cell ({row},{col}) is not free.'
            )
        return blocked

    def _check_path_blocked_ahead(self):
        """
        Check if the active path is blocked within a lookahead window.
        """
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

        blocked_run = False
        first_blocked_idx = None
        last_blocked_idx = None

        for i in range(nearest_idx, max_idx + 1):
            x, y = self.active_path_points[i]
            if self._is_waypoint_blocked(x, y):
                if not blocked_run:
                    blocked_run = True
                    first_blocked_idx = i
                last_blocked_idx = i
            else:
                if blocked_run:
                    break

        if not blocked_run:
            return False, nearest_idx, max_idx

        reconnect_idx = last_blocked_idx + 1

        while reconnect_idx < len(self.active_path_points):
            x, y = self.active_path_points[reconnect_idx]
            if not self._is_waypoint_blocked(x, y):
                break
            reconnect_idx += 1

        reconnect_idx = min(reconnect_idx, len(self.active_path_points) - 1)

        self.get_logger().info(
            f'[Layer 3] Path blocked detected: nearest_idx={nearest_idx}, '
            f'first_blocked_idx={first_blocked_idx}, last_blocked_idx={last_blocked_idx}, '
            f'reconnect_idx={reconnect_idx}'
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
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0.0:
            idx = self.world_to_grid(x1, y1)
            if idx is None:
                return False
            r, c = idx
            return self.is_cell_free_rrt(r, c)

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
            if not self.is_cell_free_rrt(r, c):
                return False

        return True

    def _rrt_sample(self, center_x, center_y, goal_x, goal_y):
        if random.random() < self.rrt_goal_sample_rate:
            return goal_x, goal_y

        for _ in range(100):
            dx = (random.random() * 2.0 - 1.0) * self.rrt_local_range
            dy = (random.random() * 2.0 - 1.0) * self.rrt_local_range
            x = center_x + dx
            y = center_y + dy

            idx = self.world_to_grid(x, y)
            if idx is None:
                continue
            r, c = idx
            if self._is_cell_free(r, c):
                return x, y

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
        neighbors = []
        radius2 = self.rrt_neighbor_radius * self.rrt_neighbor_radius
        for node in nodes:
            dx = node.x - new_node.x
            dy = node.y - new_node.y
            if dx * dx + dy * dy <= radius2:
                neighbors.append(node)
        return neighbors

    def plan_rrt_star(self, start_xy, goal_xy, center_xy):
        sx, sy = start_xy
        gx, gy = goal_xy
        cx, cy = center_xy

        self.get_logger().info(
            f'[Layer 3] RRT*: start={start_xy}, goal={goal_xy}, center={center_xy}, '
            f'max_iter={self.rrt_max_iterations}'
        )

        start_node = self._RRTNode(sx, sy, parent=None, cost=0.0)
        nodes = [start_node]
        goal_node = None

        for it in range(self.rrt_max_iterations):
            if it % 200 == 0 and it > 0:
                self.get_logger().info(
                    f'[Layer 3] RRT*: iteration {it}, nodes={len(nodes)}'
                )

            sample_x, sample_y = self._rrt_sample(cx, cy, gx, gy)

            nearest = self._rrt_nearest(nodes, sample_x, sample_y)
            if nearest is None:
                continue

            new_x, new_y = self._rrt_steer(nearest, sample_x, sample_y)

            if not self._segment_collision_free(nearest.x, nearest.y, new_x, new_y):
                continue

            new_node = self._RRTNode(new_x, new_y, parent=None, cost=float('inf'))

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

            for nb in neighbors:
                d = math.hypot(nb.x - new_node.x, nb.y - new_node.y)
                if new_node.cost + d < nb.cost and self._segment_collision_free(new_node.x, new_node.y, nb.x, nb.y):
                    nb.parent = new_node
                    nb.cost = new_node.cost + d

            dist_to_goal = math.hypot(new_node.x - gx, new_node.y - gy)
            if dist_to_goal <= self.rrt_step_size * 2.0:
                if self._segment_collision_free(new_node.x, new_node.y, gx, gy):
                    goal_node = self._RRTNode(
                        gx, gy,
                        parent=new_node,
                        cost=new_node.cost + dist_to_goal
                    )
                    self.get_logger().info(
                        f'[Layer 3] RRT*: reached goal at iteration {it}, nodes={len(nodes)}.'
                    )
                    break

        if goal_node is None:
            self.get_logger().warn('[Layer 3] RRT*: failed to find a local path.')
            return None

        path = []
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()

        self.get_logger().info(f'[Layer 3] RRT*: local path length = {len(path)} waypoints.')
        return path

    def _obstacle_too_close_ahead(self, max_distance=0.5, fov_deg=40.0):
        if self.latest_scan is None:
            return False

        scan = self.latest_scan
        half_fov = math.radians(fov_deg / 2.0)

        angle = scan.angle_min
        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue
            if abs(angle) <= half_fov and scan.range_min < r < max_distance:
                self.get_logger().info(
                    f'[Layer 3] Obstacle too close ahead: range={r:.2f} m, '
                    f'threshold={max_distance:.2f} m.'
                )
                return True
            angle += scan.angle_increment

        return False

    def _scan_min_in_sector_rad(self, center_rad: float, width_rad: float) -> float:
        if self.latest_scan is None:
            return float('inf')

        scan = self.latest_scan
        half = width_rad / 2.0

        angle = scan.angle_min
        min_r = float('inf')

        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue

            if scan.range_min <= r <= scan.range_max:
                if (center_rad - half) <= angle <= (center_rad + half):
                    if r < min_r:
                        min_r = r

            angle += scan.angle_increment

        return min_r

    def _scan_min_in_sector(self, center_deg: float, width_deg: float) -> float:
        center = math.radians(center_deg)
        width = math.radians(width_deg)
        return self._scan_min_in_sector_rad(center, width)

    def _choose_avoidance_direction(self) -> str:
        left_min = self._scan_min_in_sector(90.0, 80.0)
        right_min = self._scan_min_in_sector(-90.0, 80.0)

        self.get_logger().info(
            f'[Layer 3] AVOID_OBSTACLE: sector min ranges -> '
            f'left={left_min:.2f}, right={right_min:.2f}'
        )

        if left_min >= right_min:
            return 'LEFT'
        else:
            return 'RIGHT'

    def _start_avoidance(self):
        self.avoidance_direction = self._choose_avoidance_direction()
        self.avoidance_phase = 'BACK'
        self.avoidance_phase_start_time = self.get_clock().now()

        self.get_logger().info(
            f'[Layer 3] AVOID_OBSTACLE: starting maneuver, '
            f'direction={self.avoidance_direction}.'
        )

    def _handle_avoid_obstacle(self):
        if self.avoidance_phase is None:
            self._start_avoidance()

        now = self.get_clock().now()
        dt = (now - self.avoidance_phase_start_time).nanoseconds / 1e9

        twist = Twist()

        if self.avoidance_phase == 'BACK':
            if dt < self.obstacle_avoid_back_time:
                twist.linear.x = -self.obstacle_avoid_linear_vel
                self.cmd_vel_pub.publish(twist)
                return
            else:
                self.avoidance_phase = 'TURN'
                self.avoidance_phase_start_time = now
                self.get_logger().info(
                    '[Layer 3] AVOID_OBSTACLE: BACK phase done, switching to TURN.'
                )
                return

        if self.avoidance_phase == 'TURN':
            if dt < self.obstacle_avoid_turn_time:
                if self.avoidance_direction == 'LEFT':
                    twist.angular.z = self.obstacle_avoid_angular_vel
                else:
                    twist.angular.z = -self.obstacle_avoid_angular_vel
                self.cmd_vel_pub.publish(twist)
                return
            else:
                self.avoidance_phase = 'FORWARD'
                self.avoidance_phase_start_time = now
                self.get_logger().info(
                    '[Layer 3] AVOID_OBSTACLE: TURN phase done, switching to FORWARD.'
                )
                return

        if self.avoidance_phase == 'FORWARD':
            if dt < self.obstacle_avoid_forward_time:
                twist.linear.x = self.obstacle_avoid_linear_vel
                self.cmd_vel_pub.publish(twist)
                return
            else:
                self._publish_stop()

                self.avoidance_phase = None
                self.avoidance_direction = None
                self.avoidance_phase_start_time = None

                self.get_logger().info(
                    '[Layer 3] AVOID_OBSTACLE: maneuver complete. '
                    'Switching to PLAN_TO_WAYPOINT for replanning.'
                )

                self.state = 'PLAN_TO_WAYPOINT'
                return

    def _follow_active_path(self):
        if self.current_pose is None or not self.active_path_points:
            self._publish_stop()
            return False

        pose = self.current_pose.pose.pose
        rx = pose.position.x
        ry = pose.position.y
        yaw = self._quat_to_yaw(pose.orientation)

        gx, gy = self.active_path_points[-1]
        dist_to_goal = math.hypot(gx - rx, gy - ry)

        if dist_to_goal < self.goal_tolerance:
            self.get_logger().info(
                f'[Layer 3] Goal reached (distance={dist_to_goal:.3f} < '
                f'{self.goal_tolerance:.3f}).'
            )
            self._publish_stop()
            return True

        nearest_idx, _ = self._nearest_path_index(rx, ry, self.active_path_points)
        if nearest_idx is None:
            self._publish_stop()
            return False

        self.current_path_index = nearest_idx

        target_idx = nearest_idx
        while target_idx < len(self.active_path_points) - 1:
            tx, ty = self.active_path_points[target_idx]
            d = math.hypot(tx - rx, ty - ry)
            if d >= self.lookahead_distance:
                break
            target_idx += 1

        if target_idx >= len(self.active_path_points):
            target_idx = len(self.active_path_points) - 1

        tx, ty = self.active_path_points[target_idx]
        dx = tx - rx
        dy = ty - ry
        dist_to_target = math.hypot(dx, dy)

        if dist_to_target < 1e-6:
            self._publish_stop()
            return False

        desired_yaw = math.atan2(dy, dx)
        heading_error = self._normalize_angle(desired_yaw - yaw)

        heading_factor = max(0.0, 1.0 - abs(heading_error) / math.pi)
        linear_speed = self.max_linear_vel * heading_factor

        if dist_to_goal < 0.5:
            linear_speed *= dist_to_goal / 0.5

        angular_speed = self.angular_gain * heading_error
        if angular_speed > self.max_angular_vel:
            angular_speed = self.max_angular_vel
        elif angular_speed < -self.max_angular_vel:
            angular_speed = -self.max_angular_vel

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f'[Layer 3] FOLLOW_PATH: idx={self.current_path_index}->{target_idx}, '
            f'd_target={dist_to_target:.2f}, d_goal={dist_to_goal:.2f}, '
            f'v={linear_speed:.2f}, w={angular_speed:.2f}',
            throttle_duration_sec=0.5
        )

        return False

    def _choose_next_waypoint_index(self):
        if not self.patrol_waypoints or self.current_pose is None:
            return None

        rx = self.current_pose.pose.pose.position.x
        ry = self.current_pose.pose.pose.position.y

        best_idx = None
        best_dist = float('inf')

        for i, (wx, wy) in enumerate(self.patrol_waypoints):
            if i in self.visited_waypoints:
                continue
            d = math.hypot(wx - rx, wy - ry)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx is not None:
            self.get_logger().info(
                f'[Layer 3] _choose_next_waypoint_index: chose #{best_idx} '
                f'at dist={best_dist:.2f} m.'
            )
        else:
            self.get_logger().info('[Layer 3] _choose_next_waypoint_index: no unvisited waypoints.')

        return best_idx

    # ----------------------------------------------------------------------
    # Layer 4: Vision / ball detection
    # ----------------------------------------------------------------------
    def _image_to_cv2(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            return img
        except CvBridgeError as e:
            self.get_logger().warn(f'[Layer 4] CvBridge error: {e}')
            return None

    def _detect_balls_in_image(self, frame):
        if frame is None:
            return

        self.get_logger().info(
            '[Layer 4] Running ball detection on new camera frame...',
            throttle_duration_sec=0.5
        )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w, _ = frame.shape

        color_ranges = {
            'red': [
                ((0, 100, 80), (10, 255, 255)),
                ((160, 100, 80), (179, 255, 255)),
            ],
            'green': [
                ((35, 80, 80), (85, 255, 255)),
            ],
            'blue': [
                ((90, 80, 80), (135, 255, 255)),
            ],
        }

        debug_frame = frame.copy()

        for color in self.ball_colors:
            if color not in color_ranges:
                continue

            masks = []
            for (lower, upper) in color_ranges[color]:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                masks.append(cv2.inRange(hsv, lower_np, upper_np))

            mask = masks[0]
            for extra in masks[1:]:
                mask = cv2.bitwise_or(mask, extra)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.get_logger().debug(f'[Layer 4] No contours found for color {color}.')
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            self.get_logger().debug(
                f'[Layer 4] Color {color}: largest contour area={area:.1f} px.'
            )
            if area < self.ball_min_contour_area:
                self.get_logger().debug(
                    f'[Layer 4] Color {color}: contour area {area:.1f} < '
                    f'min_area={self.ball_min_contour_area}. Ignoring.'
                )
                continue

            (x, y, rw, rh) = cv2.boundingRect(largest)
            cx = int(x + rw / 2)
            cy = int(y + rh / 2)

            color_bgr = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
            }[color]

            cv2.rectangle(debug_frame, (x, y), (x + rw, y + rh), color_bgr, 2)
            cv2.circle(debug_frame, (cx, cy), 5, color_bgr, -1)
            cv2.putText(
                debug_frame,
                color,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                1,
                cv2.LINE_AA,
            )

            self.get_logger().info(
                f'[Layer 4] Detected {color} blob: bbox=({x},{y},{rw},{rh}), '
                f'centroid=({cx},{cy}).'
            )

            self.ball_pixel_obs[color] = (cx, cy)

            world_xy = self._localize_ball_from_pixel(color, cx, cy, w, h)
            if world_xy is not None:
                wx, wy = world_xy
                self.ball_world_est[color] = (wx, wy)
                self.detected_balls[color] = True
                self.get_logger().info(
                    f'[Layer 5] Localized {color} ball at world ≈ ({wx:.2f}, {wy:.2f}).'
                )
            else:
                self.get_logger().info(
                    f'[Layer 5] Failed to localize {color} ball from pixel observation.'
                )

        if self.show_debug_image:
            cv2.imshow('task3_camera_debug', debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.show_debug_image = False
                cv2.destroyWindow('task3_camera_debug')
                self.get_logger().info('[Layer 4] Disabled OpenCV debug window via keyboard.')

    def _localize_ball_from_pixel(self, color, cx, cy, img_width, img_height):
        if self.latest_scan is None or self.current_pose is None:
            self.get_logger().debug(
                f'[Layer 5] _localize_ball_from_pixel: no scan/pose yet for color={color}.'
            )
            return None

        img_cx = img_width / 2.0
        fov_rad = math.radians(self.camera_h_fov_deg)

        pixel_offset = (cx - img_cx) / img_cx
        bearing_cam = pixel_offset * (fov_rad / 2.0)

        self.get_logger().debug(
            f'[Layer 5] Color={color}, pixel_x={cx}, img_cx={img_cx:.1f}, '
            f'bearing_cam={bearing_cam:.3f} rad.'
        )

        range_window_rad = math.radians(10.0)
        r = self._scan_min_in_sector_rad(bearing_cam, range_window_rad)
        if math.isinf(r) or r <= 0.0:
            self.get_logger().info(
                f'[Layer 5] No valid scan range for {color} ball (bearing={bearing_cam:.2f} rad).'
            )
            return None

        lx = r * math.cos(bearing_cam)
        ly = r * math.sin(bearing_cam)

        pose = self.current_pose.pose.pose
        rx = pose.position.x
        ry = pose.position.y
        yaw = self._quat_to_yaw(pose.orientation)

        wx = rx + math.cos(yaw) * lx - math.sin(yaw) * ly
        wy = ry + math.sin(yaw) * lx + math.cos(yaw) * ly

        self.get_logger().debug(
            f'[Layer 5] Color={color}: range={r:.2f} m, robot=({rx:.2f},{ry:.2f}), '
            f'yaw={yaw:.2f} rad -> world=({wx:.2f},{wy:.2f}).'
        )

        idx = self.world_to_grid(wx, wy)
        if idx is None:
            self.get_logger().info(
                f'[Layer 5] Localized {color} ball outside map bounds, '
                f'candidate=({wx:.2f}, {wy:.2f}). Ignoring.'
            )
            return None

        row, col = idx
        if not self._is_cell_free(row, col):
            self.get_logger().info(
                f'[Layer 5] Localized {color} ball in occupied cell ({row},{col}), '
                f'candidate=({wx:.2f}, {wy:.2f}). Using but note proximity to obstacle.'
            )

        return wx, wy

    # ----------------------------------------------------------------------
    # Layer 5: Ball markers & completion
    # ----------------------------------------------------------------------
    def _publish_ball_markers(self):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        color_map_rgb = {
            'red':   (1.0, 0.0, 0.0),
            'green': (0.0, 1.0, 0.0),
            'blue':  (0.0, 0.0, 1.0),
        }

        mid = 0
        for color in self.ball_colors:
            if not self.detected_balls[color]:
                continue
            if self.ball_world_est[color] is None:
                continue

            wx, wy = self.ball_world_est[color]
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = 'map'
            m.ns = 'balls'
            m.id = mid
            mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.10
            m.pose.orientation.w = 1.0
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.color.a = 1.0
            r, g, b = color_map_rgb[color]
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.lifetime.sec = 0
            ma.markers.append(m)

        if ma.markers:
            self.get_logger().info(
                f'[Layer 5] Publishing {len(ma.markers)} ball markers.'
            )
            self.ball_markers_pub.publish(ma)
        else:
            self.get_logger().debug('[Layer 5] No ball markers to publish.')

    def _publish_balls_posearray_and_time(self):
        if not all(self.detected_balls.values()):
            return

        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'

        for color in self.ball_colors:
            wx, wy = self.ball_world_est.get(color, (None, None))
            if wx is None or wy is None:
                self.get_logger().warn(
                    f'[Layer 5] Ball {color} marked detected but no world_est. Skipping in PoseArray.'
                )
                continue
            p = Pose()
            p.position.x = wx
            p.position.y = wy
            p.position.z = 0.0
            p.orientation.w = 1.0
            pa.poses.append(p)

        self.get_logger().info(
            f'[Layer 5] Publishing PoseArray with {len(pa.poses)} ball poses.'
        )
        self.balls_posearray_pub.publish(pa)

        if self.task_start_time is not None:
            now = self.get_clock().now()
            dt = now - self.task_start_time
            elapsed_seconds = dt.nanoseconds / 1e9
            msg = Float32()
            msg.data = float(elapsed_seconds)
            self.task_time_pub.publish(msg)
            self.get_logger().info(
                f'[Layer 5] TASK COMPLETE: all balls localized. '
                f'Elapsed time ≈ {elapsed_seconds:.2f} s '
                f'(published on task3_completion_time).'
            )
        else:
            self.get_logger().warn(
                '[Layer 5] TASK COMPLETE: all balls localized, but task_start_time was None.'
            )

    def _maybe_finish_task_if_balls_found(self):
        if not all(self.detected_balls.values()):
            return

        if self.state != 'TASK_DONE':
            self.get_logger().info('[Layer 5] All balls detected. Finalizing task.')
            self._publish_ball_markers()
            self._publish_balls_posearray_and_time()
            self.task_done_time = self.get_clock().now()
            self.state = 'TASK_DONE'
            self._publish_stop()

    # ----------------------------------------------------------------------
    # Subscriber callbacks
    # ----------------------------------------------------------------------
    def amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg
        if self.state == 'WAIT_FOR_POSE' and self.map_loaded:
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            self.get_logger().info(
                f'[Task3] AMCL pose received. '
                f'Current robot pose ≈ ({px:.2f}, {py:.2f}) in map frame.'
            )

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.get_logger().debug(
            '[Layer 3] New LaserScan received.',
        )
        self.update_dynamic_occupancy_from_scan()

    def image_callback(self, msg: Image):
        self.latest_image = msg
        self.get_logger().debug(
            '[Layer 4] New camera image received.',
        )
        frame = self._image_to_cv2(msg)
        if frame is not None:
            self._detect_balls_in_image(frame)
            self._publish_ball_markers()

    # ----------------------------------------------------------------------
    # Local replanning handler (RRT* detour)
    # ----------------------------------------------------------------------
    def _handle_replan_local(self):
        self.get_logger().info('[Layer 3] Handling REPLAN_LOCAL (RRT* detour)...')

        if (
            self.current_pose is None or
            not self.active_path_points or
            self.local_replan_start_index is None or
            self.local_replan_goal_index is None
        ):
            self.get_logger().warn(
                '[Layer 3] REPLAN_LOCAL: missing data for replanning. '
                'Returning to FOLLOW_PATH.'
            )
            self.state = 'FOLLOW_PATH'
            return

        start_idx = max(0, min(self.local_replan_start_index, len(self.active_path_points) - 1))
        goal_idx = max(0, min(self.local_replan_goal_index, len(self.active_path_points) - 1))

        if start_idx >= goal_idx:
            self.get_logger().warn(
                f'[Layer 3] REPLAN_LOCAL: invalid indices start={start_idx}, goal={goal_idx}. '
                'Returning to FOLLOW_PATH.'
            )
            self.state = 'FOLLOW_PATH'
            return

        rx = self.current_pose.pose.pose.position.x
        ry = self.current_pose.pose.pose.position.y
        start_xy = (rx, ry)

        goal_xy = self.active_path_points[goal_idx]
        center_xy = (rx, ry)

        start_idx_cell = self.world_to_grid(start_xy[0], start_xy[1])
        goal_idx_cell = self.world_to_grid(goal_xy[0], goal_xy[1])
        if start_idx_cell is not None:
            sr, sc = start_idx_cell
            self.get_logger().info(
                f'[Layer 3] REPLAN_LOCAL: start cell ({sr},{sc}) free={self._is_cell_free(sr, sc)}'
            )
        else:
            self.get_logger().info('[Layer 3] REPLAN_LOCAL: start cell is outside map.')

        if goal_idx_cell is not None:
            gr, gc = goal_idx_cell
            self.get_logger().info(
                f'[Layer 3] REPLAN_LOCAL: goal cell ({gr},{gc}) free={self._is_cell_free(gr, gc)}'
            )
        else:
            self.get_logger().info('[Layer 3] REPLAN_LOCAL: goal cell is outside map.')

        self.get_logger().info(
            f'[Layer 3] REPLAN_LOCAL: running RRT* from ({start_xy[0]:.2f}, {start_xy[1]:.2f}) '
            f'to ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) '
            f'with local_range={self.rrt_local_range:.2f}.'
        )

        local_path = self.plan_rrt_star(start_xy, goal_xy, center_xy)

        if local_path is None or len(local_path) < 2:
            self.rrt_fail_count += 1
            self.get_logger().warn(
                f'[Layer 3] REPLAN_LOCAL: RRT* failed or local path too short '
                f'(fail_count={self.rrt_fail_count}). '
                'Falling back to PLAN_TO_WAYPOINT.'
            )
            self.state = 'PLAN_TO_WAYPOINT'
            return

        self.rrt_fail_count = 0

        self.get_logger().info(
            f'[Layer 3] REPLAN_LOCAL: RRT* produced path with {len(local_path)} points.'
        )
        local_path_msg = self.build_path_msg(local_path)
        self.local_path_pub.publish(local_path_msg)

        prefix = self.active_path_points[:start_idx]
        suffix = self.active_path_points[goal_idx + 1:]

        if math.hypot(local_path[-1][0] - goal_xy[0], local_path[-1][1] - goal_xy[1]) < 0.1:
            local_splice = local_path
        else:
            local_splice = local_path + [goal_xy]

        new_active = prefix + local_splice + suffix
        self.active_path_points = new_active

        self.current_path_index = 0
        self.local_replan_start_index = None
        self.local_replan_goal_index = None

        self.get_logger().info(
            f'[Layer 3] REPLAN_LOCAL: local detour spliced into active path. '
            f'New active path length = {len(self.active_path_points)}. '
            'Returning to FOLLOW_PATH.'
        )

        self.state = 'FOLLOW_PATH'

    # ----------------------------------------------------------------------
    # Timer callback / state machine
    # ----------------------------------------------------------------------
    def timer_cb(self):
        self.get_logger().info(
            f'[Layer 0-5] State={self.state}, '
            f'map_loaded={self.map_loaded}, '
            f'pose={self.current_pose is not None}, '
            f'scan={self.latest_scan is not None}, '
            f'image={self.latest_image is not None}, '
            f'waypoints_generated={self.waypoints_generated}, '
            f'visited={len(self.visited_waypoints)}/{len(self.patrol_waypoints)}, '
            f'balls_detected={sum(self.detected_balls.values())}/3',
            throttle_duration_sec=1.0
        )

        if not self.map_loaded or self.current_pose is None:
            self._publish_stop()
            return

        self._generate_patrol_waypoints_if_needed()

        if not self.patrol_waypoints:
            self.get_logger().warn('[Layer 3] No patrol_waypoints available. Stopping.')
            self._publish_stop()
            return

        self._maybe_finish_task_if_balls_found()
        if self.state == 'TASK_DONE':
            return

        if self.state == 'WAIT_FOR_POSE':
            self.state = 'SELECT_NEXT_WAYPOINT'
            if self.task_start_time is None:
                self.task_start_time = self.get_clock().now()
                self.get_logger().info('[Layer 5] Task timer started.')
            self.get_logger().info('[Layer 3] Transition: WAIT_FOR_POSE -> SELECT_NEXT_WAYPOINT')
            return

        if self.state == 'SELECT_NEXT_WAYPOINT':
            idx = self._choose_next_waypoint_index()
            if idx is None:
                self.state = 'PATROL_DONE'
                self._publish_stop()
                self.get_logger().info('[Layer 3] All waypoints visited. PATROL_DONE.')
                self._maybe_finish_task_if_balls_found()
                return

            self.current_waypoint_idx = idx
            wx, wy = self.patrol_waypoints[idx]
            self.get_logger().info(
                f'[Layer 3] Selected next waypoint #{idx}: ({wx:.2f}, {wy:.2f})'
            )
            self.state = 'PLAN_TO_WAYPOINT'
            return

        if self.state == 'PLAN_TO_WAYPOINT':
            wx, wy = self.patrol_waypoints[self.current_waypoint_idx]
            rx = self.current_pose.pose.pose.position.x
            ry = self.current_pose.pose.pose.position.y

            self.get_logger().info(
                f'[Layer 3] PLAN_TO_WAYPOINT: A* from ({rx:.2f}, {ry:.2f}) '
                f'to waypoint #{self.current_waypoint_idx} = ({wx:.2f}, {wy:.2f})'
            )

            path_world = self.astar_plan((rx, ry), (wx, wy))

            if path_world is None or len(path_world) < 2:
                self.get_logger().warn(
                    f'[Layer 3] A* failed or path too short to waypoint #{self.current_waypoint_idx}. '
                    'Marking as visited and selecting another.'
                )
                self.visited_waypoints.add(self.current_waypoint_idx)
                self.state = 'SELECT_NEXT_WAYPOINT'
                return

            self.global_path_points = path_world
            self.active_path_points = list(path_world)
            self.current_path_index = 0

            path_msg = self.build_path_msg(path_world)
            self.global_path_pub.publish(path_msg)

            self.get_logger().info(
                f'[Layer 3] A* path computed to waypoint #{self.current_waypoint_idx} '
                f'(len={len(path_world)}). Switching to FOLLOW_PATH.'
            )
            self.state = 'FOLLOW_PATH'
            return

        if self.state == 'FOLLOW_PATH':
            if self._obstacle_too_close_ahead(max_distance=self.obstacle_avoid_distance):
                self.get_logger().warn(
                    '[Layer 3] FOLLOW_PATH: obstacle too close ahead. Entering AVOID_OBSTACLE.'
                )
                self._start_avoidance()
                self.state = 'AVOID_OBSTACLE'
                return

            blocked, start_idx, goal_idx = self._check_path_blocked_ahead()
            if blocked:
                self.local_replan_start_index = start_idx
                self.local_replan_goal_index = goal_idx
                self.get_logger().info(
                    f'[Layer 3] FOLLOW_PATH: path blocked. Triggering REPLAN_LOCAL '
                    f'from index {start_idx} to {goal_idx}.'
                )
                self._publish_stop()
                self.state = 'REPLAN_LOCAL'
                return

            reached = self._follow_active_path()
            if reached:
                self.visited_waypoints.add(self.current_waypoint_idx)
                self.get_logger().info(
                    f'[Layer 3] Waypoint #{self.current_waypoint_idx} reached. '
                    f'Visited {len(self.visited_waypoints)}/{len(self.patrol_waypoints)}.'
                )
                self.state = 'SELECT_NEXT_WAYPOINT'
            return

        if self.state == 'REPLAN_LOCAL':
            self._handle_replan_local()
            return

        if self.state == 'AVOID_OBSTACLE':
            self._handle_avoid_obstacle()
            return

        if self.state == 'PATROL_DONE':
            self._publish_stop()
            self._maybe_finish_task_if_balls_found()
            return

        if self.state == 'TASK_DONE':
            self._publish_stop()
            return

    # ----------------------------------------------------------------------
    # Simple helper to stop the robot
    # ----------------------------------------------------------------------
    def _publish_stop(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.get_logger().info('[Layer 0] Shutting down Task3 node and closing OpenCV windows.')
        super().destroy_node()


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
