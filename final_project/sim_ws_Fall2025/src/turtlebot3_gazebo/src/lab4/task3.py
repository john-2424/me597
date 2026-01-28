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
    Point,
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
        self.declare_parameter('inflation_kernel', 9) # for static obstacle inflation

        # Layer 2 tuning params
        self.declare_parameter('l2_prune_dist', 3.6)          # m, min dist between waypoints
        self.declare_parameter('l2_two_opt_max_iters', 50)    # iterations for 2-opt
        self.declare_parameter('l2_min_dt_cells', 6)          # min distance-to-obstacle in cells

        # Layer 3 navigation params
        self.declare_parameter('max_linear_vel', 0.30)        # m/s
        self.declare_parameter('max_angular_vel', 0.9)        # rad/s
        self.declare_parameter('lookahead_distance', 0.35)     # m
        self.declare_parameter('goal_tolerance', 0.15)        # m
        self.declare_parameter('angular_gain', 2.0)           # P-gain on heading error
        self.declare_parameter('obstacle_avoid_distance', 0.45)  # m, simple stop threshold

        # when heading error is large, rotate-only instead of "plowing forward"
        self.declare_parameter('heading_align_only_threshold', 0.5)  # rad (~30 deg)

        # slow down when something is near but not yet at avoid distance
        self.declare_parameter('obstacle_slowdown_distance', 0.70)   # m

        # Align-to-ball (rotate-only) params 
        self.declare_parameter('align_center_tolerance', 0.1)  # normalized [0..1]
        self.declare_parameter('align_angular_gain', 1.0)       # rad/s per unit error
        # max angular speed *specifically* for ALIGN_TO_BALL (gentler)
        self.declare_parameter('align_max_angular_vel', 0.25)    # rad/s
        self.declare_parameter('align_timeout_sec', 8.0)        # s

        # PID gains for heading alignment (image-center based)
        self.declare_parameter('align_kp', 0.5)
        self.declare_parameter('align_ki', 0.0)
        self.declare_parameter('align_kd', 0.1)
        # Integral windup clamp (in "error * seconds" units)
        self.declare_parameter('align_i_max', 0.15)

        # deadband and stability requirement for alignment
        self.declare_parameter('align_deadband', 0.05)          # norm error below this → treat as 0
        self.declare_parameter('align_center_stable_frames', 4) # consecutive frames centered
        # extra settle time with v=0, w=0 before LiDAR measurement
        self.declare_parameter('align_settle_time_sec', 5.0)  # 0.3–0.5s is usually enough

        # scan-at-waypoint params (full 360° spin)
        self.declare_parameter('scan_angular_vel', 0.9)         # rad/s
        self.declare_parameter('scan_turns', 1.0)               # number of full rotations
        self.declare_parameter('scan_timeout_sec', 30.0)        # safety timeout

        # Reactive avoidance tuning (simple escape maneuver)
        self.declare_parameter('obstacle_avoid_back_time', 1.0)      # s
        self.declare_parameter('obstacle_avoid_turn_time', 1.0)      # s
        self.declare_parameter('obstacle_avoid_forward_time', 1.0)   # s
        self.declare_parameter('obstacle_avoid_linear_vel', 0.10)    # m/s
        self.declare_parameter('obstacle_avoid_angular_vel', 0.80)   # rad/s

        # Dynamic obstacle + RRT* related
        self.declare_parameter('block_check_lookahead_points', 30)
        self.declare_parameter('dynamic_inflation_kernel', 5)

        # RRT* parameters (borrowed from Task2)
        self.declare_parameter('rrt_max_iterations', 1600)
        self.declare_parameter('rrt_step_size', 0.20)         # meters
        self.declare_parameter('rrt_goal_sample_rate', 0.25)  # probability [0,1]
        self.declare_parameter('rrt_neighbor_radius', 0.5)    # meters
        self.declare_parameter('rrt_local_range', 5.0)        # meters (sampling window radius)
        self.declare_parameter('rrt_clearance_cells', 1)
        # how many consecutive RRT* failures before we give up and unstick via AVOID_OBSTACLE
        self.declare_parameter('rrt_fail_limit', 3)

        # Camera model (approximate)
        self.declare_parameter('camera_h_fov_deg', 60.0)      # assumed horizontal FOV
        # Optional camera intrinsics (will override HFOV model if non-zero)
        self.declare_parameter('camera_fx', 0.0)              # focal length in pixels
        self.declare_parameter('camera_cx', 0.0)              # principal point x in pixels

        # 👇 much larger default: balls should be mid-sized blobs, not specks
        self.declare_parameter('ball_min_contour_area', 800.0)  # px, to filter noise

        self.declare_parameter('ball_min_circularity', 0.50)  # shape filter (0–1)

        # Known physical ball size (for LiDAR + perspective gating)
        self.declare_parameter('ball_diameter_m', 0.18)       # adjust to your actual ball

        # Yaw offset between camera optical axis and LaserScan 0-angle (in degrees).
        # Positive = camera is rotated CCW relative to the scan.
        self.declare_parameter('camera_laser_yaw_offset_deg', 0.0)

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

        self.heading_align_only_threshold = (
            self.get_parameter('heading_align_only_threshold')
            .get_parameter_value().double_value
        )
        self.obstacle_slowdown_distance = (
            self.get_parameter('obstacle_slowdown_distance')
            .get_parameter_value().double_value
        )

        self.align_center_tolerance = (
            self.get_parameter('align_center_tolerance')
            .get_parameter_value().double_value
        )
        self.align_angular_gain = (
            self.get_parameter('align_angular_gain')
            .get_parameter_value().double_value
        )
        self.align_timeout_sec = (
            self.get_parameter('align_timeout_sec')
            .get_parameter_value().double_value
        )
        self.align_max_angular_vel = (
            self.get_parameter('align_max_angular_vel')
            .get_parameter_value().double_value
        )
        self.align_kp = (
            self.get_parameter('align_kp')
            .get_parameter_value().double_value
        )
        self.align_ki = (
            self.get_parameter('align_ki')
            .get_parameter_value().double_value
        )
        self.align_kd = (
            self.get_parameter('align_kd')
            .get_parameter_value().double_value
        )
        self.align_i_max = (
            self.get_parameter('align_i_max')
            .get_parameter_value().double_value
        )

        # NEW
        self.align_deadband = (
            self.get_parameter('align_deadband')
            .get_parameter_value().double_value
        )
        self.align_center_stable_frames = (
            self.get_parameter('align_center_stable_frames')
            .get_parameter_value().integer_value
        )
        # settle time
        self.align_settle_time_sec = (
            self.get_parameter('align_settle_time_sec')
            .get_parameter_value().double_value
        )
        
        # scan-at-waypoint
        self.scan_angular_vel = (
            self.get_parameter('scan_angular_vel')
            .get_parameter_value().double_value
        )
        self.scan_turns = (
            self.get_parameter('scan_turns')
            .get_parameter_value().double_value
        )
        self.scan_timeout_sec = (
            self.get_parameter('scan_timeout_sec')
            .get_parameter_value().double_value
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
        self.rrt_fail_limit = (
            self.get_parameter('rrt_fail_limit').get_parameter_value().integer_value
        )

        self.camera_h_fov_deg = (
            self.get_parameter('camera_h_fov_deg').get_parameter_value().double_value
        )
        self.camera_fx = (
            self.get_parameter('camera_fx').get_parameter_value().double_value
        )
        self.camera_cx = (
            self.get_parameter('camera_cx').get_parameter_value().double_value
        )
        # Treat this as float so we can do fractional thresholds
        self.ball_min_contour_area = (
            self.get_parameter('ball_min_contour_area').get_parameter_value().double_value
        )
        self.ball_min_circularity = (
            self.get_parameter('ball_min_circularity').get_parameter_value().double_value
        )
        self.ball_diameter_m = (
            self.get_parameter('ball_diameter_m').get_parameter_value().double_value
        )
        self.camera_laser_yaw_offset_deg = (
            self.get_parameter('camera_laser_yaw_offset_deg').get_parameter_value().double_value
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
        # track waypoints that A* failed to reach
        self.unreachable_waypoints = set()

        # Global path currently being followed
        self.global_path_points = []      # list of (x, y)
        # Active path (global + local detours from RRT*)
        self.active_path_points = []      # list of (x, y)
        self.current_path_index = 0

        # For local replanning (RRT*)
        self.local_replan_start_index = None
        self.local_replan_goal_index = None
        self.rrt_fail_count = 0

        # track whether a local (RRT*) path is currently being followed
        self.local_plan_active = False

        # --- Replan cooldown / commit-to-motion ---
        self.replan_cooldown_s = 0.8          # seconds
        self.replan_cooldown_until = 0.0
        self.emergency_stop_dist = 0.25       # meters; always stop/escape if closer than this

        # -------------------------
        # High-level task state machine
        # -------------------------
        self.state = 'WAIT_FOR_POSE'

        # Scan-at-waypoint bookkeeping
        self.scan_start_time = None
        self.scan_start_yaw = None
        self.scan_last_yaw = None
        self.scan_accum_angle = 0.0
        # task completion should wait until scan finishes
        self.pending_task_done = False

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

        # Best (minimum) LiDAR distance seen so far for each color.
        # We only update world pose if a new measurement is closer than this.
        self.ball_range_est = {c: None for c in self.ball_colors}

        # Small margin so we only treat "meaningfully closer" as an improvement
        self.relocalize_range_margin = 0.05  # meters

        """# Per-color HSV ranges (BGR->HSV) as the *primary* color gate.
        # You should tune these ranges using the notebook slider on real
        # Gazebo screenshots. These are just reasonable starting points.
        self.BALL_HSV_RANGES = {
            "red": [
                # low-H red lobe
                (np.array([  0, 150,  80], dtype=np.uint8),
                 np.array([ 10, 255, 255], dtype=np.uint8)),
                # high-H red lobe
                (np.array([170, 150,  80], dtype=np.uint8),
                 np.array([180, 255, 255], dtype=np.uint8)),
            ],
            "green": [
                (np.array([ 40, 120,  60], dtype=np.uint8),
                 np.array([ 85, 255, 255], dtype=np.uint8)),
            ],
            "blue": [
                (np.array([100, 120,  60], dtype=np.uint8),
                 np.array([135, 255, 255], dtype=np.uint8)),
            ],
        }"""
        # Per-color HSV ranges (tuned from Gazebo ball crops).
        # OpenCV HSV: H in [0,179], S,V in [0,255].
        #
        # These are deliberately *tight* on hue and fairly strict on saturation,
        # because the balls are highly saturated and the walls/floor are not.
        self.BALL_HSV_RANGES = {
            "blue": [
                # blue ball: strong cluster around H ≈ 120
                (
                    np.array([110, 150,  60], dtype=np.uint8),
                    np.array([130, 255, 255], dtype=np.uint8),
                ),
            ],
            "green": [
                # green ball: H roughly 40–60 in the central region
                (
                    np.array([ 40, 150,  60], dtype=np.uint8),
                    np.array([ 80, 255, 255], dtype=np.uint8),
                ),
            ],
            "red": [
                # red ball: classic two-lobe red in OpenCV HSV
                (
                    np.array([  0, 150,  80], dtype=np.uint8),
                    np.array([ 10, 255, 255], dtype=np.uint8),
                ),
                (
                    np.array([170, 150,  80], dtype=np.uint8),
                    np.array([180, 255, 255], dtype=np.uint8),
                ),
            ],
        }

        # Align-to-ball state (rotate-only)
        self.align_ball_color = None         # which color we're currently aligning to
        self.align_start_time = None         # rclpy.time.Time
        self.prev_state_before_align = None  # to resume navigation after alignment

        # For alignment math
        self.last_image_width = None         # updated every frame in _detect_balls_in_image

        # filtered error + stability counter
        self.align_filtered_err = 0.0
        self.align_center_stable_counter = 0

        # when we started the "settle with zero cmd_vel" phase
        self.align_settle_start_time = None

        # PID internal state for ALIGN_MEASURE (heading control)
        self.align_pid_i = 0.0
        self.align_pid_prev_err = 0.0
        self.align_pid_prev_time = None

        # once True, we are in the "centered, settling" phase
        self.align_center_latched = False

        # Quality tracking for "best" detection per color
        # Higher score = better ball candidate for that color.
        # Start at -inf so the first valid detection always wins.
        self.ball_detection_score = {
            c: float('-inf') for c in self.ball_colors
        }
        # Optional metadata for debugging / introspection
        # e.g. {'area': ..., 'circ': ..., 'center_offset': ..., 'has_world': ...}
        self.ball_detection_meta = {c: None for c in self.ball_colors}

        # How many consecutive frames a color must be seen before we "trust" it
        # and allow it to update ball_pixel_obs / ball_world_est.
        # This is key for stable, non-flickery detections.
        self.min_stable_frames_for_commit = 4

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

        # --------------------------------------------------
        # geometry_msgs/Point publishers for each color
        # These are intentionally NOT namespace-prefixed,
        # because the spec requires exact topic names.
        # --------------------------------------------------
        self.red_pos_pub = self.create_publisher(Point, '/red_pos', 10)
        self.green_pos_pub = self.create_publisher(Point, '/green_pos', 10)
        self.blue_pos_pub = self.create_publisher(Point, '/blue_pos', 10)

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
            f'[Layer 2] Building DIRECTIONAL route from robot start at '
            f'({start_x:.2f}, {start_y:.2f}) over {len(pruned_world_points)} waypoints.'
        )

        # quadrant-based, direction-respecting planning
        route_indices = self._directional_nearest_route(pruned_world_points, start_xy)

        # Optional: you can still run 2-opt on top if you want a little
        # extra smoothing, but usually not needed with directional sweeps.
        # If you prefer maximum “respect the direction” behavior, you can
        # comment the 2-opt block out.
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

    def _astar_distance_between_points(self, p1, p2):
        """
        Estimate travel cost between two waypoints using A* path length
        over the inflated map.

        Returns a distance in meters. If no path is found, returns a very
        large penalty so that route construction avoids this edge.
        """
        path = self.astar_plan(p1, p2)
        if path is None or len(path) < 2:
            # Unreachable or degenerate; treat as huge cost
            return 1e6

        total = 0.0
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    def _build_astar_distance_matrix(self, points):
        """
        Precompute pairwise A* distances between waypoints.

        points: list of (x, y) in world coordinates.

        Returns:
            dist[i][j] = A* path length (meters) from points[i] to points[j],
                         or a large value if no path exists.
        """
        n = len(points)
        dist = [[0.0] * n for _ in range(n)]

        self.get_logger().info(
            f'[Layer 2] Building A* distance matrix for {n} waypoints. '
            'This may take a few seconds...'
        )

        for i in range(n):
            dist[i][i] = 0.0
            for j in range(i + 1, n):
                d = self._astar_distance_between_points(points[i], points[j])
                dist[i][j] = d
                dist[j][i] = d

        self.get_logger().info('[Layer 2] A* distance matrix computation complete.')
        return dist

    def _nearest_neighbor_route_astar(self, points, start_xy):
        """
        Construct a route over `points` using a greedy nearest neighbor
        heuristic, where "distance" is A* path length on the map.

        This tends to respect doors / corridors and avoids suggesting
        jumps through walls like straight-line distances do.
        """
        n = len(points)
        if n == 0:
            self.get_logger().warn('[Layer 2] _nearest_neighbor_route_astar: no points.')
            return []

        # Precompute A* distances between all waypoints
        dist = self._build_astar_distance_matrix(points)

        # Choose initial waypoint as the one closest (Euclidean) to start pose
        best_first = None
        best_d = float('inf')
        for i, (x, y) in enumerate(points):
            d = math.hypot(x - start_xy[0], y - start_xy[1])
            if d < best_d:
                best_d = d
                best_first = i

        if best_first is None:
            self.get_logger().warn(
                '[Layer 2] _nearest_neighbor_route_astar: could not pick first waypoint.'
            )
            return []

        unvisited = set(range(n))
        route = []
        cur = best_first
        route.append(cur)
        unvisited.remove(cur)

        self.get_logger().info(
            f'[Layer 2] _nearest_neighbor_route_astar: starting from index {cur}.'
        )

        while unvisited:
            next_idx = None
            next_cost = float('inf')
            for j in unvisited:
                d = dist[cur][j]
                if d < next_cost:
                    next_cost = d
                    next_idx = j

            if next_idx is None:
                # Something went very wrong; break out
                self.get_logger().warn(
                    '[Layer 2] _nearest_neighbor_route_astar: no next waypoint found.'
                )
                break

            route.append(next_idx)
            unvisited.remove(next_idx)
            cur = next_idx

        self.get_logger().info(
            f'[Layer 2] _nearest_neighbor_route_astar built route of length {len(route)}.'
        )
        return route

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

    def _directional_nearest_route(self, points, start_xy):
        """
        Directional nearest-neighbor route:

        - First waypoint: globally closest to start_xy.
        - Determine its "area" (quadrant) relative to the start:
              Q0: +x, +y
              Q1: +x, -y
              Q2: -x, +y
              Q3: -x, -y
        - While there are points left in the current quadrant:
              pick the nearest to the CURRENT robot position.
        - When a quadrant is exhausted, move to the next quadrant that still
          has points, in a fixed cyclic order, and repeat.

        This guarantees:
        - all waypoints in the initial direction (+x,+y etc.) are visited
          before switching to opposite regions.
        """
        n = len(points)
        if n == 0:
            self.get_logger().warn('[Layer 2] _directional_nearest_route: no points.')
            return []

        # --- Precompute quadrant of each point relative to the *start* pose ---
        def quadrant_for(dx, dy):
            if dx >= 0.0 and dy >= 0.0:
                return 0   # +x, +y
            if dx >= 0.0 and dy < 0.0:
                return 1   # +x, -y
            if dx < 0.0 and dy >= 0.0:
                return 2   # -x, +y
            return 3       # -x, -y

        q_of = {}
        for i, (x, y) in enumerate(points):
            dx = x - start_xy[0]
            dy = y - start_xy[1]
            q_of[i] = quadrant_for(dx, dy)

        # --- Choose first waypoint: globally nearest to start ---
        best_first = None
        best_d = float('inf')
        for i, (x, y) in enumerate(points):
            d = math.hypot(x - start_xy[0], y - start_xy[1])
            if d < best_d:
                best_d = d
                best_first = i

        if best_first is None:
            self.get_logger().warn(
                '[Layer 2] _directional_nearest_route: could not pick first waypoint.'
            )
            return []

        current_xy = start_xy
        current_quadrant = q_of[best_first]

        # Fixed cyclic order of quadrants starting from the initial one.
        all_quads = [0, 1, 2, 3]
        start_pos = all_quads.index(current_quadrant)
        cyclic_quads = all_quads[start_pos:] + all_quads[:start_pos]

        unvisited = set(range(n))
        route = []

        self.get_logger().info(
            f'[Layer 2] _directional_nearest_route: first waypoint #{best_first}, '
            f'initial quadrant={current_quadrant}.'
        )

        while unvisited:
            # 1) Try to pick from the current quadrant
            candidates = [i for i in unvisited if q_of[i] == current_quadrant]

            if not candidates:
                # 2) No more points in this quadrant: move to the next quadrant
                remaining_quads = {q_of[i] for i in unvisited}
                next_quad = None
                for q in cyclic_quads:
                    if q in remaining_quads and q != current_quadrant:
                        next_quad = q
                        break

                if next_quad is None:
                    # No quadrants left (should not really happen here)
                    break

                self.get_logger().info(
                    f'[Layer 2] _directional_nearest_route: switching quadrant '
                    f'{current_quadrant} -> {next_quad}.'
                )
                current_quadrant = next_quad
                continue

            # 3) Choose nearest waypoint in the current quadrant to the *current* position
            best_idx = None
            best_dist = float('inf')
            for i in candidates:
                x, y = points[i]
                d = math.hypot(x - current_xy[0], y - current_xy[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            route.append(best_idx)
            unvisited.remove(best_idx)
            current_xy = points[best_idx]

        self.get_logger().info(
            f'[Layer 2] _directional_nearest_route built route of length {len(route)}.'
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

    def _is_cell_free_static(self, row: int, col: int) -> bool:
        """
        Check if a cell is free using ONLY the inflated static map
        (ignores dynamic_occupancy).

        We use this as a relaxed constraint for global A* so that
        temporary dynamic obstacles don't make waypoints "unreachable".
        """
        if (
            row < 0 or row >= self.map_height or
            col < 0 or col >= self.map_width
        ):
            return False

        if self.inflated_occupancy[row, col] != 0:
            return False

        return True

    def is_cell_free_rrt(self, row: int, col: int) -> bool:
        """
        RRT*-specific cell-free check.

        Special case: allow the RRT* start cell even if inflated/dynamic occupancy
        says it's blocked. This lets the tree grow *out* of a "stuck" start.
        """
        # If we know the RRT* start cell, treat it as free
        if hasattr(self, 'rrt_start_idx') and self.rrt_start_idx is not None:
            sr, sc = self.rrt_start_idx
            if row == sr and col == sc:
                return True

        # Normal clearance check everywhere else
        k = max(0, int(self.rrt_clearance_cells))
        for r in range(row - k, row + k + 1):
            for c in range(col - k, col + k + 1):
                if not self._is_cell_free(r, c):
                    return False
        return True

    def _find_nearest_free_goal_cell_rrt(self, g_row: int, g_col: int, max_radius: int = 5):
        """
        RRT*-specific version of goal snapping:
        look for the nearest cell that is free according to is_cell_free_rrt()
        (i.e., including clearance), *not* the looser _is_cell_free().
        """
        best = None
        best_dist = None

        for dr in range(-max_radius, max_radius + 1):
            for dc in range(-max_radius, max_radius + 1):
                r = g_row + dr
                c = g_col + dc

                # Use the RRT notion of free (with clearance)
                if not self.is_cell_free_rrt(r, c):
                    continue

                d = math.hypot(dr, dc)
                if best is None or d < best_dist:
                    best = (r, c)
                    best_dist = d

        return best

    def _find_nearest_free_goal_cell(self, g_row: int, g_col: int, max_radius: int = 5):
        """
        Search in a square neighborhood around (g_row, g_col) for the nearest cell
        that is free according to _is_cell_free (inflated + dynamic).

        Returns (row, col) or None if no free cell is found in that radius.
        """
        best = None
        best_dist = None

        for dr in range(-max_radius, max_radius + 1):
            for dc in range(-max_radius, max_radius + 1):
                r = g_row + dr
                c = g_col + dc
                if not self._is_cell_free(r, c):
                    continue
                d = math.hypot(dr, dc)
                if best is None or d < best_dist:
                    best = (r, c)
                    best_dist = d

        return best

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

        # --- Stage 0: allow start cell to be "occupied" (we just warn) ---
        if not self._is_cell_free(s_row, s_col):
            self.get_logger().warn(
                '[Layer 3] Start cell is occupied (static/dynamic); '
                'continuing A* planning anyway.'
            )

        # --- Stage 1: ensure goal cell is actually free (snap to nearest free if needed) ---
        if not self._is_cell_free(g_row, g_col):
            self.get_logger().warn(
                '[Layer 3] Goal cell is not free in inflated/dynamic map. '
                'Searching nearby free cell for goal...'
            )
            snapped = self._find_nearest_free_goal_cell(g_row, g_col, max_radius=5)
            if snapped is None:
                self.get_logger().warn(
                    '[Layer 3] No nearby free cell found for goal within radius 5. '
                    'A* will treat this waypoint as unreachable.'
                )
                return None
            g_row, g_col = snapped
            gx, gy = self.grid_to_world(g_row, g_col)
            self.get_logger().info(
                f'[Layer 3] Snapped goal to nearest free cell at '
                f'grid=({g_row},{g_col}), world=({gx:.2f},{gy:.2f}).'
            )

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

        def run_astar(cell_free_fn):
            def heuristic(r, c):
                return math.hypot(r - g_row, c - g_col)

            import heapq
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

                    if not cell_free_fn(nr, nc):
                        continue

                    neighbor = (nr, nc)
                    tentative_g = g_score[(cur_row, cur_col)] + move_cost

                    if neighbor in g_score and tentative_g >= g_score[neighbor]:
                        continue

                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc)
                    came_from[neighbor] = (cur_row, cur_col)
                    heapq.heappush(open_set, (f_score, neighbor))

            return None

        # --- Stage 2: A* with full constraints (inflated + dynamic) ---
        path = run_astar(self._is_cell_free)
        if path is not None:
            return path

        self.get_logger().warn(
            '[Layer 3] A* with dynamic obstacles failed or no path found. '
            'Retrying with STATIC map only (ignoring dynamic layer)...'
        )

        # --- Stage 3: A* with static-only constraints (inflated_occupancy only) ---
        path_static = run_astar(self._is_cell_free_static)
        if path_static is not None:
            self.get_logger().warn(
                '[Layer 3] A* succeeded with STATIC-only map. '
                'FOLLOW_PATH / RRT* will handle dynamic obstacles locally.'
            )
            return path_static

        self.get_logger().warn(
            '[Layer 3] A*: failed to find a path within iteration limit, '
            'even with static-only map.'
        )
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

    def _find_unvisited_waypoint_on_path(self, path_world):
        """
        Given an A* path (list of (x,y)), check if any OTHER patrol waypoint
        (unvisited & not unreachable) lies along this path.

        We look for the earliest such waypoint along the path and return
        its index in self.patrol_waypoints, or None if none found.
        """
        if not path_world or not self.patrol_waypoints:
            return None

        # Use at least goal_tolerance, but don't go crazy
        tol = max(self.goal_tolerance, 0.75)

        best_wp_idx = None
        best_path_pos = None

        for wp_idx, (wx, wy) in enumerate(self.patrol_waypoints):
            # Skip the current target; we only care about "bonus" waypoints
            if wp_idx == self.current_waypoint_idx:
                continue

            # Skip already handled waypoints
            if wp_idx in self.visited_waypoints or wp_idx in self.unreachable_waypoints:
                continue

            # Find earliest point on the path that passes near this waypoint
            for j, (px, py) in enumerate(path_world):
                if math.hypot(px - wx, py - wy) <= tol:
                    # Earlier along the path is better
                    if best_path_pos is None or j < best_path_pos:
                        best_path_pos = j
                        best_wp_idx = wp_idx
                    break  # no need to keep scanning this waypoint

        if best_wp_idx is not None:
            self.get_logger().info(
                f"[Layer 3] _find_unvisited_waypoint_on_path: "
                f"unvisited waypoint #{best_wp_idx} lies on A* path at "
                f"path index {best_path_pos}."
            )

        return best_wp_idx

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

        # --- SPECIAL CASE: robot's own cell should *not* trigger "path blocked" ---
        # We still allow A* and RRT* to see this as occupied for collision logic,
        # but for path-block detection we don't want "nearest_idx=0" to scream forever.
        if self.current_pose is not None:
            rx = self.current_pose.pose.pose.position.x
            ry = self.current_pose.pose.pose.position.y
            r_idx = self.world_to_grid(rx, ry)
            if r_idx is not None:
                rr, rc = r_idx
                if row == rr and col == rc:
                    # Treat as NOT blocked in this specific check
                    return False

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

        for i in range(nearest_idx + 1, max_idx + 1):
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

        # Remember the RRT* start cell so we can treat it as free
        self.rrt_start_xy = (sx, sy)
        self.rrt_start_idx = self.world_to_grid(sx, sy)

        # --- Debug: check map validity of start / goal for RRT* ---
        start_idx_dbg = self.world_to_grid(sx, sy)
        goal_idx_dbg  = self.world_to_grid(gx, gy)
        if start_idx_dbg is not None:
            sr, sc = start_idx_dbg
            self.get_logger().info(
                f'[Layer 3] RRT*: start cell=({sr},{sc}), '
                f'free_rrt={self.is_cell_free_rrt(sr, sc)}, '
                f'free_static={self._is_cell_free(sr, sc)}'
            )
        else:
            self.get_logger().warn('[Layer 3] RRT*: start outside map.')

        if goal_idx_dbg is not None:
            gr, gc = goal_idx_dbg
            self.get_logger().info(
                f'[Layer 3] RRT*: goal  cell=({gr},{gc}), '
                f'free_rrt={self.is_cell_free_rrt(gr, gc)}, '
                f'free_static={self._is_cell_free(gr, gc)}'
            )
        else:
            self.get_logger().warn('[Layer 3] RRT*: goal outside map.')

        # Per-run debug counters
        samples_total = 0
        segment_collision_fail = 0
        goal_connect_attempts = 0

        self.get_logger().info(
            f'[Layer 3] RRT*: start={start_xy}, goal={goal_xy}, center={center_xy}, '
            f'max_iter={self.rrt_max_iterations}'
        )

        start_node = self._RRTNode(sx, sy, parent=None, cost=0.0)
        nodes = [start_node]
        goal_node = None

        best_node = start_node
        best_dist = math.hypot(start_node.x - gx, start_node.y - gy)

        for it in range(self.rrt_max_iterations):
            if it % 200 == 0 and it > 0:
                self.get_logger().info(
                    f'[Layer 3] RRT*: iteration {it}, nodes={len(nodes)}'
                )

            sample_x, sample_y = self._rrt_sample(cx, cy, gx, gy)
            samples_total += 1

            nearest = self._rrt_nearest(nodes, sample_x, sample_y)
            if nearest is None:
                self.get_logger().debug(
                    f'[Layer 3] RRT*: no nearest node for sample '
                    f'({sample_x:.2f},{sample_y:.2f})'
                )
                continue

            new_x, new_y = self._rrt_steer(nearest, sample_x, sample_y)

            if not self._segment_collision_free(nearest.x, nearest.y, new_x, new_y):
                segment_collision_fail += 1
                if segment_collision_fail % 50 == 0:
                    self.get_logger().debug(
                        f'[Layer 3] RRT*: collision on segment '
                        f'({nearest.x:.2f},{nearest.y:.2f})'
                        f' -> ({new_x:.2f},{new_y:.2f}) '
                        f'(fails so far={segment_collision_fail})'
                    )
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

            dgoal = math.hypot(new_node.x - gx, new_node.y - gy)
            if dgoal < best_dist:
                best_dist = dgoal
                best_node = new_node

            for nb in neighbors:
                d = math.hypot(nb.x - new_node.x, nb.y - new_node.y)
                if new_node.cost + d < nb.cost and self._segment_collision_free(new_node.x, new_node.y, nb.x, nb.y):
                    nb.parent = new_node
                    nb.cost = new_node.cost + d

            dist_to_goal = math.hypot(new_node.x - gx, new_node.y - gy)
            if dist_to_goal <= self.rrt_step_size * 2.0:
                goal_connect_attempts += 1
                if self._segment_collision_free(new_node.x, new_node.y, gx, gy):
                    goal_node = self._RRTNode(
                        gx, gy,
                        parent=new_node,
                        cost=new_node.cost + dist_to_goal
                    )
                    self.get_logger().info(
                        f'[Layer 3] RRT*: reached goal at iteration {it}, '
                        f'nodes={len(nodes)}, '
                        f'samples={samples_total}, '
                        f'goal_connect_attempts={goal_connect_attempts}.'
                    )
                    break
                else:
                    segment_collision_fail += 1

        if goal_node is None:
            self.get_logger().warn(
                "[Layer 3] RRT*: FAILED to reach goal; returning best partial path. "
                f"best_dist_to_goal={best_dist:.2f} m, nodes={len(nodes)}"
            )

            # Reconstruct partial path from best_node back to start
            path = []
            cur = best_node
            while cur is not None:
                path.append((cur.x, cur.y))
                cur = cur.parent
            path.reverse()

            # Require at least a tiny “escape” (avoid returning start-only)
            if len(path) >= 2 and math.hypot(path[-1][0]-sx, path[-1][1]-sy) >= max(0.25, self.rrt_step_size):
                return path

            return None

        path = []
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()

        self.get_logger().info(f'[Layer 3] RRT*: local path length = {len(path)} waypoints.')
        self.rrt_start_idx = None
        return path

    def _obstacle_too_close_ahead(self, max_distance=None, fov_deg=90.0):
        """
        Return True if any obstacle is closer than max_distance in a symmetric
        FOV around the front. Defaults to self.obstacle_avoid_distance.
        """
        if self.latest_scan is None:
            return False

        if max_distance is None:
            max_distance = self.obstacle_avoid_distance

        scan = self.latest_scan
        half_fov = math.radians(fov_deg / 2.0)

        angle = scan.angle_min
        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue
            if r < scan.range_min or r > scan.range_max:
                angle += scan.angle_increment
                continue

            if abs(angle) <= half_fov and r < max_distance:
                self.get_logger().info(
                    f'[Layer 3] Obstacle too close ahead: range={r:.2f} m, '
                    f'threshold={max_distance:.2f} m, angle={angle:.2f} rad.'
                )
                return True
            angle += scan.angle_increment

        return False

    def _scan_median_in_sector_rad(self, center_rad: float, width_rad: float) -> float:
        """
        Return the median range in a sector around center_rad.

        Used for localization (balls), which is less sensitive to a single
        spurious beam and more interested in a stable, typical distance.
        """
        if self.latest_scan is None:
            return float('inf')

        scan = self.latest_scan
        half = width_rad / 2.0

        angle = scan.angle_min
        vals = []

        for r in scan.ranges:
            if math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue

            # Ignore extremes: walls far away or invalid close junk
            if r < scan.range_min or r > 0.99 * scan.range_max:
                angle += scan.angle_increment
                continue

            if (center_rad - half) <= angle <= (center_rad + half):
                vals.append(r)

            angle += scan.angle_increment

        if not vals:
            return float('inf')

        vals.sort()
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            med = vals[mid]
        else:
            med = 0.5 * (vals[mid - 1] + vals[mid])

        self.get_logger().debug(
            f"[Layer 5] _scan_median_in_sector_rad: center={center_rad:.3f}, "
            f"width={width_rad:.3f}, median={med:.2f}, n={len(vals)}"
        )
        return med

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

            # Ignore ranges at/near max range – often the wall behind the ball
            if r < scan.range_min or r > 0.99 * scan.range_max:
                angle += scan.angle_increment
                continue

            if (center_rad - half) <= angle <= (center_rad + half):
                if r < min_r:
                    min_r = r

            angle += scan.angle_increment

        self.get_logger().debug(
            f"[Layer 5] _scan_min_in_sector_rad: center={center_rad:.3f} rad, "
            f"width={width_rad:.3f} rad -> min_r={min_r:.2f}",
        )
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

    def _start_scan_at_waypoint(self):
        """
        Initialize 360° scan at the current waypoint.
        """
        if self.current_pose is None:
            return

        pose = self.current_pose.pose.pose
        yaw = self._quat_to_yaw(pose.orientation)

        self.scan_start_time = self.get_clock().now()
        self.scan_start_yaw = yaw
        self.scan_last_yaw = yaw
        self.scan_accum_angle = 0.0

        self.get_logger().info(
            '[Layer 3] Starting SCAN_AT_WAYPOINT: rotating in place to search for balls.'
        )

    def _handle_scan_at_waypoint(self):
        """
        Rotate in place until we accumulate ~2π * scan_turns radians of rotation
        (or hit a timeout). Vision is active only during this state.
        """
        if self.current_pose is None:
            self._publish_stop()
            return

        if self.scan_start_time is None:
            # Safety: if not properly initialized, initialize now.
            self._start_scan_at_waypoint()

        now = self.get_clock().now()
        dt = (now - self.scan_start_time).nanoseconds / 1e9

        # Check timeout
        if dt > self.scan_timeout_sec:
            self.get_logger().warn(
                f'[Layer 3] SCAN_AT_WAYPOINT timeout after {dt:.1f}s. '
                'Continuing to next waypoint.'
            )
            self._publish_stop()
            self.state = 'SELECT_NEXT_WAYPOINT'
            return

        pose = self.current_pose.pose.pose
        yaw = self._quat_to_yaw(pose.orientation)

        if self.scan_last_yaw is None:
            self.scan_last_yaw = yaw

        # Increment accumulated absolute rotation
        dyaw = self._normalize_angle(yaw - self.scan_last_yaw)
        self.scan_accum_angle += abs(dyaw)
        self.scan_last_yaw = yaw

        required_angle = 2.0 * math.pi * self.scan_turns

        if self.scan_accum_angle >= required_angle:
            self.get_logger().info(
                f'[Layer 3] SCAN_AT_WAYPOINT complete: '
                f'accum_angle={self.scan_accum_angle:.2f} rad '
                f'(required={required_angle:.2f}).'
            )
            self._publish_stop()

            # if task completion is pending (all balls found),
            # finalize *now* instead of going to next waypoint.
            if self.pending_task_done:
                self.get_logger().info(
                    '[Layer 3] SCAN_AT_WAYPOINT: 360° done and '
                    'pending_task_done=True → finalizing task.'
                )
                # Temporarily move out of SCAN state so _maybe_finish_task_...
                # will actually finalize instead of deferring.
                self.state = 'PATROL_DONE'
                self._maybe_finish_task_if_balls_found()
                return

            # Normal behavior: go to next waypoint
            self.state = 'SELECT_NEXT_WAYPOINT'
            return

        # Command in-place rotation
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = float(self.scan_angular_vel)
        self.cmd_vel_pub.publish(twist)

        self.get_logger().info(
            f'[Layer 3] SCAN_AT_WAYPOINT: accum_angle={self.scan_accum_angle:.2f} / '
            f'{required_angle:.2f} rad, cmd_w={twist.angular.z:.2f}.',
            throttle_duration_sec=0.5
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

                # ⬇️ reset RRT failure counter and force a fresh A* plan
                self.rrt_fail_count = 0
                self.local_plan_active = False
                self.active_path_points = []

                self.get_logger().info(
                    '[Layer 3] AVOID_OBSTACLE: maneuver complete. '
                    'Resetting RRT fail count and replanning to waypoint.'
                )

                # Instead of blindly going back to FOLLOW_PATH on the old path,
                # recompute a new A* path from the *new* pose to the same waypoint.
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

        # Close enough to final goal
        if dist_to_goal < self.goal_tolerance:
            self.get_logger().info(
                f'[Layer 3] Goal reached (distance={dist_to_goal:.3f} < '
                f'{self.goal_tolerance:.3f}).'
            )
            self._publish_stop()
            # We have finished this segment (global or local); allow future replans.
            self.local_plan_active = False
            return True

        # Find nearest point on path
        nearest_idx, _ = self._nearest_path_index(rx, ry, self.active_path_points)
        if nearest_idx is None:
            self._publish_stop()
            return False

        self.current_path_index = nearest_idx

        # Pure pursuit: choose a lookahead target along the path
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

        # --- Heading-only region: if we're turned far away, stop linear motion
        if abs(heading_error) > self.heading_align_only_threshold:
            heading_factor = 0.0
        else:
            heading_factor = max(0.0, 1.0 - abs(heading_error) / math.pi)

        # Base linear speed from heading + distance
        linear_speed = self.max_linear_vel * heading_factor

        # Smooth braking as we approach final goal
        if dist_to_goal < 0.6:
            # Scale down linearly inside 0.6 m
            linear_speed *= dist_to_goal / 0.6

        # --- Obstacle-aware slowdown ahead (do NOT handle full avoidance here)
        slowdown_factor = 1.0
        if self.latest_scan is not None and self.obstacle_slowdown_distance > self.obstacle_avoid_distance:
            slow_min = self._scan_min_in_sector(0.0, 60.0)  # front 60 deg
            if not math.isinf(slow_min):
                if slow_min <= self.obstacle_avoid_distance:
                    slowdown_factor = 0.0
                elif slow_min < self.obstacle_slowdown_distance:
                    # Map [avoid_distance, slowdown_distance] -> [0, 1]
                    num = slow_min - self.obstacle_avoid_distance
                    den = (self.obstacle_slowdown_distance - self.obstacle_avoid_distance + 1e-3)
                    slowdown_factor = max(0.0, min(1.0, num / den))

        linear_speed *= slowdown_factor

        # Angular control
        angular_speed = self.angular_gain * heading_error
        if angular_speed > self.max_angular_vel:
            angular_speed = self.max_angular_vel
        elif angular_speed < -self.max_angular_vel:
            angular_speed = -self.max_angular_vel

        # If obstacle is close, linear_speed may be zero from slowdown_factor
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
        if not self.patrol_waypoints:
            return None

        start_idx = 0 if self.current_waypoint_idx is None else self.current_waypoint_idx + 1

        # Forward sweep
        for i in range(start_idx, len(self.patrol_waypoints)):
            if (i not in self.visited_waypoints) and (i not in self.unreachable_waypoints):
                self.get_logger().info(
                    f'[Layer 3] _choose_next_waypoint_index: chose #{i} '
                    f'by route order (forward sweep).'
                )
                return i

        # Leftovers behind us
        for i in range(0, start_idx):
            if (i not in self.visited_waypoints) and (i not in self.unreachable_waypoints):
                self.get_logger().info(
                    f'[Layer 3] _choose_next_waypoint_index: no forward candidates, '
                    f'chose leftover #{i} behind current position.'
                )
                return i

        # --- all waypoints seem "unreachable" → clear that blacklist and try once more
        if self.patrol_waypoints and len(self.unreachable_waypoints) == len(self.patrol_waypoints):
            self.get_logger().warn(
                '[Layer 3] _choose_next_waypoint_index: all patrol waypoints are in '
                'unreachable_waypoints. Clearing unreachable set and retrying.'
            )
            self.unreachable_waypoints.clear()

            # One more attempt with a clean slate
            return self._choose_next_waypoint_index()
    
        # Nothing left
        return None

    # ----------------------------------------------------------------------
    # Layer 4: Vision / ball detection
    # ----------------------------------------------------------------------
    @staticmethod
    def _contour_circularity(contour):
        """
        Return (area, circularity) for a contour.
        circularity = 4*pi*A / P^2 in [0,1], with 1 = perfect circle.
        """
        area = cv2.contourArea(contour)
        if area <= 0.0:
            return 0.0, 0.0
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            return area, 0.0
        circ = 4.0 * math.pi * area / (perimeter * perimeter)
        return area, circ

    def _bearing_from_pixel(self, u: int, img_width: int) -> float:
        """
        Map an image x-coordinate to a bearing in the LASER frame (radians).

        If camera intrinsics (fx, cx) are provided, use:
            bearing_cam = atan2(u - cx, fx)
        otherwise fall back to the HFOV-based linear model.
        """
        # Prefer calibrated intrinsics if available
        if self.camera_fx > 0.0 and self.camera_cx > 0.0:
            bearing_cam = math.atan2((u - self.camera_cx), self.camera_fx)
        else:
            img_cx = img_width / 2.0
            pixel_offset = (u - img_cx) / max(img_cx, 1.0)  # [-1, 1]
            hfov_rad = math.radians(self.camera_h_fov_deg)
            bearing_cam = pixel_offset * (hfov_rad / 2.0)

        yaw_offset_rad = math.radians(self.camera_laser_yaw_offset_deg)
        bearing_laser = bearing_cam + yaw_offset_rad
        return bearing_laser

    def _range_from_pixel_bearing(self, cx: int, img_width: int):
        """
        Quick LiDAR range estimate for a given pixel column.
        Uses the median range in a small sector around the corresponding laser bearing.
        """
        if self.latest_scan is None:
            return None

        bearing_laser = self._bearing_from_pixel(cx, img_width)
        range_window_rad = math.radians(6.0)   # +/-3 deg window

        r = self._scan_median_in_sector_rad(bearing_laser, range_window_rad)
        if math.isinf(r) or r <= 0.0:
            return None
        return r

    def _expected_ball_pixel_radius(self, range_m: float, img_width: int):
        """
        Expected *pixel* radius of the ball for a given LiDAR range, using a pinhole model.

        If the contour radius is wildly different from this, it's likely not the ball
        (e.g., a huge patch of wall or a tiny speck).
        """
        if range_m is None or range_m <= 0.1:
            return None

        hfov_rad = math.radians(self.camera_h_fov_deg)
        # simple pinhole: f = W / (2 * tan(FOV/2))
        f = img_width / (2.0 * math.tan(hfov_rad / 2.0))

        ball_radius = self.ball_diameter_m / 2.0
        alpha = math.atan2(ball_radius, range_m)
        return f * math.tan(alpha)

    def _image_to_cv2(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            return img
        except CvBridgeError as e:
            self.get_logger().warn(f'[Layer 4] CvBridge error: {e}')
            return None

    def _detect_balls_in_image(self, frame):
        """
        Detect red, green, blue balls using:

        - STRICT per-color HSV ranges (self.BALL_HSV_RANGES)
        - Ground-region ROI (ignore top of image)
        - Hard min/max area in pixel space
        - Circularity + aspect ratio cuts
        - Optional hue-uniformity filter inside contour
        - LiDAR size-vs-distance gating (expected angular size of ball)
        - Multi-frame stability + best-so-far champion logic
        """
        if frame is None:
            return

        img_h, img_w = frame.shape[:2]
        self.last_image_width = img_w

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        debug_frame = frame.copy()

        # --- Area constraints in pixel space ---
        img_area = float(img_h * img_w)

        # Base area is a "typical" ball size at mid-range.
        base_min_area_param = float(self.ball_min_contour_area)

        # Anything below this is junk
        min_area_hard = 0.6 * base_min_area_param

        # Below this is penalized in scoring, but allowed
        min_area_soft = base_min_area_param

        # Anything larger than this fraction of the image is "too big" to be a ball
        max_area_fraction = 0.05  # 5% of image; tune if needed
        max_area_hard = max_area_fraction * img_area

        # --- Circularity constraints ---
        min_circ_param = float(self.ball_min_circularity)
        circ_hard = max(0.5, min_circ_param)  # absolute floor
        circ_soft = max(0.7, circ_hard)       # prefer >=0.7

        # Ground ROI: ignore the top ~35% of the image (mostly walls)
        roi_top_fraction = 0.35
        roi_top = int(roi_top_fraction * img_h)

        img_cx = img_w / 2.0

        for color in self.ball_colors:
            ranges = self.BALL_HSV_RANGES.get(color, None)
            if not ranges:
                continue

            # --- Strict HSV mask for this color ---
            mask_u8 = np.zeros((img_h, img_w), dtype=np.uint8)
            for lower, upper in ranges:
                mask_u8 |= cv2.inRange(hsv, lower, upper)

            # Apply ground ROI: zero out top region
            mask_u8[:roi_top, :] = 0

            # Morphological cleanup
            kernel = np.ones((5, 5), np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                self.get_logger().info(
                    f'[Layer 4] No {color} contours after HSV+ROI+morph.',
                    throttle_duration_sec=2.0,
                )
                continue

            best_contour = None
            best_score = float('-inf')
            best_meta = None

            for c in contours:
                area, circ = self._contour_circularity(c)

                # no hard max-area gate
                if area < min_area_hard:
                    continue
                if circ < circ_hard:
                    continue

                x_c, y_c, rw, rh = cv2.boundingRect(c)
                aspect = max(rw, rh) / max(1, min(rw, rh))

                # Reject elongated blobs; balls should be close to round
                if aspect > 2.5:
                    continue

                cx = int(x_c + rw / 2)
                cy = int(y_c + rh / 2)

                # Hue uniformity inside contour: textured walls have larger hue std
                mask_contour = np.zeros_like(mask_u8)
                cv2.drawContours(mask_contour, [c], -1, 255, thickness=-1)
                hue_vals = h[mask_contour == 255]
                if hue_vals.size > 0:
                    hue_std = float(np.std(hue_vals))
                    if hue_std > 8.0:
                        # Too heterogeneous in hue to be a painted ball
                        continue

                # LiDAR-based size-vs-distance gating (if we have a scan)
                size_gate_ok = True
                range_est = self._range_from_pixel_bearing(cx, img_w)
                if range_est is not None and math.isfinite(range_est):
                    R_exp = self._expected_ball_pixel_radius(range_est, img_w)
                    if R_exp is not None:
                        R_bbox = 0.5 * max(rw, rh)
                        if range_est > 0.4:  # only gate when not *too* close
                            if not (0.5 * R_exp <= R_bbox):
                                size_gate_ok = False

                if not size_gate_ok:
                    continue

                center_offset = abs(cx - img_cx) / max(img_cx, 1.0)

                # --- scoring: prefer large, circular, centered blobs ---
                score_area = math.log(area + 1.0)
                if circ >= circ_soft:
                    circ_penalty = 0.0
                else:
                    circ_penalty = (circ_soft - circ)

                elongation = (aspect - 1.0) ** 2

                score = (
                    1.0 * score_area +
                    1.0 * circ -
                    0.6 * center_offset -
                    0.8 * elongation -
                    1.0 * circ_penalty
                )

                self.get_logger().info(
                    f'[Layer 4] {color} cand: A={area:.1f}, C={circ:.3f}, '
                    f'aspect={aspect:.2f}, off={center_offset:.2f}, score={score:.2f}',
                    throttle_duration_sec=0.5,
                )

                if score > best_score:
                    best_score = score
                    best_contour = c
                    best_meta = {
                        "area": float(area),
                        "circ": float(circ),
                        "aspect": float(aspect),
                        "center_offset": float(center_offset),
                        "bbox": (x_c, y_c, rw, rh),
                        "cx": cx,
                        "cy": cy,
                    }

            if best_contour is None or best_meta is None:
                self.get_logger().info(
                    f'[Layer 4] {color} candidates rejected after size/circle/physics gates.',
                    throttle_duration_sec=2.0,
                )
                continue

            # === Frame-to-frame stability filter ===
            prev_meta = self.ball_detection_meta.get(color)
            if prev_meta is None:
                stable_frames = 1
            else:
                stable_frames = prev_meta.get("stable_frames", 1) + 1
            best_meta["stable_frames"] = stable_frames

            if stable_frames < self.min_stable_frames_for_commit:
                # Draw but don't commit yet
                color_bgr = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0)}[color]
                x, y, rw, rh = best_meta["bbox"]
                cx = best_meta["cx"]
                cy = best_meta["cy"]
                cv2.rectangle(debug_frame, (x, y), (x + rw, y + rh), color_bgr, 1)
                cv2.circle(debug_frame, (cx, cy), 3, color_bgr, -1)
                self.ball_detection_meta[color] = best_meta
                continue

            # --- This frame's best contour for this color ---
            cx = best_meta["cx"]
            cy = best_meta["cy"]
            x, y, rw, rh = best_meta["bbox"]
            center_offset = best_meta["center_offset"]
            best_area = best_meta["area"]
            best_circ = best_meta["circ"]
            
            # We **do not** localize here anymore.
            # This block is only about picking the best pixel detection.
            base_score = best_score

            prev_score = self.ball_detection_score.get(color, float('-inf'))

            # Build meta once so we can reuse it in both branches
            meta = {
                "area": best_area,
                "circ": best_circ,
                "center_offset": center_offset,
                # no has_world here anymore – ALIGN_MEASURE will handle world estimation
                "stable_frames": best_meta["stable_frames"],
            }

            # --- SPECIAL CASE: while ALIGNING to this color, always track the latest pixel ---
            if self.state == 'ALIGN_MEASURE' and color == self.align_ball_color:
                # Ignore champion-score gate; PID / align logic needs the freshest pixel
                self.ball_detection_score[color] = base_score
                self.ball_detection_meta[color] = meta
                self.ball_pixel_obs[color] = (cx, cy)

            else:
                # Normal champion logic (for SCAN_AT_WAYPOINT etc.)
                if base_score <= prev_score:
                    self.get_logger().info(
                        f"[Layer 5] New {color} detection (frame_score={base_score:.2f}) "
                        f"worse than existing best (score={prev_score:.2f}); keeping old.",
                        throttle_duration_sec=1.0,
                    )
                    # still draw for visualization
                    color_bgr = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0)}[color]
                    cv2.rectangle(debug_frame, (x, y), (x + rw, y + rh), color_bgr, 2)
                    cv2.circle(debug_frame, (cx, cy), 5, color_bgr, -1)
                    continue

                self.ball_detection_score[color] = base_score
                self.ball_detection_meta[color] = meta
                self.ball_pixel_obs[color] = (cx, cy)

            # Only pixel champion here; world localization happens in ALIGN_MEASURE
            self.get_logger().info(
                f'[Layer 5] UPDATED BEST {color} pixel-only: '
                f'score={prev_score:.2f} -> {base_score:.2f}, '
                f'area={best_area:.1f}, circ={best_circ:.3f}, '
                f'center_offset={center_offset:.2f}.',
                throttle_duration_sec=1.0,
            )

            # Decide whether to trigger ALIGN_MEASURE from SCAN_AT_WAYPOINT.
            # We ALWAYS allow a first localization for this color.
            # After that, we only re-localize if we appear closer than the
            # best range we've already recorded.
            if self.state == 'SCAN_AT_WAYPOINT':
                approx_r = self._range_from_pixel_bearing(cx, img_w)
                prev_r = self.ball_range_est.get(color, None)

                trigger_align = False

                if not self.detected_balls[color]:
                    # First time we see this color → always localize
                    trigger_align = True
                else:
                    # Already have a pose; only re-localize if new range is closer
                    if approx_r is not None and math.isfinite(approx_r) and prev_r is not None:
                        if approx_r + self.relocalize_range_margin < prev_r:
                            trigger_align = True
                            self.get_logger().info(
                                f"[Layer 5] {color} seen again at approx_r={approx_r:.2f}m "
                                f"(best={prev_r:.2f}m); triggering re-localization."
                            )
                        else:
                            self.get_logger().info(
                                f"[Layer 5] {color} seen again at approx_r={approx_r:.2f}m, "
                                f"not closer than best={prev_r:.2f}m; no ALIGN_MEASURE."
                            )
                    else:
                        # No valid range estimate; fall back to first-detection behavior only
                        self.get_logger().info(
                            f"[Layer 5] {color} re-detected but approx range unavailable; "
                            "keeping existing localization."
                        )

                if trigger_align:
                    self.align_ball_color = color
                    self.prev_state_before_align = 'SCAN_AT_WAYPOINT'
                    self.state = 'ALIGN_MEASURE'

                    # --- RESET ALIGN/MEASURE INTERNALS (important!) ---
                    now = self.get_clock().now()
                    self.align_start_time = now              # when we entered ALIGN_MEASURE
                    self.align_center_stable_counter = 0     # stable-frame counter
                    self.align_center_latched = False        # not latched yet
                    self.align_settle_start_time = None      # not settling yet

                    # reset PID
                    self.align_pid_i = 0.0
                    self.align_pid_prev_err = 0.0
                    self.align_pid_prev_time = None

                    # reset champ score so ALIGN_MEASURE always uses fresh pixels
                    self.ball_detection_score[color] = float('-inf')
                    return

            # Draw debug overlay for this best contour
            color_bgr = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0)}[color]
            cv2.rectangle(debug_frame, (x, y), (x + rw, y + rh), color_bgr, 2)
            cv2.circle(debug_frame, (cx, cy), 5, color_bgr, -1)
            cv2.putText(
                debug_frame,
                f"{color} S={base_score:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                1,
                cv2.LINE_AA,
            )

        if self.show_debug_image:
            cv2.imshow('task3_camera_debug', debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.show_debug_image = False
                cv2.destroyWindow('task3_camera_debug')

    def _handle_align_measure(self):
        color = self.align_ball_color
        if color is None:
            # No active alignment target; fall back to previous state
            self.state = self.prev_state_before_align or 'SCAN_AT_WAYPOINT'
            return

        # --- Safety: timeout so we don't spin forever on a bad detection ---
        if self.align_start_time is not None and not self.align_center_latched:
            dt_align = (self.get_clock().now() - self.align_start_time).nanoseconds / 1e9
            if dt_align > self.align_timeout_sec:
                self.get_logger().warn(
                    f"[ALIGN_MEASURE] Timeout aligning to {color} after "
                    f"{dt_align:.1f}s. Returning to {self.prev_state_before_align}."
                )
                self.align_ball_color = None
                self.align_center_stable_counter = 0
                self.align_filtered_err = 0.0  # legacy field, harmless
                self.align_settle_start_time = None
                self.align_center_latched = False

                # reset PID state
                self.align_pid_i = 0.0
                self.align_pid_prev_err = 0.0
                self.align_pid_prev_time = None

                self._publish_stop()
                self.state = self.prev_state_before_align or 'SCAN_AT_WAYPOINT'
                return

        pix = self.ball_pixel_obs.get(color)
        if pix is None or self.last_image_width is None:
            # no current measurement; just stop and wait
            self._publish_stop()
            # also reset settle timer; we don't want to reuse old one
            self.align_settle_start_time = None
            # PID shouldn't integrate with junk
            self.align_pid_prev_time = self.get_clock().now()
            return

        # --------------------------------------------------------------
        # If we're already in the settle phase, DO NOT keep running PID.
        # Just hold v=0, w=0 and wait for align_settle_time_sec, then
        # take the LiDAR measurement and finalize.
        # --------------------------------------------------------------
        if self.align_settle_start_time is not None:
            now = self.get_clock().now()
            dt_settle = (now - self.align_settle_start_time).nanoseconds / 1e9

            # Always command a full stop while settling
            self._publish_stop()

            if dt_settle < self.align_settle_time_sec:
                # Still settling – no LiDAR yet, no PID
                return

            # Settle time satisfied → read LiDAR straight ahead and finalize
            self.get_logger().info(
                f"[ALIGN_MEASURE] {color} centered & static for "
                f"{dt_settle:.2f}s; measuring distance..."
            )

            r = self._scan_median_in_sector_rad(0.0, math.radians(4.0))  # widened from 3° to 4°

            if r is None or math.isnan(r) or math.isinf(r) or r <= 0.0:
                self.get_logger().warn(
                    f"[ALIGN_MEASURE] Invalid LiDAR r={r} for color={color}. "
                    "Resetting settle and waiting for a better reading."
                )
                # Reset settle and stability, but stay in ALIGN_MEASURE
                self.align_settle_start_time = None
                self.align_center_stable_counter = 0
                self.align_pid_i = 0.0
                self.align_pid_prev_err = 0.0
                self.align_pid_prev_time = None
                self.align_center_latched = False
                self.align_start_time = None
                return

            # Slightly less strict "too close" rejection
            if r < 0.20:  # was 0.25
                self.get_logger().warn(
                    f"[ALIGN_MEASURE] Rejecting LiDAR r={r:.2f} m for {color} "
                    "(too close; likely wall/furniture)."
                )
                self.align_ball_color = None
                self.align_center_stable_counter = 0
                self.align_settle_start_time = None
                self.align_pid_i = 0.0
                self.align_pid_prev_err = 0.0
                self.align_pid_prev_time = None
                self._publish_stop()
                self.state = self.prev_state_before_align or 'SCAN_AT_WAYPOINT'
                self.align_center_latched = False
                self.align_start_time = None
                return

            # Compute world position using front beam
            pose = self.current_pose.pose.pose
            yaw = self._quat_to_yaw(pose.orientation)
            rx = pose.position.x
            ry = pose.position.y

            wx = rx + r * math.cos(yaw)
            wy = ry + r * math.sin(yaw)

            idx = self.world_to_grid(wx, wy)
            if idx is None:
                self.get_logger().warn(
                    f"[ALIGN_MEASURE] {color} ball world pose outside map "
                    f"({wx:.2f}, {wy:.2f}); not finalizing."
                )
                self.align_ball_color = None
                self.align_center_stable_counter = 0
                self.align_settle_start_time = None
                self.align_pid_i = 0.0
                self.align_pid_prev_err = 0.0
                self.align_pid_prev_time = None
                self._publish_stop()
                self.state = self.prev_state_before_align or 'SCAN_AT_WAYPOINT'
                self.align_center_latched = False
                self.align_start_time = None
                return

            row, col = idx
            static_occ = None
            if self.static_occupancy is not None:
                static_occ = int(self.static_occupancy[row, col])

            if static_occ is not None and static_occ != 0:
                self.get_logger().warn(
                    f"[ALIGN_MEASURE] {color} ball landed on STATIC occupied cell "
                    f"({row},{col}); accepting but note overlap with static map."
                )

            if self.inflated_occupancy is not None and self.inflated_occupancy[row, col] != 0:
                self.get_logger().warn(
                    f"[ALIGN_MEASURE] {color} ball landed in INFLATED occupied cell "
                    f"({row},{col}); accepting but note it is close to a wall."
                )

            # -------------------------------
            # BEST-SO-FAR RANGE LOGIC HERE 👇
            # -------------------------------
            prev_r = self.ball_range_est.get(color, None)

            if (prev_r is None) or (r + self.relocalize_range_margin < prev_r):
                # New measurement is closer → update stored pose and range
                self.ball_world_est[color] = (wx, wy)
                self.ball_range_est[color] = r
                self.detected_balls[color] = True

                self.get_logger().info(
                    f"[ALIGN_MEASURE] Updated {color} ball pose: "
                    f"r={r:.2f}m (prev={prev_r if prev_r is not None else float('nan'):.2f}), "
                    f"world=({wx:.2f}, {wy:.2f})."
                )
            else:
                # Keep the old (closer) estimate, but still mark as detected
                self.detected_balls[color] = True
                self.get_logger().info(
                    f"[ALIGN_MEASURE] New {color} measurement r={r:.2f}m is not closer "
                    f"than best r={prev_r:.2f}m; keeping existing world pose."
                )

            # Reset alignment internals
            self.align_pid_i = 0.0
            self.align_pid_prev_err = 0.0
            self.align_pid_prev_time = None
            self.align_center_stable_counter = 0
            self.align_settle_start_time = None
            self.align_center_latched = False
            self.align_start_time = None
            self.align_ball_color = None

            # Return to scanning/patrol
            self.state = self.prev_state_before_align or 'SCAN_AT_WAYPOINT'
            self._publish_ball_markers()
            self._maybe_finish_task_if_balls_found()
            return

        # ------------------------------------------------------------------
        # Phase A: PID-based heading alignment to image center
        # ------------------------------------------------------------------
        cx, cy = pix
        img_cx = self.last_image_width / 2.0

        # Raw normalized error in [-1, 1]: +ve => ball to the right of center
        raw_err = (cx - img_cx) / max(img_cx, 1.0)

        # Apply deadband: tiny errors treated as zero
        if abs(raw_err) < self.align_deadband:
            err = 0.0
        else:
            err = raw_err

        # PID time step
        now = self.get_clock().now()
        if self.align_pid_prev_time is None:
            dt = 0.0
        else:
            dt = (now - self.align_pid_prev_time).nanoseconds / 1e9
        self.align_pid_prev_time = now

        # Protect against absurd dt (e.g. first frame)
        if dt < 1e-4:
            dt = 0.0

        # --- PID update on heading error (image-based) ---
        # Integral term
        if dt > 0.0:
            self.align_pid_i += err * dt
            # clamp integral to avoid windup
            self.align_pid_i = max(-self.align_i_max, min(self.align_i_max, self.align_pid_i))

        # Derivative term
        if dt > 0.0:
            d_err = (err - self.align_pid_prev_err) / dt
        else:
            d_err = 0.0

        self.align_pid_prev_err = err

        # PID output -> angular velocity command (sign flipped because
        # positive error means ball is to the RIGHT, so we need to turn LEFT)
        u = self.align_kp * err + self.align_ki * self.align_pid_i + self.align_kd * d_err
        w_cmd = -u

        # Clamp to a gentle max angular speed
        if w_cmd > self.align_max_angular_vel:
            w_cmd = self.align_max_angular_vel
        elif w_cmd < -self.align_max_angular_vel:
            w_cmd = -self.align_max_angular_vel

        if abs(err) <= self.align_center_tolerance:
            self.align_center_stable_counter += 1
        else:
            self.align_center_stable_counter = 0
            # Do NOT touch align_settle_start_time here.
            # Once we enter settle mode, the early-return branch above
            # keeps us there until we either finish or reset explicitly.

        self.get_logger().info(
            f"[ALIGN_MEASURE] err={err:.3f}, I={self.align_pid_i:.3f}, "
            f"stable={self.align_center_stable_counter}/{self.align_center_stable_frames}, "
            f"w_cmd={w_cmd:.3f}"
        )

        # --------------------------------------------------------------
        # LATCH WHEN CENTERED: enough stable frames + decent area
        # --------------------------------------------------------------
        meta = self.ball_detection_meta.get(color)
        best_area = None
        if meta is not None:
            best_area = meta.get("area", None)

        if (
            not self.align_center_latched and
            self.align_center_stable_counter >= self.align_center_stable_frames and
            best_area is not None and best_area > 20000.0  # tweak if needed
        ):
            # We've been centered for long enough on a solid blob -> latch
            self.align_center_latched = True
            self.align_settle_start_time = self.get_clock().now()
            self.align_center_stable_counter = 0

            self.get_logger().info(
                f"[ALIGN_MEASURE] {color} centered; starting settle timer "
                f"for {self.align_settle_time_sec:.2f}s before LiDAR read."
            )

            # Immediately stop; settle branch will keep us stopped
            self._publish_stop()
            return

        # ------------------------------------------------------------------
        # Still in Phase A (not latched yet): command PID-based turn
        # ------------------------------------------------------------------
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = w_cmd
        self.cmd_vel_pub.publish(twist)

    def _localize_ball_from_pixel(self, color, cx, cy, img_width, img_height):
        """
        Convert a ball pixel (cx, cy) into a world (x, y) using:

        - camera HFOV + camera_laser_yaw_offset_deg  → laser-frame bearing
        - median LiDAR range in a small sector around that bearing
        - robot pose + yaw  → world coordinates

        This uses the same _bearing_from_pixel() helper that we use for
        LiDAR size-vs-distance gating, so geometry stays consistent.
        """
        if self.latest_scan is None or self.current_pose is None:
            self.get_logger().debug(
                f'[Layer 5] _localize_ball_from_pixel: no scan/pose yet for color={color}.'
            )
            return None

        # 1) Pixel -> laser bearing
        bearing_laser = self._bearing_from_pixel(cx, img_width)

        self.get_logger().info(
            f'[Layer 5] Color={color}, pixel_x={cx}, img_w={img_width}, '
            f'bearing_laser={bearing_laser:.3f} rad (offset={self.camera_laser_yaw_offset_deg:.1f} deg).',
            throttle_duration_sec=0.5,
        )

        # 2) Range: median LiDAR distance around that bearing (narrow cone)
        range_window_rad = math.radians(2.0)  # total width ≈ 2° (±1°)
        r = self._scan_median_in_sector_rad(bearing_laser, range_window_rad)
        if math.isinf(r) or r <= 0.0:
            self.get_logger().debug(
                f'[Layer 5] No valid scan range for {color} ball at this bearing; '
                'keeping detection as image-only for now.'
            )
            return None

        # 3) Laser frame -> robot base frame (we assume same origin) -> world
        lx = r * math.cos(bearing_laser)
        ly = r * math.sin(bearing_laser)

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

        # reject localizations that land on walls / occupied cells
        static_occ = None
        if self.static_occupancy is not None:
            static_occ = int(self.static_occupancy[row, col])

        if static_occ is not None and static_occ != 0:
            self.get_logger().info(
                f'[Layer 5] Localized {color} ball in STATIC occupied cell '
                f'({row},{col}), candidate=({wx:.2f}, {wy:.2f}). Rejecting.'
            )
            return None

        # Optionally also reject inflated obstacles (very close to walls)
        if self.inflated_occupancy is not None and self.inflated_occupancy[row, col] != 0:
            self.get_logger().info(
                    f'[Layer 5] {color} ball fell on INFLATED cell ({row},{col}); '
                    'accepting but note it is close to a wall.'
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
            if not self.detected_balls.get(color, False):
                continue

            world_xy = self.ball_world_est.get(color, None)
            if (
                world_xy is None or
                not isinstance(world_xy, (tuple, list)) or
                len(world_xy) != 2
            ):
                self.get_logger().warn(
                    f"[Layer 5] Skipping marker for '{color}': "
                    f"invalid world_est = {world_xy}"
                )
                continue

            wx, wy = world_xy

            # 🔍 log the exact marker pose we’re about to publish
            self.get_logger().info(
                f"[Layer 5] Marker for {color} at world=({wx:.2f}, {wy:.2f})",
                throttle_duration_sec=1.0,
            )
            idx = self.world_to_grid(wx, wy)
            if idx is not None:
                r, c = idx
                occ = self.static_occupancy[r, c] if self.static_occupancy is not None else -1
                self.get_logger().info(
                    f"[Layer 5] {color} marker grid cell=({r},{c}), static_occ={occ}",
                    throttle_duration_sec=1.0,
                )
            
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = 'map'
            m.ns = 'balls'
            m.id = mid
            mid += 1

            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = float(wx)
            m.pose.position.y = float(wy)
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
            m.lifetime.nanosec = 0

            ma.markers.append(m)

        if ma.markers:
            self.get_logger().info(
                f"[Layer 5] Publishing {len(ma.markers)} ball markers.",
                throttle_duration_sec=1.0,
            )
            self.ball_markers_pub.publish(ma)
        else:
            self.get_logger().debug("[Layer 5] No ball markers to publish.")

    def _publish_ball_points(self):
        """
        Publish final geometry_msgs/Point for each ball color on:
          /red_pos, /green_pos, /blue_pos

        Only called once when all balls are localized, from
        _maybe_finish_task_if_balls_found().
        """
        color_to_pub = {
            'red': self.red_pos_pub,
            'green': self.green_pos_pub,
            'blue': self.blue_pos_pub,
        }

        for color in self.ball_colors:
            wx, wy = self.ball_world_est.get(color, (None, None))
            if wx is None or wy is None:
                self.get_logger().warn(
                    f"[Layer 5] Cannot publish /{color}_pos: missing world_est."
                )
                continue

            pt = Point()
            pt.x = float(wx)
            pt.y = float(wy)
            pt.z = 0.0

            pub = color_to_pub.get(color, None)
            if pub is not None:
                pub.publish(pt)
                self.get_logger().info(
                    f"[Layer 5] Published {color} position on /{color}_pos: "
                    f"({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})"
                )

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
        """
        Called periodically from timer_cb.

        New semantics:
        - As soon as ALL balls are localized at least once in world coordinates,
          we:
            * publish RViz markers
            * publish PoseArray
            * publish completion time
            * publish geometry_msgs/Point on /red_pos, /green_pos, /blue_pos
          and then:
            * switch the state machine to TASK_DONE
            * stop the robot permanently (until node restart)
        """
        # re-validate that each "localized" ball is actually on a free cell.
        for color in self.ball_colors:
            if not self.detected_balls.get(color, False):
                continue

            world_xy = self.ball_world_est.get(color, None)
            if not isinstance(world_xy, (tuple, list)) or len(world_xy) != 2:
                self.get_logger().warn(
                    f"[Layer 5] {color} marked detected but world_est invalid; "
                    f"clearing and continuing search."
                )
                self.detected_balls[color] = False
                self.ball_world_est[color] = None
                continue

            wx, wy = world_xy
            idx = self.world_to_grid(wx, wy)
            if idx is None:
                self.get_logger().warn(
                    f"[Layer 5] {color} world_est ({wx:.2f},{wy:.2f}) now outside map; "
                    "clearing and continuing search."
                )
                self.detected_balls[color] = False
                self.ball_world_est[color] = None
                continue

            row, col = idx
            static_occ = None
            if self.static_occupancy is not None:
                static_occ = int(self.static_occupancy[row, col])

            if static_occ is not None and static_occ != 0:
                self.get_logger().warn(
                    f"[Layer 5] {color} world_est moved onto STATIC occupied cell "
                    f"({row},{col}); clearing and continuing search."
                )
                self.detected_balls[color] = False
                self.ball_world_est[color] = None
                continue
        
        # Not all colors localized yet
        if not all(self.detected_balls.values()):
            return

        # ---- BLOCK: delay finalization until scan is done ----
        required_angle = 2.0 * math.pi * self.scan_turns
        if (
            self.state in ['SCAN_AT_WAYPOINT', 'ALIGN_MEASURE']
            and self.scan_accum_angle < required_angle - 1e-3
        ):
            # We know all balls are done, but we're in the middle of a scan.
            # Mark that task completion is pending, but do not finalize yet.
            if not self.pending_task_done:
                self.get_logger().info(
                    '[Layer 5] All balls localized during scan; '
                    'will finalize after current 360° completes.'
                )
            self.pending_task_done = True
            return
        
        # Already finalized once – don't spam topics or keep re-stopping
        if self.task_done_time is not None and self.state == 'TASK_DONE':
            return

        self.get_logger().info('[Layer 5] All balls localized. Finalizing and stopping robot.')

        # Publish markers + PoseArray + completion time
        self._publish_ball_markers()
        self._publish_balls_posearray_and_time()

        # Publish geometry_msgs/Point to /red_pos, /green_pos, /blue_pos
        self._publish_ball_points()

        # Mark logical "task completion time"
        self.task_done_time = self.get_clock().now()

        # Freeze the state machine and stop the robot
        self.state = 'TASK_DONE'
        self._publish_stop()

        # clear pending flag
        self.pending_task_done = False

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
        if frame is None:
            return

        # Only run detection when we are scanning or aligning to a ball.
        if self.state in ['SCAN_AT_WAYPOINT', 'ALIGN_MEASURE']:
            self._detect_balls_in_image(frame)
            self._publish_ball_markers()
        else:
            # While driving between waypoints, we ignore vision
            # so navigation can focus on moving quickly.
            pass

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

            # If RRT sees the goal as not free (due to dynamic + clearance),
            # snap it to the nearest cell that *RRT* considers free.
            if not self.is_cell_free_rrt(gr, gc):
                self.get_logger().warn(
                    f'[Layer 3] REPLAN_LOCAL: RRT* goal cell ({gr},{gc}) not free. '
                    'Searching nearby RRT-free cell...'
                )
                snapped = self._find_nearest_free_goal_cell_rrt(gr, gc, max_radius=5)
                if snapped is None:
                    self.get_logger().warn(
                        "[Layer 3] REPLAN_LOCAL: no nearby RRT-free cell for local goal. "
                        "Running AVOID_OBSTACLE and replanning to SAME waypoint (no skipping)."
                    )
                    self._start_avoidance()
                    self.state = 'AVOID_OBSTACLE'
                    return

                ngr, ngc = snapped
                goal_xy = self.grid_to_world(ngr, ngc)
                self.get_logger().info(
                    f'[Layer 3] REPLAN_LOCAL: snapped RRT* goal from ({gr},{gc}) '
                    f'to nearest RRT-free ({ngr},{ngc}) at world=({goal_xy[0]:.2f},{goal_xy[1]:.2f}).'
                )

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
            connect_ok = (
                math.hypot(local_path[-1][0] - goal_xy[0], local_path[-1][1] - goal_xy[1]) < 0.35
                and self._segment_collision_free(local_path[-1][0], local_path[-1][1], goal_xy[0], goal_xy[1])
            )

            local_splice = local_path + ([goal_xy] if connect_ok else [])

        new_active = prefix + local_splice + suffix
        self.active_path_points = new_active

        self.current_path_index = 0
        self.local_replan_start_index = None
        self.local_replan_goal_index = None

        # mark that we are now following a local (RRT*) plan
        self.local_plan_active = True

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.replan_cooldown_until = now_sec + self.replan_cooldown_s

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

        # Keep ball markers alive in RViz even if camera messages are sparse.
        # This will republish markers at 10 Hz based on the latest world estimates.
        self._publish_ball_markers()

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
                self.get_logger().info(
                    f'[Layer 3] No more reachable waypoints. '
                    f'visited={len(self.visited_waypoints)}, '
                    f'unreachable={len(self.unreachable_waypoints)}, '
                    f'total={len(self.patrol_waypoints)}.'
                )
                self.state = 'PATROL_DONE'
                self._publish_stop()
                self._maybe_finish_task_if_balls_found()
                return

            # We DO have a candidate waypoint
            self.current_waypoint_idx = idx
            wx, wy = self.patrol_waypoints[idx]

            rx = self.current_pose.pose.pose.position.x
            ry = self.current_pose.pose.pose.position.y
            d = math.hypot(wx - rx, wy - ry)

            # Optional: if we already happen to be inside tolerance, auto-mark visited
            if d <= self.goal_tolerance:
                self.visited_waypoints.add(idx)
                self.get_logger().info(
                    f'[Layer 3] Waypoint #{idx} already within tolerance '
                    f'(d={d:.2f} <= {self.goal_tolerance:.2f}). Marking visited.'
                )
                # Stay in SELECT_NEXT_WAYPOINT so next tick picks the next one
                return

            self.get_logger().info(
                f'[Layer 3] Transition: SELECT_NEXT_WAYPOINT -> PLAN_TO_WAYPOINT '
                f'for waypoint #{idx} at ({wx:.2f}, {wy:.2f}).'
            )
            self.state = 'PLAN_TO_WAYPOINT'
            return

        if self.state == 'PLAN_TO_WAYPOINT':
            target_idx = self.current_waypoint_idx
            wx, wy = self.patrol_waypoints[target_idx]
            rx = self.current_pose.pose.pose.position.x
            ry = self.current_pose.pose.pose.position.y

            self.get_logger().info(
                f'[Layer 3] PLAN_TO_WAYPOINT: A* from ({rx:.2f}, {ry:.2f}) '
                f'to waypoint #{target_idx} = ({wx:.2f}, {wy:.2f})'
            )

            # First, plan to the intended target
            path_world = self.astar_plan((rx, ry), (wx, wy))

            # If A* failed or returned a degenerate path, mark as UNREACHABLE
            if path_world is None or len(path_world) < 2:
                self.get_logger().warn(
                    f'[Layer 3] A* failed or path too short to waypoint '
                    f'#{target_idx}. Marking as UNREACHABLE and selecting another.'
                )
                self.unreachable_waypoints.add(target_idx)
                self.state = 'SELECT_NEXT_WAYPOINT'
                return

            # --- opportunistic pickup of waypoints that lie on this A* path ---
            alt_idx = self._find_unvisited_waypoint_on_path(path_world)

            if alt_idx is not None and alt_idx != target_idx:
                awx, awy = self.patrol_waypoints[alt_idx]
                self.get_logger().info(
                    f'[Layer 3] PLAN_TO_WAYPOINT: switching goal from '
                    f'#{target_idx} to on-path waypoint #{alt_idx} at '
                    f'({awx:.2f}, {awy:.2f}). Replanning A*.'
                )

                # Try to replan directly to this on-path waypoint
                new_path = self.astar_plan((rx, ry), (awx, awy))
                if new_path is not None and len(new_path) >= 2:
                    path_world = new_path
                    self.current_waypoint_idx = alt_idx
                else:
                    self.get_logger().warn(
                        f'[Layer 3] A* to on-path waypoint #{alt_idx} failed; '
                        'keeping original target and path.'
                    )
                    # fall back to original target_idx and path_world

            # A* (possibly re)planned successfully: store and publish path, then FOLLOW_PATH
            self.global_path_points = path_world
            self.active_path_points = list(path_world)
            self.current_path_index = 0

            # 🔧 fresh global path → fresh RRT* fuse
            self.rrt_fail_count = 0
            self.local_plan_active = False

            path_msg = self.build_path_msg(path_world)
            self.global_path_pub.publish(path_msg)

            self.get_logger().info(
                f'[Layer 3] A* path computed to waypoint '
                f'#{self.current_waypoint_idx} (len={len(path_world)}). '
                'Switching to FOLLOW_PATH.'
            )

            self.state = 'FOLLOW_PATH'
            return

        if self.state == 'SCAN_AT_WAYPOINT':
            self._handle_scan_at_waypoint()
            return

        if self.state == 'ALIGN_MEASURE':
            self._handle_align_measure()
            return

        if self.state == 'FOLLOW_PATH':
            now_sec = self.get_clock().now().nanoseconds / 1e9

            # Always honor emergency distance (true "brace for impact")
            if self._obstacle_too_close_ahead(max_distance=self.emergency_stop_dist):
                self.get_logger().warn(
                    '[Layer 3] FOLLOW_PATH: EMERGENCY obstacle too close. Running AVOID_OBSTACLE.'
                )
                self._start_avoidance()
                self.state = 'AVOID_OBSTACLE'
                return

            # If we just replanned locally, commit to moving for a short window
            if now_sec < self.replan_cooldown_until:
                # skip the normal near-threshold trigger so we actually move
                pass
            else:
                # normal threshold behavior
                if self._obstacle_too_close_ahead(max_distance=self.obstacle_avoid_distance):
                    self.get_logger().warn(
                        '[Layer 3] FOLLOW_PATH: obstacle too close ahead. Triggering AVOID_OBSTACLE.'
                    )
                    self._start_avoidance()
                    self.state = 'AVOID_OBSTACLE'
                    return

            # 2) Normal blocked-path check using occupancy (static + dynamic)
            if not self.local_plan_active:
                blocked, start_idx, goal_idx = self._check_path_blocked_ahead()
                if blocked:
                    if self.rrt_fail_count >= self.rrt_fail_limit:
                        self.get_logger().warn(
                            f"[Layer 3] FOLLOW_PATH: path blocked near idx={start_idx}, "
                            f"rrt_fail_count={self.rrt_fail_count} >= limit={self.rrt_fail_limit}. "
                            "Running AVOID_OBSTACLE and replanning to SAME waypoint (no skipping)."
                        )
                        self._start_avoidance()
                        self.state = 'AVOID_OBSTACLE'
                        return

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
                # Mark waypoint visited and start scan-at-waypoint
                self.visited_waypoints.add(self.current_waypoint_idx)
                self.get_logger().info(
                    f'[Layer 3] Waypoint #{self.current_waypoint_idx} reached. '
                    f'Visited {len(self.visited_waypoints)}/{len(self.patrol_waypoints)}. '
                    'Starting SCAN_AT_WAYPOINT.'
                )
                self._start_scan_at_waypoint()
                self.state = 'SCAN_AT_WAYPOINT'
            return

        if self.state == 'REPLAN_LOCAL':
            self._handle_replan_local()
            return

        if self.state == 'AVOID_OBSTACLE':
            self._handle_avoid_obstacle()
            return

        if self.state == 'PATROL_DONE':
            # Maybe we actually finished:
            self._maybe_finish_task_if_balls_found()
            if self.state == 'TASK_DONE':
                return

            balls_found = sum(self.detected_balls.values())
            self.get_logger().warn(
                f"[Layer 3] PATROL_DONE reached with only {balls_found}/3 balls "
                "localized. Restarting patrol over reachable waypoints."
            )

            # If *every* waypoint is currently marked unreachable, the blacklist
            # is clearly too aggressive – clear it and start fresh.
            if self.patrol_waypoints and len(self.unreachable_waypoints) == len(self.patrol_waypoints):
                self.get_logger().warn(
                    '[Layer 3] PATROL_DONE: all waypoints marked unreachable. '
                    'Clearing unreachable_waypoints so we can attempt them again.'
                )
                self.unreachable_waypoints.clear()

            # Allow revisiting everything that’s not in unreachable
            self.visited_waypoints.clear()

            # 🔧 Optional: new patrol sweep, so reset RRT* fail counter too
            self.rrt_fail_count = 0

            self.state = 'SELECT_NEXT_WAYPOINT'
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
