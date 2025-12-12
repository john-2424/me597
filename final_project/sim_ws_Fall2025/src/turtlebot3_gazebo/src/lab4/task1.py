#!/usr/bin/env python3

import math
import heapq
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict, Set

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, Path
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import LaserScan

from tf2_ros import Buffer, TransformListener


class PID:
    """Simple PID controller."""
    def __init__(self, kp: float, ki: float, kd: float,
                 i_limit: float = 1.0,
                 out_limit: Optional[Tuple[float, float]] = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.out_limit = out_limit  # (min, max) or None

        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def step(self, e: float, dt: float) -> float:
        """Compute PID output for error e with timestep dt."""
        if dt <= 0.0:
            dt = 1e-3

        # proportional
        p = self.kp * e

        # integral with clamp
        self.i += e * dt
        self.i = max(-self.i_limit, min(self.i_limit, self.i))

        # derivative
        if self.first:
            d = 0.0
            self.first = False
        else:
            d = (e - self.prev_e) / dt

        self.prev_e = e

        val = p + self.ki * self.i + self.kd * d

        if self.out_limit is not None:
            lo, hi = self.out_limit
            val = max(lo, min(hi, val))

        return val


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class Task1(Node):
    """
    Task 1 – Autonomous Mapping
    """

    # clustering / planning parameters
    CLUSTER_RADIUS_CELLS = 2       # how far neighbors can be and still be in same cluster
    MIN_GOAL_DIST_CELLS = 12        # minimum distance of goal from the robot (in grid cells)

    # NEW: frontier path-length band (in cells along A* path)
    FRONTIER_PATH_LEN_MIN_CELLS = 12
    FRONTIER_PATH_LEN_MAX_CELLS = 24

    # NEW: parameters for "far" alternative goals for inflated frontiers
    FRONTIER_FAR_ALT_MIN_OFFSET_CELLS = 4     # start this many cells away from original
    FRONTIER_FAR_ALT_MAX_OFFSET_CELLS = 15    # max search offset along rays

    # main inflation radius for A* / traversability
    PATH_CLEARANCE_CELLS = 6       # 6 cells * 0.05 m ≈ 0.30 m

    # extra safety for frontier / goal cells (pulled slightly further from walls)
    FRONTIER_CLEARANCE_CELLS = 6

    # NEW: max radius (in cells) when snapping start/goal to nearest traversable
    NEAREST_TRAVERSABLE_RADIUS_CELLS = 18

    BLOCKED_GOAL_RADIUS_CELLS = 3

    def __init__(self):
        super().__init__('task1_node')

        # ---------------------------
        # Map-related state (Stage 1)
        # ---------------------------
        self.map_received = False

        self.map_width: Optional[int] = None
        self.map_height: Optional[int] = None
        self.map_resolution: Optional[float] = None  # meters/cell
        self.map_origin_x: Optional[float] = None
        self.map_origin_y: Optional[float] = None

        # flat list of int8: -1 unknown, 0 free, 100 occupied
        self.map_data: Optional[List[int]] = None

        # ---------------------------
        # Robot pose state (Stage 2)
        # ---------------------------
        # Pose in map frame (meters, radians)
        self.robot_x: Optional[float] = None
        self.robot_y: Optional[float] = None
        self.robot_yaw: Optional[float] = None

        # Pose in map grid indices (cells)
        self.robot_mx: Optional[int] = None
        self.robot_my: Optional[int] = None

        self.robot_pose_received: bool = False

        # ---------------------------
        # TF buffer/listener (for map -> base_footprint)
        # ---------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------------------
        # Frontier / path state (Stage 4/5)
        # ---------------------------
        self.frontier_clusters: List[List[Tuple[int, int]]] = []
        self.frontier_goal: Optional[Tuple[int, int]] = None
        self.last_frontier_goal: Optional[Tuple[int, int]] = None  # remember last goal
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.current_path_world: Optional[List[Tuple[float, float]]] = None
        self.path_idx: Optional[int] = None

        # ---------------------------
        # Scan / obstacle state (Stage 5)
        # ---------------------------
        self.min_front_range: Optional[float] = None
        self.front_range_filt: Optional[float] = None
        self.obstacle_stop_dist = 0.35  # m

        self.min_front_range = None
        self.front_range_filt = None
        self.min_left_range = None
        self.min_right_range = None

        # ---------------------------
        # PID controllers (Stage 5)
        # ---------------------------
        # slightly slower but smoother body motion
        self.speed_max = 0.60      # linear speed cap
        self.speed_min = 0.08      # minimum creeping speed

        # sharper heading control
        self.heading_max = 1.0     # rad/s (faster turning allowed)
        self.heading_deadband = 0.04  # rad ≈ 2.3 degrees

        self.yaw_tol = 0.1         # rad
        self.slow_down_dist = 1.10 # m, start braking earlier

        self.last_ctrl_time: Optional[float] = None
        self.speed_hist: List[Tuple[float, float, float]] = []  # (t, x, y)

        self.max_accel = 0.35        # m/s^2  (safe-ish)
        self.max_decel = 0.55        # m/s^2  (can brake harder than accel)
        self.prev_speed_cmd = 0.0

        self.pid_speed = PID(
            kp=2.2,      # was 3.0
            ki=0.08,     # was 0.10
            kd=0.25,     # was 0.20 (more damping)
            i_limit=1.0,
            out_limit=(-self.speed_max, self.speed_max)
        )
        self.pid_heading = PID(
            kp=0.9,
            ki=0.008,
            kd=0.24,
            i_limit=0.6,
            out_limit=(-self.heading_max, self.heading_max)
        )

        # ---------------------------
        # Exploration state (Stage 6)
        # ---------------------------
        self.state = "WAIT_FOR_MAP"  # or "EXPLORE", "DONE"

        # Blocked goal bookkeeping (permanent-fix)
        self.blocked_goal_ttl_s = 25.0          # tune: 15–40s works well
        self.blocked_goals_max = 800            # hard cap so it can't explode
        self.blocked_goals = deque()            # holds tuples: (mx, my, stamp_sec)
        self.blocked_goals_set = set()          # fast dedupe

        # Frontier visit counts: how many times each frontier goal has been reached
        self.frontier_visit_counts: Dict[Tuple[int, int], int] = {}

        # NEW: frontier cells whose clusters were rejected due to inflation / clearance
        self.inflated_filtered_frontiers: List[Tuple[int, int]] = []

        # NEW: dynamic frontier clearance (we can shrink this in narrow passages)
        self.frontier_clearance_current = self.FRONTIER_CLEARANCE_CELLS
        self.frontier_clearance_min = 0

        # NEW: dynamic path clearance (for A* traversability / smoothing)
        self.path_clearance_current = self.PATH_CLEARANCE_CELLS
        self.path_clearance_min = 0

        self.escape_active = False
        self.escape_until = 0.0
        self.escape_dir = 1.0           # +1 left, -1 right
        self.escape_w = 0.8             # rad/s
        self.escape_duration_s = 0.9    # seconds per escape attempt
        self.escape_attempts = 0
        self.escape_attempt_limit = 5

        # ---------------------------
        # Misc counters
        # ---------------------------
        self.timer_counter = 0

        # ---------------------------
        # Subscribers
        # ---------------------------
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.map_update_sub = self.create_subscription(
            OccupancyGridUpdate,
            '/map_updates',
            self.map_update_callback,
            40
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # ---------------------------
        # Publishers
        # ---------------------------
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        self.frontier_markers_pub = self.create_publisher(
            MarkerArray,
            '/frontier_markers',
            10
        )
        self.global_plan_pub = self.create_publisher(
            Path,
            '/global_plan',
            10
        )

        # ---------------------------
        # Timer – behavior loop
        # ---------------------------
        self.timer = self.create_timer(0.1, self.timer_cb)  # 10 Hz control

        self.get_logger().info(
            f'Task1 node initialized. PATH_CLEARANCE_CELLS={self.PATH_CLEARANCE_CELLS}, '
            f'FRONTIER_CLEARANCE_CELLS={self.FRONTIER_CLEARANCE_CELLS}'
        )

    # -------------------------------------------------------------------------
    # Map callbacks (Stage 1)
    # -------------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        """Receive the full map and (re)initialize our internal grid."""
        info = msg.info

        first_time = not self.map_received
        resized = False

        if first_time:
            resized = True
        else:
            if (info.width != self.map_width or
                info.height != self.map_height or
                info.resolution != self.map_resolution or
                info.origin.position.x != self.map_origin_x or
                info.origin.position.y != self.map_origin_y):
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

            # Map geometry changed => old blocked-goal grid coords are unreliable
            self.blocked_goals.clear()
            self.blocked_goals_set.clear()
            self.inflated_filtered_frontiers.clear()  # also good to reset

            # Invalidate the current path on any geometry change
            if self.current_path is not None:
                self.get_logger().info(
                    'Map geometry changed (size/origin/resolution). '
                    'Clearing current path to replan on updated grid.'
                )
                self.clear_current_path()

            self.get_logger().info(
                f'/map received: size=({self.map_width} x {self.map_height}), '
                f'res={self.map_resolution:.3f} m/cell, '
                f'origin=({self.map_origin_x:.2f}, {self.map_origin_y:.2f})'
            )
        else:
            self.map_data = list(msg.data)
            self.map_received = True
            self.get_logger().info(
                '/map updated with same size (refresh).',
                throttle_duration_sec=5.0
            )

        if self.map_received and self.map_resolution is not None:
            self.stop_dist_tol = 0.6 * self.map_resolution

    def map_update_callback(self, msg: OccupancyGridUpdate):
        """
        Apply an incremental patch to our cached map_data.
        Assumes we have already received at least one full /map.
        """
        if not self.map_received or self.map_data is None:
            self.get_logger().warn('Got /map_updates before /map; ignoring.')
            return

        if self.map_width is None or self.map_height is None:
            self.get_logger().warn('Map dimensions unknown; ignoring /map_updates.')
            return

        # Safety: ignore updates that would go out of bounds
        if (msg.x < 0 or msg.y < 0 or
                msg.x + msg.width > self.map_width or
                msg.y + msg.height > self.map_height):
            self.get_logger().warn(
                f'Received invalid map update region: '
                f'({msg.x},{msg.y}) size=({msg.width}x{msg.height}) '
                f'vs map size=({self.map_width}x{self.map_height})'
            )
            return

        idx = 0
        for dy in range(msg.height):
            my = msg.y + dy
            for dx in range(msg.width):
                mx = msg.x + dx
                map_idx = my * self.map_width + mx
                if 0 <= map_idx < len(self.map_data):
                    self.map_data[map_idx] = msg.data[idx]
                idx += 1

        self.get_logger().info(
            f'Applied /map_updates patch at ({msg.x},{msg.y}) size=({msg.width}x{msg.height})',
            throttle_duration_sec=1.0
        )

    # -------------------------------------------------------------------------
    # Map helpers (Stage 1)
    # -------------------------------------------------------------------------

    def map_index(self, mx: int, my: int) -> Optional[int]:
        """Convert (mx, my) cell coordinates to flat array index."""
        if (self.map_width is None or self.map_height is None or
                mx < 0 or my < 0 or
                mx >= self.map_width or my >= self.map_height):
            return None
        return my * self.map_width + mx

    def world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates (meters, in map frame) to map cell indices."""
        if self.map_resolution is None or self.map_origin_x is None or self.map_origin_y is None:
            return None

        mx = int(math.floor((x - self.map_origin_x) / self.map_resolution))
        my = int(math.floor((y - self.map_origin_y) / self.map_resolution))

        if self.map_index(mx, my) is None:
            return None
        return mx, my

    def map_to_world(self, mx: int, my: int) -> Optional[Tuple[float, float]]:
        """Convert map cell indices to world coordinates (meters, in map frame)."""
        if (self.map_resolution is None or
                self.map_origin_x is None or self.map_origin_y is None):
            return None

        if self.map_index(mx, my) is None:
            return None

        x = self.map_origin_x + (mx + 0.5) * self.map_resolution
        y = self.map_origin_y + (my + 0.5) * self.map_resolution
        return x, y

    def cell_value(self, mx: int, my: int) -> Optional[int]:
        """Return raw occupancy value at (mx, my): -1, 0-100."""
        idx = self.map_index(mx, my)
        if idx is None or self.map_data is None:
            return None
        return self.map_data[idx]

    def is_unknown(self, mx: int, my: int) -> bool:
        v = self.cell_value(mx, my)
        return v == -1

    def is_free(self, mx: int, my: int) -> bool:
        v = self.cell_value(mx, my)
        return v == 0

    def is_occupied(self, mx: int, my: int, occ_threshold: int = 50) -> bool:
        v = self.cell_value(mx, my)
        return v is not None and v >= occ_threshold

    def is_traversable(self, mx: int, my: int) -> bool:
        """
        Traversable for the global planner:
        - must be free/unknown
        - must have some clearance to occupied cells
        """
        return self.is_safe_for_path(mx, my, clearance_cells=self.path_clearance_current)

    # -------------------------------------------------------------------------
    # Scan callback (Stage 5)
    # -------------------------------------------------------------------------

    def scan_callback(self, msg: LaserScan):
        if not msg.ranges:
            return

        # Helper: min range inside [a0, a1] radians
        def sector_min(a0, a1):
            vals = []
            for i, r in enumerate(msg.ranges):
                if math.isinf(r) or math.isnan(r) or r <= 0.0:
                    continue
                ang = msg.angle_min + i * msg.angle_increment
                if a0 <= ang <= a1:
                    vals.append(r)
            return min(vals) if vals else None

        # Sectors (tweak if needed)
        front = sector_min(math.radians(-20), math.radians(20))
        left  = sector_min(math.radians(20),  math.radians(80))
        right = sector_min(math.radians(-80), math.radians(-20))

        self.min_front_range = front
        self.min_left_range  = left
        self.min_right_range = right

        # Low-pass filter front to avoid jitter false-stops
        if front is None:
            self.front_range_filt = None
            return
        if self.front_range_filt is None:
            self.front_range_filt = front
        else:
            self.front_range_filt = 0.8 * self.front_range_filt + 0.2 * front

    # -------------------------------------------------------------------------
    # TF-based robot pose update (Stage 2)
    # -------------------------------------------------------------------------

    def update_robot_pose_from_tf(self):
        """
        Query TF for the transform map -> base_footprint and update
        robot_x, robot_y, robot_yaw and grid indices (robot_mx, robot_my).
        """
        try:
            t = self.tf_buffer.lookup_transform(
                'map',              # target frame
                'base_footprint',   # source frame
                Time()              # latest available
            )

            tx = t.transform.translation.x
            ty = t.transform.translation.y
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            yaw = self.yaw_from_quaternion(qx, qy, qz, qw)

            self.robot_x = tx
            self.robot_y = ty
            self.robot_yaw = yaw
            self.robot_pose_received = True

            # Also compute grid indices if map is ready
            if self.map_received:
                cell = self.world_to_map(tx, ty)
                if cell is not None:
                    self.robot_mx, self.robot_my = cell
                else:
                    self.robot_mx = None
                    self.robot_my = None

        except Exception as e:
            self.get_logger().warn(
                f'Failed to lookup transform map->base_footprint: {e}',
                throttle_duration_sec=5.0
            )

    @staticmethod
    def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
        """Convert a quaternion into yaw (rotation about Z)."""
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    # -------------------------------------------------------------------------
    # A* implementation (Stage 3)
    # -------------------------------------------------------------------------

    def bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Integer grid line between (x0, y0) and (x1, y1) using Bresenham's algorithm.
        Includes both endpoints.
        """
        points: List[Tuple[int, int]] = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0

        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx >= dy:
            err = dx // 2
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy // 2
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x1, y1))
        return points

    def has_line_of_sight(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """
        Return True if there is a collision-free *inflated* straight line
        between cells a and b.

        We require every cell along the line to be safe for path planning
        (free + clearance to obstacles). This preserves the inflation
        radius even after smoothing.
        """
        if not self.map_received or self.map_data is None:
            return False

        x0, y0 = a
        x1, y1 = b

        for (mx, my) in self.bresenham_line(x0, y0, x1, y1):
            # bounds check
            if self.map_index(mx, my) is None:
                return False

            # require inflated safety using the *current* path clearance
            if not self.is_safe_for_path(mx, my, self.path_clearance_current):
                return False

        return True

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Simplify an A* path by removing intermediate waypoints where possible,
        keeping the path collision-free AND inflated using has_line_of_sight.

        Returns a new, shorter list of (mx, my) cells.
        """
        if not path:
            return path
        if len(path) <= 2:
            return path

        smoothed: List[Tuple[int, int]] = []
        i = 0
        n = len(path)

        # Always keep the first point
        smoothed.append(path[0])

        while i < n - 1:
            j = n - 1
            # Try to jump as far as possible from i to j while maintaining LOS
            while j > i + 1:
                if self.has_line_of_sight(path[i], path[j]):
                    break
                j -= 1

            # If we only could connect to the next cell (no long jump), j == i+1
            smoothed.append(path[j])
            i = j

        return smoothed

    def find_nearest_traversable(self, mx: int, my: int,
                                 max_radius: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """
        Search in expanding 'rings' around (mx, my) for the nearest traversable cell.

        Uses is_traversable (free/unknown + PATH_CLEARANCE_CELLS) and stops
        as soon as it finds *any* traversable cell on the ring at radius r.

        Complexity: O(R^2) cells for radius R (8 * sum_{r=1..R} r),
        which is cheap even for R ~ 20.
        """
        if self.map_width is None or self.map_height is None:
            return None

        if max_radius is None:
            max_radius = self.NEAREST_TRAVERSABLE_RADIUS_CELLS

        # If the center is already traversable, return it immediately
        if self.is_traversable(mx, my):
            return (mx, my)

        # Expand outwards in Chebyshev distance r = 1..max_radius
        for r in range(1, max_radius + 1):
            # top and bottom edges of the square ring
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    nx = mx + dx
                    ny = my + dy
                    if self.map_index(nx, ny) is None:
                        continue
                    if self.is_traversable(nx, ny):
                        return (nx, ny)

            # left and right edges (excluding corners which were checked above)
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    nx = mx + dx
                    ny = my + dy
                    if self.map_index(nx, ny) is None:
                        continue
                    if self.is_traversable(nx, ny):
                        return (nx, ny)

        # Nothing found within max_radius
        return None

    def is_safe_for_path(self, mx: int, my: int,
                         clearance_cells: int) -> bool:
        """
        A cell is safe for path planning if:
        - it is free
        - there is no occupied cell within `clearance_cells` in grid space.
        """
        v_center = self.cell_value(mx, my)
        if v_center is None:
            return False
        # Only treat >=50 as obstacle; allow free (0) and unknown (-1)
        if v_center >= 50:
            return False
        
        if self.map_width is None or self.map_height is None:
            return False

        for dx in range(-clearance_cells, clearance_cells + 1):
            for dy in range(-clearance_cells, clearance_cells + 1):
                nx = mx + dx
                ny = my + dy
                if self.map_index(nx, ny) is None:
                    continue
                v = self.cell_value(nx, ny)
                if v is None:
                    continue
                if v >= 50:     # treat as obstacle
                    return False

        return True

    def astar_plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        allow_diagonal: bool = True
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Grid-based A* from start (mx,my) to goal (mx,my) on the occupancy grid.

        Returns:
            List of (mx, my) cells including start and goal, or None if no path.
        """
        if not self.map_received or self.map_width is None or self.map_height is None:
            self.get_logger().warn('A*: map not ready.')
            return None

        sx, sy = start
        gx, gy = goal

        if not self.is_traversable(sx, sy):
            alt = self.find_nearest_traversable(sx, sy, max_radius=self.NEAREST_TRAVERSABLE_RADIUS_CELLS)
            if alt is None:
                self.get_logger().warn(
                    f'A*: start cell {start} is not traversable and no free neighbor found.'
                )
                return None
            else:
                self.get_logger().info(
                    f'A*: start cell {start} not traversable, using nearest free cell {alt} as start.'
                )
                sx, sy = alt
                start = (sx, sy)

        if not self.is_traversable(gx, gy):
            alt_goal = self.find_nearest_traversable(gx, gy, max_radius=self.NEAREST_TRAVERSABLE_RADIUS_CELLS)
            if alt_goal is None:
                self.get_logger().warn(
                    f'A*: goal cell {goal} is not traversable and no nearby free cell found.'
                )
                return None
            else:
                self.get_logger().info(
                    f'A*: goal cell {goal} not traversable, using nearest free cell {alt_goal} as goal.'
                )
                gx, gy = alt_goal
                goal = (gx, gy)

        # Neighbor motions: dx, dy, cost
        if allow_diagonal:
            neighbors = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (-1, -1, math.sqrt(2)),
            ]
        else:
            neighbors = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
            ]

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            ax, ay = a
            bx, by = b
            return math.hypot(ax - bx, ay - by)

        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, (sx, sy)))

        g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        closed_set: Set[Tuple[int, int]] = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            closed_set.add(current)

            if current == (gx, gy):
                # Reconstruct path
                return self.reconstruct_path(came_from, start, goal)

            cx, cy = current

            for dx, dy, step_cost in neighbors:
                nx = cx + dx
                ny = cy + dy
                neighbor_cell = (nx, ny)

                # Bounds & traversability check
                if self.map_index(nx, ny) is None:
                    continue
                if not self.is_traversable(nx, ny):
                    continue
                if neighbor_cell in closed_set:
                    continue

                tentative_g = g_score[current] + step_cost

                if neighbor_cell not in g_score or tentative_g < g_score[neighbor_cell]:
                    g_score[neighbor_cell] = tentative_g
                    came_from[neighbor_cell] = current
                    f_score = tentative_g + heuristic(neighbor_cell, (gx, gy))
                    heapq.heappush(open_heap, (f_score, neighbor_cell))

        # No path
        self.get_logger().warn(f'A*: no path found from {start} to {goal}.')
        return None

    @staticmethod
    def reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary."""
        current = goal
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # -------------------------------------------------------------------------
    # Frontier detection & clustering (Stage 4)
    # -------------------------------------------------------------------------

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _prune_blocked_goals(self):
        if not self.blocked_goals:
            return
        now = self._now_sec()
        # pop from left while expired
        while self.blocked_goals and (now - self.blocked_goals[0][2] > self.blocked_goal_ttl_s):
            mx, my, _ = self.blocked_goals.popleft()
            self.blocked_goals_set.discard((mx, my))

    def add_blocked_goal(self, mx: int, my: int):
        self._prune_blocked_goals()
        key = (mx, my)
        if key in self.blocked_goals_set:
            return
        self.blocked_goals.append((mx, my, self._now_sec()))
        self.blocked_goals_set.add(key)

        # enforce max size
        while len(self.blocked_goals) > self.blocked_goals_max:
            ox, oy, _ = self.blocked_goals.popleft()
            self.blocked_goals_set.discard((ox, oy))

    def is_blocked_goal_cell(self, mx: int, my: int) -> bool:
        self._prune_blocked_goals()
        return (mx, my) in self.blocked_goals_set

    def remember_inflated_frontier(self, cell: Tuple[int, int]):
        """
        Record a frontier-related cell whose cluster was rejected because
        no traversable goal could be found near it (inflation / clearance).
        """
        if cell not in self.inflated_filtered_frontiers:
            self.inflated_filtered_frontiers.append(cell)

    def is_frontier_cell(self, mx: int, my: int) -> bool:
        """
        A frontier cell is:
          - free
          - and has at least one 4-connected neighbor that is unknown.
        """
        if not self.is_free(mx, my):
            return False

        # 4-connected neighbors
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in neighbors:
            nx = mx + dx
            ny = my + dy
            if self.map_index(nx, ny) is None:
                continue
            if self.is_unknown(nx, ny):
                return True
        return False

    def is_safe_known_cell(self, mx: int, my: int,
                           clearance_cells: Optional[int] = None) -> bool:
        """
        A 'safe known' cell:
        - is free
        - is NOT a frontier (so it is fully in explored space)
        - has no occupied cells within the given clearance radius.
        """
        if clearance_cells is None:
            clearance_cells = self.frontier_clearance_current

        # must be free
        if not self.is_free(mx, my):
            return False

        # avoid frontier cells; we want already-explored interior cells
        if self.is_frontier_cell(mx, my):
            return False

        if self.map_width is None or self.map_height is None:
            return False

        for dx in range(-clearance_cells, clearance_cells + 1):
            for dy in range(-clearance_cells, clearance_cells + 1):
                nx = mx + dx
                ny = my + dy
                if self.map_index(nx, ny) is None:
                    continue
                v = self.cell_value(nx, ny)
                if v is None:
                    continue
                if v >= 50:  # occupied
                    return False

        return True

    def find_frontier_cells(self) -> List[Tuple[int, int]]:
        """
        Scan the map and return a list of all frontier cells.
        """
        frontiers: List[Tuple[int, int]] = []
        if not self.map_received or self.map_data is None:
            return frontiers

        for my in range(self.map_height):
            for mx in range(self.map_width):
                if self.is_frontier_cell(mx, my):
                    frontiers.append((mx, my))

        return frontiers

    def cluster_frontiers(self, frontier_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        Cluster frontier cells using region growing with a radius in grid cells.
        Returns a list of clusters, each cluster is a list of (mx, my).
        """
        clusters: List[List[Tuple[int, int]]] = []
        if not frontier_cells:
            return clusters

        frontier_set: Set[Tuple[int, int]] = set(frontier_cells)
        visited: Set[Tuple[int, int]] = set()

        # neighbors within a square radius in Chebyshev distance
        R = self.CLUSTER_RADIUS_CELLS
        neighbor_offsets = [
            (dx, dy)
            for dx in range(-R, R + 1)
            for dy in range(-R, R + 1)
            if not (dx == 0 and dy == 0)
        ]

        for cell in frontier_cells:
            if cell in visited:
                continue

            cluster: List[Tuple[int, int]] = []
            queue: List[Tuple[int, int]] = [cell]
            visited.add(cell)

            while queue:
                cx, cy = queue.pop(0)
                cluster.append((cx, cy))

                for dx, dy in neighbor_offsets:
                    nx = cx + dx
                    ny = cy + dy
                    neighbor = (nx, ny)
                    if neighbor in frontier_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    def is_safe_frontier_cell(self, mx: int, my: int,
                              clearance_cells: Optional[int] = None) -> bool:
        """
        A 'safe' frontier cell is free AND has no occupied cells
        within a given radius in grid space.
        """
        if clearance_cells is None:
            clearance_cells = self.frontier_clearance_current

        if not self.is_free(mx, my):
            return False

        if self.map_width is None or self.map_height is None:
            return False

        for dx in range(-clearance_cells, clearance_cells + 1):
            for dy in range(-clearance_cells, clearance_cells + 1):
                nx = mx + dx
                ny = my + dy
                if self.map_index(nx, ny) is None:
                    continue
                v = self.cell_value(nx, ny)
                if v is None:
                    continue
                if v >= 50:
                    return False

        return True

    def backoff_from_obstacle(
        self,
        mx: int,
        my: int,
        max_back_cells: int = 5,
        clearance_cells: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Given a (possibly unsafe) frontier cell (mx, my),
        step back along the line from robot -> frontier
        by up to max_back_cells, and return the first cell
        that has enough clearance (is_safe_frontier_cell).

        Returns (bx, by) or None if no suitable cell found.
        """
        if clearance_cells is None:
            clearance_cells = self.frontier_clearance_current

        if self.robot_mx is None or self.robot_my is None:
            return None

        dx = mx - self.robot_mx
        dy = my - self.robot_my

        # Degenerate case: frontier is at the robot
        if dx == 0 and dy == 0:
            return None

        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None

        # unit direction from robot -> frontier
        ux = dx / length
        uy = dy / length

        for step in range(1, max_back_cells + 1):
            bx = int(round(mx - ux * step))
            by = int(round(my - uy * step))

            if self.map_index(bx, by) is None:
                continue

            # we want a free cell with clearance
            if self.is_safe_frontier_cell(bx, by, clearance_cells):
                return (bx, by)

        return None

    def backoff_to_known_safe_cell(
        self,
        mx: int,
        my: int,
        max_back_cells: int = 4,
        clearance_cells: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Given a frontier cell (mx, my), step back along the line from
        robot -> frontier by up to max_back_cells, and return the first
        cell that is:
        - free
        - not a frontier (already explored)
        - has clearance from obstacles (is_safe_known_cell).
        """
        if clearance_cells is None:
            clearance_cells = self.frontier_clearance_current

        if self.robot_mx is None or self.robot_my is None:
            return None

        dx = mx - self.robot_mx
        dy = my - self.robot_my

        # Degenerate case: frontier at robot
        if dx == 0 and dy == 0:
            return None

        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None

        # unit direction from robot -> frontier
        ux = dx / length
        uy = dy / length

        for step in range(1, max_back_cells + 1):
            bx = int(round(mx - ux * step))
            by = int(round(my - uy * step))

            if self.map_index(bx, by) is None:
                continue

            if self.is_safe_known_cell(bx, by, clearance_cells):
                return (bx, by)

        return None

    def choose_frontier_goal(self, clusters: List[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """
        Choose a frontier-related cell using a *geometric* distance band instead of
        repeatedly running A* for scoring.

        Priority:
          1) Cells whose straight-line distance (in cells) to the robot lies in
             [FRONTIER_PATH_LEN_MIN_CELLS, FRONTIER_PATH_LEN_MAX_CELLS], with
             smallest visit count and then largest distance (farthest in band).
          2) All other "safe" candidates (same filters) with smallest visit
             count and then smallest distance.
          3) NEW: if no safe candidate exists at all, fall back to the nearest
             raw frontier cell (closest to the robot), even if it doesn't yet
             pass clearance/min-distance; later stages will try to adjust or
             reject it.
        """
        if not clusters:
            return None
        if self.robot_mx is None or self.robot_my is None:
            return None

        band_min = self.FRONTIER_PATH_LEN_MIN_CELLS
        band_max = self.FRONTIER_PATH_LEN_MAX_CELLS

        band_min2 = band_min * band_min
        band_max2 = band_max * band_max

        # Each entry: (visits, dist2, mx, my)
        band_candidates: List[Tuple[int, int, int, int]] = []
        fb_candidates: List[Tuple[int, int, int, int]] = []

        # NEW: raw per-cluster fallback candidates (no clearance/min-dist filters)
        raw_frontiers: List[Tuple[int, int, int, int]] = []

        for cluster in clusters:
            if not cluster:
                continue

            # 1) pick the frontier cell in this cluster that is nearest (Euclidean) to the robot
            best_cell = None
            best_d2 = float("inf")

            for (mx, my) in cluster:
                dx = mx - self.robot_mx
                dy = my - self.robot_my
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_cell = (mx, my)

            if best_cell is None:
                continue

            bx, by = best_cell

            # NEW: skip clusters whose representative is in a blocked-goal region
            if self.is_blocked_goal_cell(bx, by):
                self.get_logger().info(
                    f'Skipping frontier cluster whose representative ({bx}, {by}) '
                    f'is inside a blocked-goal region.'
                )
                continue

            # record raw representative for "desperate" fallback later
            raw_visits = self.frontier_visit_counts.get((bx, by), 0)
            raw_frontiers.append((raw_visits, best_d2, bx, by))

            # From here on, we try to build a *safe* candidate

            tx, ty = bx, by

            # 2) back off slightly into known-safe territory if possible
            backed = self.backoff_to_known_safe_cell(
                tx,
                ty,
                max_back_cells=4,
                clearance_cells=self.frontier_clearance_current,
            )
            if backed is not None:
                tx, ty = backed

            # 3) must be traversable with inflation
            if not self.is_traversable(tx, ty):
                continue

            dx = tx - self.robot_mx
            dy = ty - self.robot_my
            dist2 = dx * dx + dy * dy

            # enforce a hard minimum distance from the robot
            if dist2 < self.MIN_GOAL_DIST_CELLS * self.MIN_GOAL_DIST_CELLS:
                continue

            visits = self.frontier_visit_counts.get((tx, ty), 0)

            # Use d^2 instead of d to avoid sqrt; band boundaries are squared too.
            if band_min2 <= dist2 <= band_max2:
                band_candidates.append((visits, dist2, tx, ty))
            else:
                fb_candidates.append((visits, dist2, tx, ty))

        # 4) pick according to priority rules

        # First: candidates whose distance lies in [band_min, band_max],
        # with smallest visit count and *largest* distance in that band.
        if band_candidates:
            band_candidates.sort(key=lambda t: (t[0], -t[1]))  # (visits asc, dist2 desc)
            _, _, gx, gy = band_candidates[0]
            return (gx, gy)

        # Second: all other "safe" candidates, with smallest visit count and *smallest* distance.
        if fb_candidates:
            fb_candidates.sort(key=lambda t: (t[0], t[1]))  # (visits asc, dist2 asc)
            _, _, gx, gy = fb_candidates[0]
            return (gx, gy)

        # Third (NEW): if no safe candidate exists at all, fall back to the nearest
        # raw frontier representative (closest to robot), ignoring safety filters.
        if raw_frontiers:
            # NEW: ignore raw representatives inside blocked-goal regions
            usable_raw = [
                t for t in raw_frontiers
                if not self.is_blocked_goal_cell(t[2], t[3])
            ]

            if usable_raw:
                usable_raw.sort(key=lambda t: (t[0], t[1]))  # (visits asc, dist2 asc)
                _, _, gx, gy = usable_raw[0]
                self.get_logger().warn(
                    f'choose_frontier_goal: no safe candidates; '
                    f'falling back to raw frontier cell ({gx}, {gy}).'
                )
                return (gx, gy)

        # Truly nothing usable
        return None

        # Truly nothing usable
        return None

    # -------------------------------------------------------------------------
    # Visualization helpers (Stage 4)
    # -------------------------------------------------------------------------

    def publish_frontier_markers(self):
        """
        Publish one marker per cluster (at its centroid) and
        a special marker for the chosen frontier goal.
        """
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # cluster centroids
        cluster_id = 0
        for cluster in self.frontier_clusters:
            if not cluster:
                continue
            # centroid in grid coordinates
            sum_x = sum(mx for mx, _ in cluster)
            sum_y = sum(my for _, my in cluster)
            cx = sum_x / len(cluster)
            cy = sum_y / len(cluster)

            world = self.map_to_world(int(round(cx)), int(round(cy)))
            if world is None:
                continue
            wx, wy = world

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'frontier_clusters'
            m.id = cluster_id
            cluster_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            # scale: roughly one cell
            size = self.map_resolution * 2.0 if self.map_resolution else 0.1
            m.scale.x = size
            m.scale.y = size
            m.scale.z = size
            # blue-ish
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 0.8
            m.lifetime = Duration(sec=1, nanosec=0)

            markers.markers.append(m)

        # chosen goal marker
        if self.frontier_goal is not None:
            gx, gy = self.frontier_goal
            world = self.map_to_world(gx, gy)
            if world is not None:
                wx, wy = world

                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp = now
                m.ns = 'frontier_goal'
                m.id = 0
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = wx
                m.pose.position.y = wy
                m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                size = self.map_resolution * 3.0 if self.map_resolution else 0.15
                m.scale.x = size
                m.scale.y = size
                m.scale.z = size
                # red
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 1.0
                m.lifetime = Duration(sec=1, nanosec=0)

                markers.markers.append(m)

        self.frontier_markers_pub.publish(markers)

    def publish_global_plan(self):
        """
        Publish the current_path as a nav_msgs/Path on /global_plan.
        """
        if not self.current_path:
            return

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for (mx, my) in self.current_path:
            world = self.map_to_world(mx, my)
            if world is None:
                continue
            wx, wy = world
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # yaw = 0 for visualization
            path_msg.poses.append(pose)

        self.global_plan_pub.publish(path_msg)

    # -------------------------------------------------------------------------
    # Path following & obstacle handling (Stage 5 + grid-based)
    # -------------------------------------------------------------------------

    def set_current_path(self, path_cells: List[Tuple[int, int]]):
        """Store new current path, reset indices and convert to world."""
        self.current_path = path_cells
        self.current_path_world = []
        if path_cells:
            for (mx, my) in path_cells:
                world = self.map_to_world(mx, my)
                if world is not None:
                    self.current_path_world.append(world)
        self.path_idx = 0
        self.pid_speed.reset()
        self.pid_heading.reset()
        self.speed_hist.clear()
        self.last_ctrl_time = None

    def clear_current_path(self):
        """Clear path and stop the robot."""
        self.current_path = None
        self.current_path_world = None
        self.path_idx = None
        self.publish_cmd_vel(0.0, 0.0)

    def publish_cmd_vel(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)

    def is_path_still_valid(self, look_ahead_cells: int = 20) -> bool:
        """
        Check if the upcoming part of the current path remains safe
        in the occupancy grid for *local* execution.

        We now re-check full traversability (inflated safety), not just
        "not occupied", so if an obstacle appears near the path, the
        path becomes invalid and we replan.
        """
        if self.current_path is None or self.path_idx is None:
            return False

        start_idx = max(0, self.path_idx)
        end_idx = min(len(self.current_path), start_idx + look_ahead_cells)

        for i in range(start_idx, end_idx):
            mx, my = self.current_path[i]

            # If the cell is outside the current map, path is no longer valid.
            if self.map_index(mx, my) is None:
                return False

            # require traversability (free + clearance), not just non-occupied
            if not self.is_traversable(mx, my):
                return False

        return True

    def _extend_collinear_segment(self, start_idx, curr_path_wrld, tol=1e-6):
        """
        From start_idx, look forward and find the longest contiguous run of points
        that lie on the same infinite line defined by waypoints[start_idx] and
        waypoints[start_idx+1]. Returns the last index of that collinear run.
        """
        N = len(curr_path_wrld)
        i = start_idx
        if i >= N-1:
            return i  # nothing to extend

        p0 = curr_path_wrld[i]
        d = curr_path_wrld[i+1] - p0
        # If the direction is (near) zero (duplicate points), just stop here.
        if np.allclose(d, 0, atol=tol):
            return i+1

        # Cross-product test for collinearity with all subsequent points
        rel = curr_path_wrld[i+1:] - p0                    # shape (N-i-1, 2)
        cross = rel[:, 0] * d[1] - rel[:, 1] * d[0]   # scalar 2D cross
        mask = np.isclose(cross, 0.0, atol=tol)

        # We only want the longest *contiguous* True prefix
        if mask.all():
            prefix_len = len(mask)
        else:
            # index of first False in mask (prefix length of True)
            prefix_len = np.argmax(~mask)

        # Last collinear index = i + prefix_len (since mask starts at i+1)
        return i + prefix_len
    
    def choose_next(self, rx, ry):
        curr_path_wrld = np.array(self.current_path_world)
        N = len(curr_path_wrld)

        if self.path_idx is None:
            self.path_idx = 0

        # If we are already at (or beyond) the final index, just stick to it
        if self.path_idx >= N - 1:
            return N - 1

        # Consider only strictly future indices
        idxs = np.arange(self.path_idx + 1, N)

        dx = curr_path_wrld[idxs, 0] - rx
        dy = curr_path_wrld[idxs, 1] - ry
        distances = np.sqrt(dx**2 + dy**2)

        # Closest future index
        rel = np.argmin(distances)
        next_idx_o = idxs[rel]

        # Extend collinear segment forward from that index
        next_idx = self._extend_collinear_segment(next_idx_o, curr_path_wrld)

        self.get_logger().info(f'[**DEBUG**] Robot: {rx}, {ry}; Curr Path World: {curr_path_wrld}; Indexes: {idxs}; Distances: {distances}; Next Index Before Extend: {next_idx_o}; Next Index After Extend: {next_idx}')

        self.last_idx = next_idx
        if self.last_idx == N - 1:
            self.final_idx = True

        return next_idx
    
    def advance_waypoint_if_reached(self, rx: float, ry: float, waypoint_tol: float):
        """
        Strictly follow the path in order:
        - Only move to the next waypoint when the current one is within waypoint_tol.
        - Never jump ahead to the 'closest' waypoint.
        """
        if self.current_path_world is None or self.path_idx is None:
            return

        N = len(self.current_path_world)
        # Progress through waypoints in order, but don't skip any
        while self.path_idx < N - 1:
            wx, wy = self.current_path_world[self.path_idx]
            dist = math.hypot(wx - rx, wy - ry)
            if dist < waypoint_tol:
                self.path_idx += 1
            else:
                break

    def follow_current_path(self):
        """
        Follow the current path using PID on speed and heading.
        Uses BOTH:
          - /scan front distance
          - occupancy grid along the path
        to decide when to stop and replan.
        """
        # pick a tolerance smaller than a cell
        cell = self.map_resolution if self.map_resolution is not None else 0.05
        waypoint_tol = 0.5 * cell  # for intermediate waypoints
        goal_tol     = 0.8 * cell  # for final goal
        min_goal_speed = 0.02

        if (self.current_path_world is None or
                not self.current_path_world or
                self.path_idx is None or
                not self.robot_pose_received or
                self.robot_x is None or
                self.robot_y is None or
                self.robot_yaw is None):
            # Nothing to do or no pose: stop for safety
            self.publish_cmd_vel(0.0, 0.0)
            return

        # --- UNION OBSTACLE CRITERIA ---

        # 1) /scan-based obstacle
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # If we are currently escaping, keep doing it until timeout
        if self.escape_active and now_sec < self.escape_until:
            self.publish_cmd_vel(0.0, self.escape_dir * self.escape_w)
            return
        else:
            self.escape_active = False

        # --- Obstacle logic (smarter) ---
        if self.front_range_filt is not None and self.front_range_filt < self.obstacle_stop_dist:
            # Pick turn direction by whichever side has more clearance
            l = self.min_left_range if self.min_left_range is not None else -1.0
            r = self.min_right_range if self.min_right_range is not None else -1.0

            if l == -1.0 and r == -1.0:
                self.escape_dir = 1.0
            else:
                self.escape_dir = 1.0 if l >= r else -1.0

            self.escape_attempts += 1

            # Try a few escape turns BEFORE giving up on this goal/path
            if self.escape_attempts <= self.escape_attempt_limit:
                self.escape_active = True
                self.escape_until = now_sec + self.escape_duration_s
                self.get_logger().warn(
                    f"Obstacle close (front={self.front_range_filt:.2f}m). "
                    f"Escape-turn attempt {self.escape_attempts}/{self.escape_attempt_limit} "
                    f"dir={'L' if self.escape_dir > 0 else 'R'}."
                )
                self.publish_cmd_vel(0.0, self.escape_dir * self.escape_w)
                return

            # If we STILL can't clear it after multiple turns, then block & replan
            if self.frontier_goal is not None:
                gx, gy = self.frontier_goal
                self.add_blocked_goal(gx, gy)  # upgraded in Fix 2 below
                self.get_logger().warn(
                    f"Escape failed; blocking goal region around {self.frontier_goal} and replanning."
                )

            self.escape_attempts = 0
            self.clear_current_path()
            return

        # If we got here, no immediate obstacle → reset escape counter
        self.escape_attempts = 0

        # 2) map-based obstacle along path (traversability including clearance)
        if not self.is_path_still_valid():
            self.get_logger().warn(
                'Current path cells lost traversability (new obstacles / reduced clearance). '
                'Clearing path for replanning.',
                throttle_duration_sec=1.0
            )
            self.clear_current_path()
            return

        # --- NORMAL PATH FOLLOWING BELOW ---

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # dt for PID
        if self.last_ctrl_time is None:
            dt = 0.1  # ~10 Hz fallback
        else:
            dt = max(1e-3, now_sec - self.last_ctrl_time)
        self.last_ctrl_time = now_sec

        rx = self.robot_x
        ry = self.robot_y
        ryaw = self.robot_yaw

        # Update speed history for speed estimation
        self.speed_hist.append((now_sec, rx, ry))
        if len(self.speed_hist) > 5:
            self.speed_hist.pop(0)

        # Estimate current speed (scalar)
        if len(self.speed_hist) >= 2:
            t_prev, x_prev, y_prev = self.speed_hist[-2]
            dt_speed = max(1e-3, now_sec - t_prev)
            dist = math.hypot(rx - x_prev, ry - y_prev)
            speed_curr = dist / dt_speed
        else:
            speed_curr = 0.0

        # Advance waypoint index when close to current waypoint
        while self.path_idx < len(self.current_path_world) - 1:
            gx, gy = self.current_path_world[self.path_idx]
            dist_to_wp = math.hypot(gx - rx, gy - ry)
            if dist_to_wp < waypoint_tol:
                self.path_idx += 1
            else:
                break

        # Check final goal
        gx, gy = self.current_path_world[self.path_idx]
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        
        if (self.path_idx == len(self.current_path_world) - 1 and
            dist_to_goal < goal_tol and
            speed_curr < min_goal_speed):

            if self.frontier_goal is not None:
                old_count = self.frontier_visit_counts.get(self.frontier_goal, 0)
                new_count = old_count + 1
                self.frontier_visit_counts[self.frontier_goal] = new_count

                self.get_logger().info(
                    f'Reached frontier goal {self.frontier_goal}. '
                    f'Visit count now = {new_count}. Clearing path.',
                    throttle_duration_sec=2.0
                )
            else:
                self.get_logger().info(
                    'Reached current path goal. Stopping and clearing path.',
                    throttle_duration_sec=2.0
                )

            # Reset both clearances back to their original safe values
            self.frontier_clearance_current = self.FRONTIER_CLEARANCE_CELLS
            self.path_clearance_current = self.PATH_CLEARANCE_CELLS

            self.clear_current_path()
            return

        # Desired heading
        dx = gx - rx
        dy = gy - ry
        desired_yaw = math.atan2(dy, dx)
        heading_err = wrap_angle(desired_yaw - ryaw)

        # Deadband to eliminate micro-oscillations
        if abs(heading_err) < self.heading_deadband:
            heading_err = 0.0

        # stronger turn penalty: drop faster as heading_err grows
        turn_ratio = max(0.05, 1.0 - (abs(heading_err) / 1.2))  # 1.2 rad ~ 69 deg

        # Angular velocity from heading PID
        heading_cmd = self.pid_heading.step(heading_err, dt)

        # Heading factor: penalize misalignment but not too aggressively
        heading_factor = max(0.0, math.cos(heading_err))

        if dist_to_goal > self.slow_down_dist:
            speed_goal = self.speed_max * heading_factor * turn_ratio
        else:
            base = self.speed_max * (dist_to_goal / max(self.slow_down_dist, 1e-3))
            speed_goal = max(self.speed_min * heading_factor,
                             base * heading_factor) * turn_ratio

        # --- Obstacle-aware speed scaling (smooth slowdown before hard-stop) ---
        d = self.front_range_filt if self.front_range_filt is not None else 10.0

        slow_radius = 0.90  # start slowing when obstacle within this range (tune 0.7–1.2)
        stop_dist   = self.obstacle_stop_dist  # your existing hard stop threshold

        if d <= stop_dist:
            obs_scale = 0.0
        elif d >= slow_radius:
            obs_scale = 1.0
        else:
            obs_scale = (d - stop_dist) / (slow_radius - stop_dist)

        speed_goal *= obs_scale

        speed_goal = max(0.0, speed_goal)

        # Speed PID on (speed_goal - current_estimated_speed)
        speed_err = speed_goal - speed_curr
        speed_cmd = self.pid_speed.step(speed_err, dt)

        # Extra safety: if very misaligned, stop and only rotate
        if abs(heading_err) > 0.7:  # allow bigger error before freezing linear speed
            speed_cmd = 0.0

        # never go backwards, clamp to limits
        speed_cmd = max(0.0, min(self.speed_max, speed_cmd))

        # --- Slew-rate limit linear speed (prevents jerky accel that causes instability) ---
        dv_up   = self.max_accel * dt
        dv_down = self.max_decel * dt

        if speed_cmd > self.prev_speed_cmd + dv_up:
            speed_cmd = self.prev_speed_cmd + dv_up
        elif speed_cmd < self.prev_speed_cmd - dv_down:
            speed_cmd = self.prev_speed_cmd - dv_down

        self.prev_speed_cmd = speed_cmd

        self.publish_cmd_vel(speed_cmd, heading_cmd)

    # -------------------------------------------------------------------------
    # Timer callback - behavior loop (Stage 6)
    # -------------------------------------------------------------------------

    def which_cluster_contains(
        self,
        clusters: List[List[Tuple[int, int]]],
        cell: Tuple[int, int],
        max_dist_cells: int = 2
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the cluster whose cells are closest to 'cell', as long as the
        minimum distance is within max_dist_cells (in grid units).
        """
        mx, my = cell
        best_cluster = None
        best_dist2 = float('inf')

        for cluster in clusters:
            for (cx, cy) in cluster:
                dx = cx - mx
                dy = cy - my
                dist2 = dx * dx + dy * dy
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_cluster = cluster

        # Only accept if reasonably close to some cluster cell
        if best_cluster is not None and best_dist2 <= max_dist_cells * max_dist_cells:
            return best_cluster

        return None

    def find_alternative_goal_for_cluster(
        self,
        desired_goal: Tuple[int, int],
        cluster: List[Tuple[int, int]],
        neighborhood_radius: int = 4,
    ) -> Optional[Tuple[int, int]]:
        """
        Given a desired frontier-related goal (mx,my) and its cluster,
        search for the nearest cell that:
          - is traversable (is_traversable = free + PATH_CLEARANCE_CELLS)
          - is not in a blocked region
          - is at least MIN_GOAL_DIST_CELLS away from the robot

        This is the "project the goal to a valid nearby cell" logic.
        """
        if self.robot_mx is None or self.robot_my is None:
            return None

        gx, gy = desired_goal
        candidates: Set[Tuple[int, int]] = set()

        # 1) Start with the desired goal itself
        candidates.add((gx, gy))

        # 2) Try backoff_to_known_safe_cell from the desired goal
        backed = self.backoff_to_known_safe_cell(
            gx, gy,
            max_back_cells=6,
            clearance_cells=self.frontier_clearance_current,
        )
        if backed is not None:
            candidates.add(backed)

        # 3) Add all cells in the cluster and their backed versions
        for (mx, my) in cluster:
            candidates.add((mx, my))

            backed2 = self.backoff_to_known_safe_cell(
                mx, my,
                max_back_cells=6,
                clearance_cells=self.frontier_clearance_current,
            )
            if backed2 is not None:
                candidates.add(backed2)

        # 4) Add a neighborhood around the cluster
        for (cx, cy) in cluster:
            for dx in range(-neighborhood_radius, neighborhood_radius + 1):
                for dy in range(-neighborhood_radius, neighborhood_radius + 1):
                    nx = cx + dx
                    ny = cy + dy
                    if self.map_index(nx, ny) is None:
                        continue
                    candidates.add((nx, ny))

        valid_scored: List[Tuple[float, Tuple[int, int]]] = []

        for (mx, my) in candidates:
            # Must be inside map
            if self.map_index(mx, my) is None:
                continue

            # Respect traversability gate
            if not self.is_traversable(mx, my):
                continue

            # Skip blocked regions
            if self.is_blocked_goal_cell(mx, my):
                continue

            # Respect minimum distance from robot
            dx = mx - self.robot_mx
            dy = my - self.robot_my
            if dx * dx + dy * dy < self.MIN_GOAL_DIST_CELLS ** 2:
                continue

            # Score: prefer staying close to the originally desired goal
            ddx = mx - gx
            ddy = my - gy
            dist2_to_desired = ddx * ddx + ddy * ddy
            valid_scored.append((dist2_to_desired, (mx, my)))

        if not valid_scored:
            return None

        # Pick the valid candidate closest to the original goal
        valid_scored.sort(key=lambda x: x[0])
        return valid_scored[0][1]

    def compute_far_alternative_goal(
        self,
        mx: int,
        my: int,
        max_offset_cells: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        For a frontier cell that was rejected due to inflation / clearance,
        search in axis and diagonal directions for a traversable cell that is
        farther away from the original inflated area.

        This is deliberately cheap: we only walk along a few rays.
        """
        if self.map_width is None or self.map_height is None:
            return None

        if max_offset_cells is None:
            max_offset_cells = self.FRONTIER_FAR_ALT_MAX_OFFSET_CELLS

        directions = [
            (1, 0),  (-1, 0),
            (0, 1),  (0, -1),
            (1, 1),  (1, -1),
            (-1, 1), (-1, -1),
        ]

        for step in range(self.FRONTIER_FAR_ALT_MIN_OFFSET_CELLS,
                          max_offset_cells + 1):
            for dx, dy in directions:
                ax = mx + dx * step
                ay = my + dy * step

                if self.map_index(ax, ay) is None:
                    continue

                # Must be traversable (respects PATH_CLEARANCE_CELLS and unknown-as-free change)
                if not self.is_traversable(ax, ay):
                    continue

                # Respect blocked-goal exclusion
                if self.is_blocked_goal_cell(ax, ay):
                    continue

                # Keep some minimum distance from robot
                if self.robot_mx is not None and self.robot_my is not None:
                    ddx = ax - self.robot_mx
                    ddy = ay - self.robot_my
                    if ddx * ddx + ddy * ddy < self.MIN_GOAL_DIST_CELLS ** 2:
                        continue

                return (ax, ay)

        return None

    def try_inflated_backlog_goals(self) -> bool:
        """
        Third-tier: try to use 'far' alternative goals for frontiers whose
        clusters were rejected due to inflation / clearance.
        Returns True if we successfully set a new path.
        """
        if self.robot_mx is None or self.robot_my is None:
            return False
        if not self.inflated_filtered_frontiers:
            return False

        # Work on a copy so we can safely pop while iterating
        for idx, (fx, fy) in enumerate(list(self.inflated_filtered_frontiers)):
            alt = self.compute_far_alternative_goal(fx, fy)
            if alt is None:
                continue

            gx, gy = alt
            path = self.astar_plan((self.robot_mx, self.robot_my), (gx, gy))
            if path is None:
                continue

            smoothed = self.smooth_path(path)

            self.get_logger().info(
                f'A* to far-alt frontier: raw_len={len(path)}, '
                f'start={smoothed[0]}, end={smoothed[-1]}, '
                f'raw_frontier={(fx, fy)}, alt_goal={alt}'
            )

            # Use the alternate as the actual goal; remember original
            self.frontier_goal = alt
            self.last_frontier_goal = (fx, fy)
            self.set_current_path(smoothed)

            # Remove this frontier from backlog now that we used it
            try:
                self.inflated_filtered_frontiers.pop(idx)
            except IndexError:
                pass

            return True

        # Could not find any usable far-alt goal
        return False

    def timer_cb(self):
        """
        Behavior loop:
          - WAIT_FOR_MAP: wait until map & pose ready, then go to EXPLORE.
          - EXPLORE: detect frontiers, pick goals, plan paths, follow paths.
          - DONE: no frontiers left, stop and chill.
        """
        self.timer_counter += 1

        # 1) Update map stats occasionally
        if self.map_received and self.map_data is not None:
            if self.timer_counter % 50 == 0:  # every ~5s at 10 Hz
                unknown = free = occupied = 0
                for v in self.map_data:
                    if v == -1:
                        unknown += 1
                    elif v == 0:
                        free += 1
                    elif v >= 50:
                        occupied += 1
                total = len(self.map_data)
                self.get_logger().info(
                    f'Map stats: total={total}, unknown={unknown}, free={free}, occupied={occupied}'
                )
        else:
            self.get_logger().info('Waiting for /map...', throttle_duration_sec=2.0)

        # 2) Update robot pose from TF
        self.update_robot_pose_from_tf()
        if self.robot_pose_received and self.robot_mx is not None and self.robot_my is not None:
            if self.timer_counter % 20 == 0:  # every ~2s
                self.get_logger().info(
                    f'Robot pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, '
                    f'yaw={math.degrees(self.robot_yaw):.1f} deg, '
                    f'grid=({self.robot_mx}, {self.robot_my})'
                )
        else:
            self.get_logger().info(
                'Waiting for TF transform map->base_footprint or robot outside map...',
                throttle_duration_sec=2.0
            )

        # ---------------------------
        # State machine
        # ---------------------------

        # WAIT_FOR_MAP: map + pose must be ready, plus a short warmup
        if self.state == "WAIT_FOR_MAP":
            if (self.map_received and self.robot_pose_received and
                    self.robot_mx is not None and self.robot_my is not None and
                    self.timer_counter > 30):
                self.get_logger().info('Map & pose ready. Switching to EXPLORE.')
                self.state = "EXPLORE"
            else:
                self.publish_cmd_vel(0.0, 0.0)
                return

        # DONE: just stop and do nothing else
        if self.state == "DONE":
            self.publish_cmd_vel(0.0, 0.0)
            return

        # EXPLORE:
        if self.state == "EXPLORE":
            # 3) Frontier detection & clustering (every N steps)
            if self.timer_counter % 3 == 0:  # every ~0.3s
                frontier_cells = self.find_frontier_cells()
                clusters = self.cluster_frontiers(frontier_cells)
                self.frontier_clusters = clusters
                self.get_logger().info(
                    f'Frontiers: {len(frontier_cells)} cells, {len(clusters)} clusters.'
                )

                # --- Stage 6: stop exploring when there are NO frontiers ---
                if len(clusters) == 0:
                    self.get_logger().info(
                        'No frontier clusters left. Exploration DONE.'
                    )
                    self.clear_current_path()
                    self.state = "DONE"
                    self.publish_cmd_vel(0.0, 0.0)
                    return

                if self.current_path is None:
                    # Try with gradually relaxed frontier clearance:
                    # start from current value down to 0.
                    original_frontier_clearance = self.frontier_clearance_current
                    original_path_clearance = self.path_clearance_current
                    found_valid_goal = False

                    '''while self.frontier_clearance_current >= self.frontier_clearance_min and not found_valid_goal:
                        remaining_clusters = list(clusters)

                        self.get_logger().info(
                            f'Trying frontier selection with clearance={self.frontier_clearance_current} cells.'
                        )

                        while remaining_clusters and not found_valid_goal:
                            raw_goal = self.choose_frontier_goal(remaining_clusters)
                            if raw_goal is None:
                                # no more acceptable cells in these clusters
                                break

                            cluster_for_goal = self.which_cluster_contains(remaining_clusters, raw_goal)
                            if cluster_for_goal is None:
                                cluster_for_goal = [raw_goal]

                            alt_goal = self.find_alternative_goal_for_cluster(
                                desired_goal=raw_goal,
                                cluster=cluster_for_goal,
                                neighborhood_radius=4,
                            )

                            if alt_goal is None:
                                # No valid traversable cell near this frontier under current clearance
                                self.remember_inflated_frontier(raw_goal)
                                self.get_logger().warn(
                                    f'No valid alternative goal near frontier {raw_goal} '
                                    f'with clearance={self.frontier_clearance_current}; '
                                    f'skipping this cluster (added to inflated backlog).'
                                )
                                if cluster_for_goal in remaining_clusters:
                                    remaining_clusters.remove(cluster_for_goal)
                                continue

                            gx, gy = alt_goal
                            path = self.astar_plan((self.robot_mx, self.robot_my), (gx, gy))
                            if path is None:
                                # Projected goal still unreachable; mark this frontier as inflation-filtered
                                self.remember_inflated_frontier(raw_goal)
                                self.get_logger().warn(
                                    f'A*: no path to adjusted frontier goal {alt_goal} '
                                    f'for original {raw_goal} at clearance={self.frontier_clearance_current} '
                                    f'— skipping this cluster.'
                                )
                                if cluster_for_goal in remaining_clusters:
                                    remaining_clusters.remove(cluster_for_goal)
                                continue

                            smoothed = self.smooth_path(path)

                            self.get_logger().info(
                                f'A* to frontier: raw_len={len(path)}, smooth_len={len(smoothed)}, '
                                f'start={smoothed[0]}, end={smoothed[-1]}, '
                                f'raw_goal={raw_goal}, used_goal={alt_goal}, '
                                f'clearance={self.frontier_clearance_current}'
                            )

                            self.frontier_goal = alt_goal
                            self.last_frontier_goal = raw_goal
                            self.set_current_path(smoothed)
                            found_valid_goal = True

                        # If we still didn't find anything at this clearance,
                        # try far-alt backlog goals at the same clearance.
                        if not found_valid_goal:
                            used_backlog = self.try_inflated_backlog_goals()
                            if used_backlog:
                                found_valid_goal = True
                                break

                            if (self.frontier_clearance_current > self.frontier_clearance_min or
                                self.path_clearance_current > self.path_clearance_min):

                                new_frontier = max(self.frontier_clearance_min,
                                                    self.frontier_clearance_current - 1)
                                new_path = max(self.path_clearance_min,
                                                self.path_clearance_current - 1)

                                self.get_logger().warn(
                                    f'No valid frontier goals for clearance='
                                    f'(frontier={self.frontier_clearance_current}, path={self.path_clearance_current}). '
                                    f'Reducing to (frontier={new_frontier}, path={new_path}) and retrying.'
                                )

                                self.frontier_clearance_current = new_frontier
                                self.path_clearance_current = new_path
                            else:
                                # Already at minimum; break out.
                                break'''

                    # Try with gradually relaxed frontier clearance in a guaranteed staircase:
                    # (frontier = N, N-1, ..., min)
                    # and keep path clearance in lockstep.
                    for clearance in range(original_frontier_clearance, self.frontier_clearance_min - 1, -1):
                        # decrease frontier clearance step-by-step
                        self.frontier_clearance_current = clearance

                        # decrease path clearance by the same number of steps (lockstep)
                        step = original_frontier_clearance - clearance
                        self.path_clearance_current = max(self.path_clearance_min, original_path_clearance - step)

                        remaining_clusters = list(clusters)

                        self.get_logger().info(
                            f'Trying frontier selection with clearance={self.frontier_clearance_current} cells.'
                        )

                        while remaining_clusters and not found_valid_goal:
                            raw_goal = self.choose_frontier_goal(remaining_clusters)
                            if raw_goal is None:
                                break

                            cluster_for_goal = self.which_cluster_contains(remaining_clusters, raw_goal)
                            if cluster_for_goal is None:
                                cluster_for_goal = [raw_goal]

                            alt_goal = self.find_alternative_goal_for_cluster(
                                desired_goal=raw_goal,
                                cluster=cluster_for_goal,
                                neighborhood_radius=4,
                            )

                            if alt_goal is None:
                                self.remember_inflated_frontier(raw_goal)
                                self.get_logger().warn(
                                    f'No valid alternative goal near frontier {raw_goal} '
                                    f'with clearance={self.frontier_clearance_current}; '
                                    f'skipping this cluster (added to inflated backlog).'
                                )
                                if cluster_for_goal in remaining_clusters:
                                    remaining_clusters.remove(cluster_for_goal)
                                continue

                            gx, gy = alt_goal
                            path = self.astar_plan((self.robot_mx, self.robot_my), (gx, gy))
                            if path is None:
                                self.remember_inflated_frontier(raw_goal)
                                self.get_logger().warn(
                                    f'A*: no path to adjusted frontier goal {alt_goal} '
                                    f'for original {raw_goal} at clearance={self.frontier_clearance_current} '
                                    f'— skipping this cluster.'
                                )
                                if cluster_for_goal in remaining_clusters:
                                    remaining_clusters.remove(cluster_for_goal)
                                continue

                            smoothed = self.smooth_path(path)

                            self.get_logger().info(
                                f'A* to frontier: raw_len={len(path)}, smooth_len={len(smoothed)}, '
                                f'start={(self.robot_mx, self.robot_my)}, end={smoothed[-1] if smoothed else None}, '
                                f'raw_goal={raw_goal}, used_goal={(gx, gy)}, clearance={self.frontier_clearance_current}'
                            )

                            self.frontier_goal = alt_goal
                            self.last_frontier_goal = raw_goal
                            self.set_current_path(smoothed)
                            found_valid_goal = True

                        # If we still didn't find anything at this clearance,
                        # try far-alt backlog goals at the same clearance.
                        if not found_valid_goal:
                            used_backlog = self.try_inflated_backlog_goals()
                            if used_backlog:
                                found_valid_goal = True

                        if found_valid_goal:
                            break

                    if not found_valid_goal:
                        # Could not find any goal even with clearance reduced down to min.
                        self.get_logger().warn(
                            'No usable frontier goals even after relaxing frontier clearance. '
                            'Clearing path and continuing to spin.'
                        )
                        self.frontier_goal = None
                        self.clear_current_path()
                        self.frontier_clearance_current = original_frontier_clearance
                        self.path_clearance_current = original_path_clearance

            # 4) Visualization
            self.publish_frontier_markers()
            self.publish_global_plan()

            # 5) Follow current path using PID + union of obstacle criteria
            self.follow_current_path()


def main(args=None):
    rclpy.init(args=args)

    task1 = Task1()

    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
