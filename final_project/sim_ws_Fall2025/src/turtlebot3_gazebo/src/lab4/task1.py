#!/usr/bin/env python3

import math
import heapq
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

    Stage 1:
        - Maintain an up-to-date occupancy grid using /map and /map_updates.

    Stage 2:
        - Use TF (map -> base_footprint) to track robot pose in the map frame,
          and convert that to grid indices on the occupancy grid.

    Stage 3:
        - Implement a grid-based A* planner that can compute a path between
          two cells on the occupancy grid.

    Stage 4:
        - Detect frontier cells (free adjacent to unknown), cluster them,
          select a frontier goal (closest cluster + closest point in it),
          and visualize clusters, chosen goal, and A* path in RViz.

    Stage 5:
        - Follow the current A* path using PID control for speed and heading,
          publish /cmd_vel, and perform simple obstacle avoidance using /scan.

    Stage 6:
        - Add a simple exploration state machine:
            WAIT_FOR_MAP -> EXPLORE -> DONE
          Stop exploring only when there are no more frontiers.
    """

    # clustering parameters
    CLUSTER_RADIUS_CELLS = 2      # how far neighbors can be and still be in same cluster
    MIN_CLUSTER_SIZE = 5          # ignore tiny clusters as noise
    MIN_GOAL_DIST_CELLS = 12       # minumum distance of frontier from the robot

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
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.current_path_world: Optional[List[Tuple[float, float]]] = None
        self.path_idx: Optional[int] = None

        # ---------------------------
        # Scan / obstacle state (Stage 5)
        # ---------------------------
        self.min_front_range: Optional[float] = None
        self.front_range_filt: Optional[float] = None
        self.obstacle_stop_dist: float = 0.4  # m

        # ---------------------------
        # PID controllers (Stage 5)
        # ---------------------------
        self.speed_max = 0.22
        self.speed_min = 0.05
        self.heading_max = 2.0  # rad/s

        self.yaw_tol = 0.1        # rad
        self.stop_dist_tol = 0.1  # m
        self.slow_down_dist = 0.35 # m

        self.last_ctrl_time: Optional[float] = None
        self.speed_hist: List[Tuple[float, float, float]] = []  # (t, x, y)

        self.pid_speed = PID(
            kp=2.8, ki=0.10, kd=0.25,
            i_limit=0.8,
            out_limit=(-self.speed_max, self.speed_max)
        )
        self.pid_heading = PID(
            kp=2.0, ki=0.02, kd=0.24,
            i_limit=0.8,
            out_limit=(-self.heading_max, self.heading_max)
        )

        # ---------------------------
        # Exploration state (Stage 6)
        # ---------------------------
        self.state = "WAIT_FOR_MAP"  # or "EXPLORE", "DONE"

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
            10
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

        self.get_logger().info('Task1 node initialized (Stage 1–6).')

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
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

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
        - must be free
        - must have some clearance to occupied cells
        """
        # tune clearance_cells as needed (2–3 is typical with 5cm resolution)
        return self.is_safe_frontier_cell(mx, my, clearance_cells=5)

    # -------------------------------------------------------------------------
    # Scan callback (Stage 5)
    # -------------------------------------------------------------------------
    
    def scan_callback(self, msg: LaserScan):
        """
        Store min range in front sector to detect close obstacles.
        """
        if not msg.ranges:
            return

        n = len(msg.ranges)
        center = n // 2
        half_window = max(1, n // 20)  # ~10% of FOV
        start = max(0, center - half_window)
        end = min(n, center + half_window)

        vals = [
            r for r in msg.ranges[start:end]
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]

        if not vals:
            # No valid measurement in the front window – keep previous filtered value
            self.min_front_range = None
            return

        # raw minimum
        self.min_front_range = min(vals)

        # Smooth front range to avoid jittery false-stops
        if self.front_range_filt is None:
            # first valid reading
            self.front_range_filt = self.min_front_range
        else:
            # low-pass filter: new = 80% old + 20% new
            self.front_range_filt = (
                0.8 * self.front_range_filt + 0.2 * self.min_front_range
            )

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

    def find_nearest_traversable(self, mx: int, my: int,
                                 max_radius: int = 5) -> Optional[Tuple[int, int]]:
        """
        Search in an expanding square around (mx, my) for the nearest
        traversable cell (based on is_traversable). Returns (nx, ny)
        or None if none found within max_radius.
        """
        if self.map_width is None or self.map_height is None:
            return None

        if self.is_traversable(mx, my):
            return (mx, my)

        best_cell = None
        best_dist2 = float('inf')

        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx = mx + dx
                    ny = my + dy
                    if self.map_index(nx, ny) is None:
                        continue
                    if not self.is_traversable(nx, ny):
                        continue
                    dist2 = dx * dx + dy * dy
                    if dist2 < best_dist2:
                        best_dist2 = dist2
                        best_cell = (nx, ny)

            if best_cell is not None:
                break

        return best_cell

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
            alt = self.find_nearest_traversable(sx, sy, max_radius=5)
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
            self.get_logger().warn(f'A*: goal cell {goal} is not traversable.')
            return None

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

            # ignore tiny clusters as noise
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                clusters.append(cluster)

        return clusters
    
    def is_safe_frontier_cell(self, mx: int, my: int,
                              clearance_cells: int = 5) -> bool:
        """
        A 'safe' frontier cell is free AND has no occupied cells
        within a given radius in grid space.

        This acts like a cheap inflation layer: we avoid goals that are
        right next to walls/obstacles.
        """
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
                # treat anything above occ_threshold as obstacle
                if v >= 50:
                    return False

        return True

    def choose_frontier_goal(self, clusters: List[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """
        Choose a frontier goal by:
          1) For each cluster, compute the closest distance from the robot
             to any cell in that cluster. Use that to pick the "closest cluster".
          2) Within the chosen cluster, pick a frontier cell that is at least
             MIN_GOAL_DIST_CELLS away from the robot (in grid), preferring
             the *closest* among those that satisfy the minimum distance.
          3) If no cell satisfies the minimum distance, fall back to the truly
             closest cell in the cluster.
        """
        if not clusters:
            return None
        if self.robot_mx is None or self.robot_my is None:
            return None

        # 1) pick the closest cluster (by any cell)
        best_cluster_idx = None
        best_cluster_min_dist2 = float('inf')

        for idx, cluster in enumerate(clusters):
            cluster_min_dist2 = float('inf')
            for (mx, my) in cluster:
                dx = mx - self.robot_mx
                dy = my - self.robot_my
                dist2 = dx * dx + dy * dy
                if dist2 < cluster_min_dist2:
                    cluster_min_dist2 = dist2

            if cluster_min_dist2 < best_cluster_min_dist2:
                best_cluster_min_dist2 = cluster_min_dist2
                best_cluster_idx = idx

        if best_cluster_idx is None:
            return None
    
        chosen_cluster = clusters[best_cluster_idx]

        # 2) within this cluster, prefer cells at least MIN_GOAL_DIST_CELLS away
        #    AND 'safe' from nearby obstacles.
        min_dist2_required = self.MIN_GOAL_DIST_CELLS ** 2

        candidate_cell = None
        candidate_dist2 = float('inf')

        fallback_cell = None
        fallback_dist2 = float('inf')

        # also track a safe fallback (closest safe cell even if too close)
        safe_fallback_cell = None
        safe_fallback_dist2 = float('inf')

        for (mx, my) in chosen_cluster:
            dx = mx - self.robot_mx
            dy = my - self.robot_my
            dist2 = dx * dx + dy * dy

            # absolute closest (could be unsafe)
            if dist2 < fallback_dist2:
                fallback_dist2 = dist2
                fallback_cell = (mx, my)

            # skip cells that are too close to walls/obstacles
            if not self.is_safe_frontier_cell(mx, my):
                continue

            # track closest safe cell (regardless of min distance)
            if dist2 < safe_fallback_dist2:
                safe_fallback_dist2 = dist2
                safe_fallback_cell = (mx, my)

            # prefer safe cells that are not too close to robot
            if dist2 >= min_dist2_required and dist2 < candidate_dist2:
                candidate_dist2 = dist2
                candidate_cell = (mx, my)

        # Priority:
        # 1) safe + not too close
        if candidate_cell is not None:
            return candidate_cell
        # 2) closest safe cell
        if safe_fallback_cell is not None:
            return safe_fallback_cell
        # 3) if nothing safe, fall back to closest cell in cluster
        return fallback_cell

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

        For exploration, we only treat *occupied* cells as hard obstacles.
        Unknown (-1) is allowed, so the robot can still move toward frontiers.
        If any upcoming path cell becomes occupied, the path is considered
        invalid and we should replan.
        """
        if self.current_path is None or self.path_idx is None:
            return False

        start_idx = max(0, self.path_idx)
        end_idx = min(len(self.current_path), start_idx + look_ahead_cells)

        for i in range(start_idx, end_idx):
            mx, my = self.current_path[i]
            # only hard-block on occupied cells
            if self.is_occupied(mx, my):
                return False

        return True

    def follow_current_path(self):
        """
        Follow the current path using PID on speed and heading.
        Uses BOTH:
          - /scan front distance
          - occupancy grid along the path
        to decide when to stop and replan.
        """
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
        if (
            self.front_range_filt is not None and
            self.front_range_filt < self.obstacle_stop_dist
        ):
            self.get_logger().warn(
                f'Obstacle too close (scan front range={self.front_range_filt:.2f} m). '
                f'Stopping and clearing path for replanning.',
                throttle_duration_sec=1.0
            )
            self.clear_current_path()
            return

        # 2) map-based obstacle along path (occupied cells only)
        if not self.is_path_still_valid():
            self.get_logger().warn(
                'Current path intersects newly occupied/unknown cells in map. '
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
            if dist_to_wp < 0.1:  # waypoint tolerance
                self.path_idx += 1
            else:
                break

        # Check final goal
        gx, gy = self.current_path_world[self.path_idx]
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        if self.path_idx == len(self.current_path_world) - 1 and dist_to_goal < self.stop_dist_tol:
            self.get_logger().info(
                'Reached current path goal. Stopping and clearing path.',
                throttle_duration_sec=2.0
            )
            self.clear_current_path()
            return

        # Desired heading
        dx = gx - rx
        dy = gy - ry
        desired_yaw = math.atan2(dy, dx)
        heading_err = wrap_angle(desired_yaw - ryaw)

        # --- Smooth, heuristic-based speed control (no speed PID) ---

        # If turning in place, give grace period — don't clear path yet
        if abs(heading_err) > 0.7:
            if (
                self.front_range_filt is not None and
                self.front_range_filt < self.obstacle_stop_dist * 0.8
            ):
                self.publish_cmd_vel(0.0, heading_cmd)  # keep turning
                return
        
        # Turn-in-place behavior when we are badly misaligned
        heading_cmd = self.pid_heading.step(heading_err, dt)

        if abs(heading_err) > 1.0:      # only stop if very misaligned
            speed_cmd = 0.0
        elif abs(heading_err) > 0.4:    # turning while moving slowly
            speed_cmd = 0.04
        else:
            speed_cmd = self.speed_max

        # safety: never go backwards
        speed_cmd = max(0.0, min(self.speed_max, speed_cmd))

        self.publish_cmd_vel(speed_cmd, heading_cmd)

    # -------------------------------------------------------------------------
    # Timer callback - behavior loop (Stage 6)
    # -------------------------------------------------------------------------

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

                # If no current path, choose new goal and plan
                if self.current_path is None:
                    goal_cell = self.choose_frontier_goal(clusters)
                    self.frontier_goal = goal_cell

                    if goal_cell is not None:
                        gx, gy = goal_cell
                        self.get_logger().info(
                            f'Chosen frontier goal: grid=({gx}, {gy})'
                        )
                        if self.is_traversable(gx, gy):
                            path = self.astar_plan((self.robot_mx, self.robot_my), (gx, gy))
                            if path is not None:
                                self.get_logger().info(
                                    f'A* to frontier: path length={len(path)}, '
                                    f'start={path[0]}, end={path[-1]}'
                                )
                                self.set_current_path(path)
                            else:
                                self.get_logger().warn(
                                    'A* to frontier: no path found (maybe blocked).'
                                )
                        else:
                            self.get_logger().warn(
                                'Chosen frontier goal is not traversable when planning.'
                            )
                    else:
                        self.get_logger().warn('No valid frontier goal found.')

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





'''#!/usr/bin/env python3

import math
import heapq
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

    Stage 1:
        - Maintain an up-to-date occupancy grid using /map and /map_updates.

    Stage 2:
        - Use TF (map -> base_footprint) to track robot pose in the map frame,
          and convert that to grid indices on the occupancy grid.

    Stage 3:
        - Implement a grid-based A* planner that can compute a path between
          two cells on the occupancy grid.

    Stage 4:
        - Detect frontier cells (free adjacent to unknown), cluster them,
          select a frontier goal (closest cluster + closest point in it),
          and visualize clusters, chosen goal, and A* path in RViz.

    Stage 5:
        - Follow the current A* path using PID control for speed and heading,
          publish /cmd_vel, and perform simple obstacle avoidance using /scan.
    """

    # clustering parameters
    CLUSTER_RADIUS_CELLS = 2      # how far neighbors can be and still be in same cluster
    MIN_CLUSTER_SIZE = 5          # ignore tiny clusters as noise

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
        # Frontier / path state (Stage 4)
        # ---------------------------
        self.frontier_clusters: List[List[Tuple[int, int]]] = []
        self.frontier_goal: Optional[Tuple[int, int]] = None
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.current_path_world: Optional[List[Tuple[float, float]]] = None
        self.path_idx: Optional[int] = None

        # ---------------------------
        # Scan / obstacle state (Stage 5)
        # ---------------------------
        self.min_front_range: Optional[float] = None
        self.obstacle_stop_dist: float = 0.35

        # ---------------------------
        # PID controllers (Stage 5)
        # ---------------------------
        # Tuning parameters (can be adjusted)
        self.speed_max = 0.22
        self.speed_min = 0.05
        self.heading_max = 2.0  # rad/s

        self.yaw_tol = 0.1      # rad, when close to final heading
        self.stop_dist_tol = 0.1  # m, when close enough to final goal
        self.slow_down_dist = 0.5  # start slowing down near goal

        self.last_ctrl_time: Optional[float] = None
        self.speed_hist: List[Tuple[float, float, float]] = []  # (t, x, y)

        self.pid_speed = PID(
            kp=2.8, ki=0.10, kd=0.25,
            i_limit=0.8,
            out_limit=(-self.speed_max, self.speed_max)
        )
        self.pid_heading = PID(
            kp=2.0, ki=0.02, kd=0.24,
            i_limit=0.8,
            out_limit=(-self.heading_max, self.heading_max)
        )

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
            10
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
        # Timer – debug + behavior loop
        # ---------------------------
        self.timer = self.create_timer(0.1, self.timer_cb)  # 10 Hz control

        self.get_logger().info('Task1 node initialized (Stage 1–5).')

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
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

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
        Traversable = free.
        Unknown or occupied are treated as non-traversable for now.
        """
        if self.map_index(mx, my) is None:
            return False
        return self.is_free(mx, my)

    # -------------------------------------------------------------------------
    # Scan callback (Stage 5)
    # -------------------------------------------------------------------------

    def scan_callback(self, msg: LaserScan):
        """
        Store min range in front sector to detect close obstacles.
        """
        if not msg.ranges:
            return
        n = len(msg.ranges)
        center = n // 2
        half_window = max(1, n // 20)  # ~10% of FOV
        start = max(0, center - half_window)
        end = min(n, center + half_window)
        vals = [
            r for r in msg.ranges[start:end]
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]
        if vals:
            self.min_front_range = min(vals)
        else:
            self.min_front_range = None

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
            self.get_logger().warn(f'A*: start cell {start} is not traversable.')
            return None
        if not self.is_traversable(gx, gy):
            self.get_logger().warn(f'A*: goal cell {goal} is not traversable.')
            return None

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

            # ignore tiny clusters as noise
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                clusters.append(cluster)

        return clusters

    def choose_frontier_goal(self, clusters: List[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """
        Choose a frontier goal by:
          - for each cluster, find the cell in that cluster closest (in grid distance)
            to the robot's current cell.
          - pick the cluster whose closest cell is nearest to the robot.
        """
        if not clusters:
            return None
        if self.robot_mx is None or self.robot_my is None:
            return None

        best_cell: Optional[Tuple[int, int]] = None
        best_dist2 = float('inf')

        for cluster in clusters:
            for (mx, my) in cluster:
                dx = mx - self.robot_mx
                dy = my - self.robot_my
                dist2 = dx * dx + dy * dy
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_cell = (mx, my)

        return best_cell

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
    # Path following (Stage 5)
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

    def follow_current_path(self):
        """
        Follow the current path using PID on speed and heading.
        Uses /scan for simple obstacle stop.
        """
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

        # Obstacle stop check
        if self.min_front_range is not None and self.min_front_range < self.obstacle_stop_dist:
            self.get_logger().warn(
                f'Obstacle too close (front range={self.min_front_range:.2f} m). Stopping.',
                throttle_duration_sec=1.0
            )
            self.publish_cmd_vel(0.0, 0.0)
            return

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
            if dist_to_wp < 0.1:  # waypoint tolerance
                self.path_idx += 1
            else:
                break

        # Check final goal
        gx, gy = self.current_path_world[self.path_idx]
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        if self.path_idx == len(self.current_path_world) - 1 and dist_to_goal < self.stop_dist_tol:
            self.get_logger().info('Reached current path goal. Stopping and clearing path.',
                                   throttle_duration_sec=2.0)
            self.clear_current_path()
            return

        # Desired heading
        dx = gx - rx
        dy = gy - ry
        desired_yaw = math.atan2(dy, dx)
        heading_err = wrap_angle(desired_yaw - ryaw)

        # Simple speed target: move faster when far, slower near goal
        if dist_to_goal > self.slow_down_dist:
            speed_goal = self.speed_max
        else:
            speed_goal = self.speed_min
        speed_err = speed_goal - speed_curr

        # PID outputs
        speed_cmd = self.pid_speed.step(speed_err, dt)
        heading_cmd = self.pid_heading.step(heading_err, dt)

        # If heading error is large, reduce forward speed to help turning
        if abs(heading_err) > 0.7:  # ~40 deg
            speed_cmd *= 0.3

        self.publish_cmd_vel(speed_cmd, heading_cmd)

    # -------------------------------------------------------------------------
    # Timer callback - behavior loop (Stage 1–5)
    # -------------------------------------------------------------------------

    def timer_cb(self):
        """
        Behavior loop:
          - report map stats (slowly)
          - update robot pose via TF
          - detect & cluster frontiers
          - if no current path, select a frontier goal and A* plan to it
          - publish visualization markers and global plan
          - follow current path with PID + obstacle stop
        """
        self.timer_counter += 1

        # 1) Map stats occasionally
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

        # If not ready, don't do frontier or path following
        if not (self.map_received and self.robot_pose_received and
                self.robot_mx is not None and self.robot_my is not None):
            self.publish_cmd_vel(0.0, 0.0)
            return

        # Small warmup before planning so map isn't too tiny
        if self.timer_counter < 30:
            self.publish_cmd_vel(0.0, 0.0)
            return

        # 3) Frontier detection & clustering (every N steps)
        if self.timer_counter % 10 == 0:  # every ~1s
            frontier_cells = self.find_frontier_cells()
            clusters = self.cluster_frontiers(frontier_cells)
            self.frontier_clusters = clusters
            self.get_logger().info(
                f'Frontiers: {len(frontier_cells)} cells, {len(clusters)} clusters.'
            )

            # 4) If no current path, choose new goal and plan
            if self.current_path is None:
                goal_cell = self.choose_frontier_goal(clusters)
                self.frontier_goal = goal_cell

                if goal_cell is not None:
                    gx, gy = goal_cell
                    self.get_logger().info(
                        f'Chosen frontier goal: grid=({gx}, {gy})'
                    )
                    if self.is_traversable(gx, gy):
                        path = self.astar_plan((self.robot_mx, self.robot_my), (gx, gy))
                        if path is not None:
                            self.get_logger().info(
                                f'A* to frontier: path length={len(path)}, '
                                f'start={path[0]}, end={path[-1]}'
                            )
                            self.set_current_path(path)
                        else:
                            self.get_logger().warn(
                                'A* to frontier: no path found (maybe blocked).'
                            )
                    else:
                        self.get_logger().warn(
                            'Chosen frontier goal is not traversable when planning.'
                        )
                else:
                    self.get_logger().warn('No valid frontier goal found.')

        # 5) Visualization
        self.publish_frontier_markers()
        self.publish_global_plan()

        # 6) Follow current path using PID
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
'''




'''#!/usr/bin/env python3

import math
import heapq
from typing import Optional, Tuple, List, Dict, Set

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, Path
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from tf2_ros import Buffer, TransformListener


class Task1(Node):
    """
    Task 1 – Autonomous Mapping

    Stage 1:
        - Maintain an up-to-date occupancy grid using /map and /map_updates.

    Stage 2:
        - Use TF (map -> base_footprint) to track robot pose in the map frame,
          and convert that to grid indices on the occupancy grid.

    Stage 3:
        - Implement a grid-based A* planner that can compute a path between
          two cells on the occupancy grid.

    Stage 4:
        - Detect frontier cells (free adjacent to unknown), cluster them (coarser),
          select a frontier goal (closest cluster + closest point in it),
          and visualize clusters, chosen goal, and A* path in RViz.
    """

    # clustering parameters
    CLUSTER_RADIUS_CELLS = 2      # how far neighbors can be and still be in same cluster
    MIN_CLUSTER_SIZE = 5          # ignore tiny clusters as noise

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
        self.map_data: Optional[list[int]] = None

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
        # Frontier state (Stage 4)
        # ---------------------------
        self.frontier_clusters: List[List[Tuple[int, int]]] = []
        self.frontier_goal: Optional[Tuple[int, int]] = None
        self.current_path: Optional[List[Tuple[int, int]]] = None

        # small warmup so slam_toolbox can build some map
        self.timer_counter = 0

        # ---------------------------
        # Publishers
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
            10
        )

        # Visualization publishers
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
        # Timer – debug + future state machine
        # ---------------------------
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info('Task1 node initialized (Stage 1–4 with visualization).')

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
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

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
        Traversable = free.
        Unknown or occupied are treated as non-traversable for now.
        We can later change this for exploration (e.g. unknown as high-cost).
        """
        if self.map_index(mx, my) is None:
            return False
        return self.is_free(mx, my)

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
            self.get_logger().warn(f'A*: start cell {start} is not traversable.')
            return None
        if not self.is_traversable(gx, gy):
            self.get_logger().warn(f'A*: goal cell {goal} is not traversable.')
            return None

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

        This is more aggressive than simple 8-connected BFS, so you get
        fewer, larger clusters.
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

            # ignore tiny clusters as noise
            if len(cluster) >= self.MIN_CLUSTER_SIZE:
                clusters.append(cluster)

        return clusters

    def choose_frontier_goal(self, clusters: List[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """
        Choose a frontier goal by:
          - for each cluster, find the cell in that cluster closest (in grid distance)
            to the robot's current cell.
          - pick the cluster whose closest cell is nearest to the robot.

        Returns:
            (goal_mx, goal_my) or None if no valid frontier.
        """
        if not clusters:
            return None
        if self.robot_mx is None or self.robot_my is None:
            return None

        best_cell = None
        best_dist2 = float('inf')

        for cluster in clusters:
            for (mx, my) in cluster:
                dx = mx - self.robot_mx
                dy = my - self.robot_my
                dist2 = dx * dx + dy * dy
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_cell = (mx, my)

        return best_cell

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
    # Timer callback - map + pose + frontier/A* + visualization (Stage 1–4)
    # -------------------------------------------------------------------------

    def timer_cb(self):
        """
        For now:
          - report map stats
          - update and report robot pose & grid cell via TF
          - repeatedly run frontier detection + clustering + goal selection + A* plan
            (after a short warmup).
          - publish markers and global plan for RViz.

        Later, this will be replaced by a proper exploration state machine.
        """
        self.timer_counter += 1

        # 1) Map stats
        if not self.map_received or self.map_data is None:
            self.get_logger().info('Waiting for /map...', throttle_duration_sec=2.0)
        else:
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
                f'Map stats: total={total}, unknown={unknown}, free={free}, occupied={occupied}',
                throttle_duration_sec=5.0
            )

        # 2) Update robot pose from TF and log
        self.update_robot_pose_from_tf()

        if self.robot_pose_received and self.robot_mx is not None and self.robot_my is not None:
            self.get_logger().info(
                f'Robot pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, '
                f'yaw={math.degrees(self.robot_yaw):.1f} deg, '
                f'grid=({self.robot_mx}, {self.robot_my})',
                throttle_duration_sec=1.0
            )
        else:
            self.get_logger().info(
                'Waiting for TF transform map->base_footprint or robot outside map...',
                throttle_duration_sec=2.0
            )

        # 3) Frontier detection + A* + visualization (after warmup)
        if (self.map_received and self.robot_pose_received and
                self.robot_mx is not None and self.robot_my is not None and
                self.timer_counter > 20):

            frontier_cells = self.find_frontier_cells()
            self.get_logger().info(
                f'Frontier detection: found {len(frontier_cells)} frontier cells.',
                throttle_duration_sec=5.0
            )

            clusters = self.cluster_frontiers(frontier_cells)
            self.frontier_clusters = clusters
            self.get_logger().info(
                f'Frontier clustering: {len(clusters)} clusters (after min-size & radius).',
                throttle_duration_sec=5.0
            )

            goal_cell = self.choose_frontier_goal(clusters)
            self.frontier_goal = goal_cell

            if goal_cell is not None:
                gx, gy = goal_cell
                self.get_logger().info(
                    f'Chosen frontier goal (closest cell): grid=({gx}, {gy})',
                    throttle_duration_sec=5.0
                )

                if self.is_traversable(gx, gy):
                    path = self.astar_plan((self.robot_mx, self.robot_my), (gx, gy))
                    self.current_path = path
                    if path is not None:
                        self.get_logger().info(
                            f'A* to frontier: path length={len(path)}, '
                            f'start={path[0]}, end={path[-1]}',
                            throttle_duration_sec=5.0
                        )
                    else:
                        self.get_logger().warn(
                            'A* to frontier: no path found (maybe behind obstacle).',
                            throttle_duration_sec=5.0
                        )
                else:
                    self.get_logger().warn(
                        'Chosen frontier cell is not traversable when planning.',
                        throttle_duration_sec=5.0
                    )
            else:
                self.get_logger().warn(
                    'No valid frontier goal found.',
                    throttle_duration_sec=5.0
                )

            # publish visualization
            self.publish_frontier_markers()
            self.publish_global_plan()
        else:
            self.get_logger().info(
                f'Waiting to run frontier planning... counter={self.timer_counter}',
                throttle_duration_sec=5.0
            )


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
    main()'''




'''#!/usr/bin/env python3

import math
import heapq
from typing import Optional, Tuple, List, Dict, Set

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate

from tf2_ros import Buffer, TransformListener


class Task1(Node):
    """
    Task 1 – Autonomous Mapping

    Stage 1:
        - Maintain an up-to-date occupancy grid using /map and /map_updates.

    Stage 2:
        - Use TF (map -> base_footprint) to track robot pose in the map frame,
          and convert that to grid indices on the occupancy grid.

    Stage 3:
        - Implement a grid-based A* planner that can compute a path between
          two cells on the occupancy grid.

    Stage 4:
        - Detect frontier cells (free adjacent to unknown), cluster them,
          and select a frontier goal (closest cluster + closest point in it).
    """

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
        self.map_data: Optional[list[int]] = None

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
        # Debug flags / counters
        # ---------------------------
        self.debug_astar_tested = False  # one-time debug plan
        self.counter = 0

        # for Stage 4 debug
        self.debug_frontier_tested = False
        self.current_frontier_goal: Optional[Tuple[int, int]] = None

        # ---------------------------
        # Subscribers (Stage 1)
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
            10
        )

        # ---------------------------
        # Timer – debug + future state machine
        # ---------------------------
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info('Task1 node initialized (Stage 1–4).')

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
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

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
        Traversable = free.
        Unknown or occupied are treated as non-traversable for now.
        We can later change this for exploration (e.g. unknown as high-cost).
        """
        if self.map_index(mx, my) is None:
            return False
        return self.is_free(mx, my)

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
            self.get_logger().warn(f'A*: start cell {start} is not traversable.')
            return None
        if not self.is_traversable(gx, gy):
            self.get_logger().warn(f'A*: goal cell {goal} is not traversable.')
            return None

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
        Cluster frontier cells using BFS (8-connected).
        Returns a list of clusters, each cluster is a list of (mx, my).
        """
        clusters: List[List[Tuple[int, int]]] = []
        if not frontier_cells:
            return clusters

        frontier_set: Set[Tuple[int, int]] = set(frontier_cells)
        visited: Set[Tuple[int, int]] = set()

        # 8-connected neighbors for clustering
        neighbors = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        for cell in frontier_cells:
            if cell in visited:
                continue
            # start new cluster
            cluster: List[Tuple[int, int]] = []
            queue: List[Tuple[int, int]] = [cell]
            visited.add(cell)

            while queue:
                cx, cy = queue.pop(0)
                cluster.append((cx, cy))

                for dx, dy in neighbors:
                    nx = cx + dx
                    ny = cy + dy
                    neighbor = (nx, ny)
                    if neighbor in frontier_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    def choose_frontier_goal(self, clusters: List[List[Tuple[int, int]]]) -> Optional[Tuple[int, int]]:
        """
        Choose a frontier goal by:
          - for each cluster, find the cell in that cluster closest (in grid distance)
            to the robot's current cell.
          - pick the cluster whose closest cell is nearest to the robot.

        Returns:
            (goal_mx, goal_my) or None if no valid frontier.
        """
        if not clusters:
            return None
        if self.robot_mx is None or self.robot_my is None:
            return None

        best_cell = None
        best_dist2 = float('inf')

        for cluster in clusters:
            for (mx, my) in cluster:
                dx = mx - self.robot_mx
                dy = my - self.robot_my
                dist2 = dx * dx + dy * dy
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_cell = (mx, my)

        return best_cell

    # -------------------------------------------------------------------------
    # Timer callback - debug map + robot pose + frontier/A* test (Stage 1–4)
    # -------------------------------------------------------------------------

    def timer_cb(self):
        """
        For now:
          - report map stats
          - update and report robot pose & grid cell via TF
          - run a one-time frontier detection + A* test (for debugging)
            once map & pose are ready.

        Later, this will run the exploration state machine.
        """
        # 1) Map stats
        if not self.map_received or self.map_data is None:
            self.get_logger().info('Waiting for /map...', throttle_duration_sec=2.0)
        else:
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
                f'Map stats: total={total}, unknown={unknown}, free={free}, occupied={occupied}',
                throttle_duration_sec=5.0
            )

        # 2) Update robot pose from TF and log
        self.update_robot_pose_from_tf()

        if self.robot_pose_received and self.robot_mx is not None and self.robot_my is not None:
            self.get_logger().info(
                f'Robot pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, '
                f'yaw={math.degrees(self.robot_yaw):.1f} deg, '
                f'grid=({self.robot_mx}, {self.robot_my})',
                throttle_duration_sec=1.0
            )
        else:
            self.get_logger().info(
                'Waiting for TF transform map->base_footprint or robot outside map...',
                throttle_duration_sec=2.0
            )

        # 3) Frontier detection + A* debug (one-time, after some time so the map grows a bit)
        if (self.map_received and self.robot_pose_received and
                self.robot_mx is not None and self.robot_my is not None and
                not self.debug_frontier_tested and self.counter > 25):

            frontier_cells = self.find_frontier_cells()
            self.get_logger().info(
                f'Frontier detection: found {len(frontier_cells)} frontier cells.',
                throttle_duration_sec=2.0
            )

            clusters = self.cluster_frontiers(frontier_cells)
            self.get_logger().info(
                f'Frontier clustering: {len(clusters)} clusters.',
                throttle_duration_sec=2.0
            )

            goal_cell = self.choose_frontier_goal(clusters)
            if goal_cell is None:
                self.get_logger().warn('No valid frontier goal found.')
            else:
                gx, gy = goal_cell
                self.current_frontier_goal = goal_cell
                self.get_logger().info(
                    f'Chosen frontier goal (closest cell): grid=({gx}, {gy})',
                    throttle_duration_sec=2.0
                )

                # Optional: run A* to this goal for debug
                if self.is_traversable(gx, gy):
                    path = self.astar_plan((self.robot_mx, self.robot_my), (gx, gy))
                    if path is not None:
                        self.get_logger().info(
                            f'A* to frontier: path length={len(path)}, '
                            f'start={path[0]}, end={path[-1]}'
                        )
                    else:
                        self.get_logger().warn(
                            'A* to frontier: no path found (maybe cluster behind obstacle).'
                        )
                else:
                    self.get_logger().warn(
                        'Chosen frontier cell is not traversable when planning (should not happen).'
                    )

            self.debug_frontier_tested = True

        else:
            self.counter += 1
            self.get_logger().info(
                f'Waiting to run frontier debug... counter={self.counter}, '
                f'frontier_tested={self.debug_frontier_tested}',
                throttle_duration_sec=5.0
            )


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
'''




'''#!/usr/bin/env python3

import math
import heapq
from typing import Optional, Tuple, List, Dict

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate

from tf2_ros import Buffer, TransformListener


class Task1(Node):
    """
    Task 1 – Autonomous Mapping

    Stage 1:
        - Maintain an up-to-date occupancy grid using /map and /map_updates.

    Stage 2:
        - Use TF (map -> base_footprint) to track robot pose in the map frame,
          and convert that to grid indices on the occupancy grid.

    Stage 3:
        - Implement a grid-based A* planner that can compute a path between
          two cells on the occupancy grid.
    """

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
        self.map_data: Optional[list[int]] = None

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
        # Debug flags for Stage 3
        # ---------------------------
        # We'll use this later to test A* once, not every timer tick
        self.debug_astar_tested = False
        self.counter = 0

        # ---------------------------
        # Subscribers (Stage 1)
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
            10
        )

        # ---------------------------
        # Timer – debug + future state machine
        # ---------------------------
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info('Task1 node initialized (Stage 1 + 2 + 3).')

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
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

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
        Traversable = free.
        Unknown or occupied are treated as non-traversable for now.
        We can later change this for exploration (e.g. unknown as high-cost).
        """
        if self.map_index(mx, my) is None:
            return False
        return self.is_free(mx, my)

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
            self.get_logger().warn(f'A*: start cell {start} is not traversable.')
            return None
        if not self.is_traversable(gx, gy):
            self.get_logger().warn(f'A*: goal cell {goal} is not traversable.')
            return None

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

        closed_set = set()

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
    # Timer callback - debug map + robot pose + A* test (Stage 1–3)
    # -------------------------------------------------------------------------

    def timer_cb(self):
        """
        For now:
          - report map stats
          - update and report robot pose & grid cell via TF
          - run a one-time A* test (for debugging) once map & pose are ready

        Later, this will run the exploration state machine.
        """
        # 1) Map stats
        if not self.map_received or self.map_data is None:
            self.get_logger().info('Waiting for /map...', throttle_duration_sec=2.0)
        else:
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
                f'Map stats: total={total}, unknown={unknown}, free={free}, occupied={occupied}',
                throttle_duration_sec=5.0
            )

        # 2) Update robot pose from TF and log
        self.update_robot_pose_from_tf()

        if self.robot_pose_received and self.robot_mx is not None and self.robot_my is not None:
            self.get_logger().info(
                f'Robot pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, '
                f'yaw={math.degrees(self.robot_yaw):.1f} deg, '
                f'grid=({self.robot_mx}, {self.robot_my})',
                throttle_duration_sec=1.0
            )
        else:
            self.get_logger().info(
                'Waiting for TF transform map->base_footprint or robot outside map...',
                throttle_duration_sec=2.0
            )

        # 3) One-time A* debug test (optional, harmless)
        #    Once the map has some free space and we know our pose,
        #    try to plan a short path ahead of the robot.
        if (self.map_received and self.robot_pose_received and
                self.robot_mx is not None and self.robot_my is not None and
                not self.debug_astar_tested and self.counter > 25):

            # pick a simple goal a few cells "ahead" in x direction
            goal_mx = self.robot_mx + 5
            goal_my = self.robot_my + 5

            if self.map_index(goal_mx, goal_my) is not None and self.is_traversable(goal_mx, goal_my):
                path = self.astar_plan((self.robot_mx, self.robot_my), (goal_mx, goal_my))
                if path is not None:
                    self.get_logger().info(
                        f'A* debug: planned path of length {len(path)} '
                        f'from {path[0]} to {path[-1]}'
                    )
                else:
                    self.get_logger().warn('A* debug: no path found.')
            else:
                self.get_logger().warn(
                    f'A* debug: chosen goal cell ({goal_mx},{goal_my}) is not traversable or out of bounds.'
                )

            self.debug_astar_tested = True  # avoid spamming planning every second
        else:
            self.counter += 1
            self.get_logger().info(
                f'Waiting for enough space to plan a path...; Is tested {self.debug_astar_tested}',
                throttle_duration_sec=2.0
            )


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
    main()'''





'''#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate

from tf2_ros import Buffer, TransformListener


class Task1(Node):
    """
    Environment mapping task.

    Stage 1:
        - Maintain an up-to-date occupancy grid using /map and /map_updates.
    Stage 2:
        - Use TF (map -> base_footprint) to track robot pose in the map frame,
          and convert that to grid indices on the occupancy grid.
    """

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
        self.map_data: Optional[list[int]] = None

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
        # Subscribers (Stage 1)
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
            10
        )

        # ---------------------------
        # Timer – debug + future state machine
        # ---------------------------
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info('Task1 node initialized (Stage 1 + Stage 2 w/ TF).')

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
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

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

    # -------------------------------------------------------------------------
    # TF-based robot pose update (Stage 2)
    # -------------------------------------------------------------------------

    def update_robot_pose_from_tf(self):
        """
        Query TF for the transform map -> base_footprint and update
        robot_x, robot_y, robot_yaw and grid indices (robot_mx, robot_my).
        """
        try:
            # Time(0) or Time() means "latest available transform"
            t = self.tf_buffer.lookup_transform(
                'map',              # target frame
                'base_footprint',   # source frame (from slam_toolbox params)
                Time()
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
            # TF might not be ready yet or transform can briefly disappear
            self.get_logger().warn(
                f'Failed to lookup transform map->base_footprint: {e}',
                throttle_duration_sec=5.0
            )

    @staticmethod
    def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Convert a quaternion into yaw (rotation about Z).
        """
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    # -------------------------------------------------------------------------
    # Timer callback - debug map + robot pose (Stage 1 + 2)
    # -------------------------------------------------------------------------

    def timer_cb(self):
        """
        For now, just:
          - report map stats
          - update and report robot pose & grid cell via TF

        Later, this will run the exploration state machine.
        """
        # 1) Map stats
        if not self.map_received or self.map_data is None:
            self.get_logger().info('Waiting for /map...', throttle_duration_sec=2.0)
        else:
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
                f'Map stats: total={total}, unknown={unknown}, free={free}, occupied={occupied}',
                throttle_duration_sec=5.0
            )

        # 2) Update robot pose from TF and log
        self.update_robot_pose_from_tf()

        if not self.robot_pose_received:
            self.get_logger().info('Waiting for TF transform map->base_footprint...', throttle_duration_sec=2.0)
        else:
            if self.robot_mx is not None and self.robot_my is not None:
                self.get_logger().info(
                    f'Robot pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, '
                    f'yaw={math.degrees(self.robot_yaw):.1f} deg, '
                    f'grid=({self.robot_mx}, {self.robot_my})',
                    throttle_duration_sec=1.0
                )
            else:
                self.get_logger().warn(
                    f'Robot world pose x={self.robot_x:.2f}, y={self.robot_y:.2f} '
                    f'is outside current map bounds.',
                    throttle_duration_sec=2.0
                )


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
    main()'''



'''#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import PoseWithCovarianceStamped


class Task1(Node):
    """
    Task 1 – Autonomous Mapping
    Stage 1: Map maintenance (/map, /map_updates)
    Stage 2: Robot pose tracking (/amcl_pose) + pose -> grid conversion
    """

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

        # flat list of int8: -1 unknown, 0 free, 100+ occupied
        self.map_data: Optional[list[int]] = None

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

        self.robot_pose_received = False

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
            10
        )

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            10
        )

        # ---------------------------
        # Timer – debug + future state machine
        # ---------------------------
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info('Task1 node initialized (Stage 1 + Stage 2).')

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
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            self.map_data = list(msg.data)
            self.map_received = True

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

    # -------------------------------------------------------------------------
    # AMCL pose callback (Stage 2)
    # -------------------------------------------------------------------------

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        """
        Track robot pose in the map frame using /amcl_pose.
        """
        pose = msg.pose.pose
        x = pose.position.x
        y = pose.position.y
        yaw = self.yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        self.robot_x = x
        self.robot_y = y
        self.robot_yaw = yaw
        self.robot_pose_received = True

        # Also compute grid indices if map is ready
        if self.map_received:
            cell = self.world_to_map(x, y)
            if cell is not None:
                self.robot_mx, self.robot_my = cell
            else:
                self.robot_mx = None
                self.robot_my = None

    @staticmethod
    def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Convert a quaternion into yaw (rotation about Z).
        """
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    # -------------------------------------------------------------------------
    # Timer callback – debug (Stage 1 + 2)
    # -------------------------------------------------------------------------

    def timer_cb(self):
        """
        For now: report map stats and robot pose/grid cell periodically.
        Later: this will run the exploration state machine.
        """
        # Map debug
        if not self.map_received or self.map_data is None:
            self.get_logger().info('Waiting for /map...', throttle_duration_sec=2.0)
        else:
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
                f'Map stats: total={total}, unknown={unknown}, '
                f'free={free}, occupied={occupied}',
                throttle_duration_sec=5.0
            )

        # Robot pose debug
        if not self.robot_pose_received:
            self.get_logger().info('Waiting for /amcl_pose...', throttle_duration_sec=2.0)
        else:
            if self.robot_mx is not None and self.robot_my is not None:
                self.get_logger().info(
                    f'Robot pose: x={self.robot_x:.2f}, y={self.robot_y:.2f}, '
                    f'yaw={math.degrees(self.robot_yaw):.1f} deg, '
                    f'grid=({self.robot_mx}, {self.robot_my})',
                    throttle_duration_sec=1.0
                )
            else:
                self.get_logger().warn(
                    f'Robot pose x={self.robot_x:.2f}, y={self.robot_y:.2f} '
                    f'is outside current map bounds.',
                    throttle_duration_sec=2.0
                )


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
    main()'''



'''#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate


class Task1(Node):
    """
    Environment mapping task - Stage 1: Map maintenance.
    Keeps an up-to-date occupancy grid using /map and /map_updates.
    """

    def __init__(self):
        super().__init__('task1_node')

        # --- Map-related state ---
        self.map_received = False

        self.map_width: Optional[int] = None
        self.map_height: Optional[int] = None
        self.map_resolution: Optional[float] = None  # meters/cell
        self.map_origin_x: Optional[float] = None
        self.map_origin_y: Optional[float] = None

        # flat list of int8: -1 unknown, 0 free, 100 occupied
        self.map_data = None  # type: Optional[list[int]]

        # --- Subscribers ---
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
            10
        )

        # --- Timer for debug / future state machine ---
        self.timer = self.create_timer(1.0, self.timer_cb)

        self.get_logger().info('Task1 Stage 1: Map maintenance node initialized.')

    # -------------------------------------------------------------------------
    # Map callbacks
    # -------------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        """Receive the full map and (re)initialize our internal grid."""
        info = msg.info

        first_time = not self.map_received
        resized = False

        if first_time:
            resized = True
        else:
            # If dimensions changed (e.g., SLAM resized map), we must re-init
            if info.width != self.map_width or info.height != self.map_height:
                resized = True

        if resized:
            self.map_width = info.width
            self.map_height = info.height
            self.map_resolution = info.resolution
            self.map_origin_x = info.origin.position.x
            self.map_origin_y = info.origin.position.y

            # copy full map data
            self.map_data = list(msg.data)
            self.map_received = True

            self.get_logger().info(
                f'/map received: size=({self.map_width} x {self.map_height}), '
                f'res={self.map_resolution:.3f} m/cell, '
                f'origin=({self.map_origin_x:.2f}, {self.map_origin_y:.2f})'
            )
        else:
            # Just refresh map_data if same size
            self.map_data = list(msg.data)
            self.map_received = True
            self.get_logger().info('/map updated with same size (refresh).', throttle_duration_sec=5.0)

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
                # row-major indexing: data[my * width + mx]
                map_idx = my * self.map_width + mx
                if 0 <= map_idx < len(self.map_data):
                    self.map_data[map_idx] = msg.data[idx]
                idx += 1

        self.get_logger().info(
            f'Applied /map_updates patch at ({msg.x},{msg.y}) size=({msg.width}x{msg.height})',
            throttle_duration_sec=1.0
        )

    # -------------------------------------------------------------------------
    # Map helpers
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

    # -------------------------------------------------------------------------
    # Timer callback - for now, just debug the map
    # -------------------------------------------------------------------------

    def timer_cb(self):
        """
        For Stage 1, just report map status periodically.
        Later, this is where we’ll hook in the exploration state machine.
        """
        if not self.map_received or self.map_data is None:
            self.get_logger().info('Waiting for /map...', throttle_duration_sec=2.0)
            return

        # Simple stats: count unknown / free / occupied
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
            f'Map stats: total={total}, unknown={unknown}, free={free}, occupied={occupied}',
            throttle_duration_sec=5.0
        )
        # Later we’ll also track "visited" cells and coverage here.


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
    main()'''



'''#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Import other python packages that you think necessary


class Task1(Node):
    """
    Environment mapping task.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need

    def timer_cb(self):
        self.get_logger().info('Task1 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function

    # Define function(s) that complete the (automatic) mapping task


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
    main()'''