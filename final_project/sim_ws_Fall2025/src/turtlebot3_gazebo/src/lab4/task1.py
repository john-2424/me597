#!/usr/bin/env python3

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
    main()




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