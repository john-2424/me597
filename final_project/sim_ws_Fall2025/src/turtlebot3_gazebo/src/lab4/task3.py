#!/usr/bin/env python3

import os
import math

import rclpy
from rclpy.node import Node

# ROS message types
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

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
    - Layer 2: Automatic waypoint generation & route optimization
      using distance-transform maxima (geometry-based coverage).
    """

    def __init__(self):
        super().__init__('task3_node')

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
        self.dynamic_occupancy = None  # 0 free, 1 occupied (from LaserScan; later)

        # -------------------------
        # Robot state
        # -------------------------
        self.current_pose = None          # PoseWithCovarianceStamped
        self.latest_scan = None           # LaserScan
        self.latest_image = None          # Image (OpenCV conversion later)

        # -------------------------
        # Waypoints & navigation (Layer 2 and later)
        # -------------------------
        self.patrol_waypoints = []        # list of (x, y) in map frame (ordered)
        self.current_waypoint_idx = 0
        self.waypoints_generated = False  # flag to avoid regenerating

        self.global_path_points = []      # for actual A* path (later)
        self.active_path_points = []      # for local+global nav (later)
        self.current_path_index = 0

        # -------------------------
        # High-level task state machine
        # -------------------------
        self.state = 'WAIT_FOR_POSE'      # will evolve in later layers

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

        # Future A* global path (keep reserved)
        self.global_path_pub = self.create_publisher(
            Path,
            self._ns_topic('global_plan'),
            10
        )

        # Local path (later)
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

        self.get_logger().info('Task3 node initialized (Layers 0 + 1 + 2).')

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
        """
        Use distance-transform maxima over waypoint_free_mask to generate
        room-covering waypoints:

        1) waypoint_free_mask = free & inflated-free
        2) distanceTransform
        3) pick local maxima above a distance threshold
        4) convert to world, prune by distance
        5) route via NN + 2-opt
        """
        waypoint_free_mask = self._compute_waypoint_free_mask()
        if waypoint_free_mask is None or waypoint_free_mask.sum() == 0:
            self.get_logger().warn(
                '[Layer 2] waypoint_free_mask has no free cells. '
                'No patrol waypoints will be generated.'
            )
            self.patrol_waypoints = []
            return

        # --- Distance transform (in cells) ---
        dist = cv2.distanceTransform(waypoint_free_mask, cv2.DIST_L2, 3)
        min_dt = float(dist[waypoint_free_mask == 1].min()) if waypoint_free_mask.sum() > 0 else 0.0
        max_dt = float(dist.max())
        self.get_logger().info(
            f'[Layer 2] DistanceTransform stats (cells): '
            f'min={min_dt:.2f}, max={max_dt:.2f}'
        )

        # Ignore tiny pockets very close to walls (in cells)
        min_dist_cells = max(1, int(self.l2_min_dt_cells))
        candidate_mask = np.logical_and(
            waypoint_free_mask == 1,
            dist >= float(min_dist_cells)
        )

        self.get_logger().info(
            f'[Layer 2] Candidate cells after DT threshold >= {min_dist_cells}: '
            f'{int(candidate_mask.sum())}'
        )

        # --- Local maxima detection ---
        kernel = np.ones((3, 3), np.uint8)
        dist_dilated = cv2.dilate(dist, kernel)

        local_max_mask = np.logical_and(
            candidate_mask,
            dist >= dist_dilated - 1e-6  # float tolerance
        )

        ys, xs = np.where(local_max_mask)
        self.get_logger().info(
            f'[Layer 2] Distance-transform local maxima count (grid): {len(xs)}'
        )

        if len(xs) == 0:
            self.get_logger().warn(
                '[Layer 2] No local maxima found. Consider lowering l2_min_dt_cells.'
            )

        # Convert to world coordinates
        raw_world_points = []
        for r, c in zip(ys, xs):
            world_xy = self.grid_to_world(int(r), int(c))
            if world_xy is not None:
                raw_world_points.append(world_xy)

        self.get_logger().info(
            f'[Layer 2] Raw world waypoints from DT maxima: {len(raw_world_points)}'
        )

        # Debug: log bounding box and extreme points of raw_world_points
        if raw_world_points:
            xs_w = [p[0] for p in raw_world_points]
            ys_w = [p[1] for p in raw_world_points]
            self.get_logger().info(
                f'[Layer 2] Raw waypoint world bounds: '
                f'x in [{min(xs_w):.2f}, {max(xs_w):.2f}], '
                f'y in [{min(ys_w):.2f}, {max(ys_w):.2f}]'
            )

            # log the "most bottom-right" candidate (max x, min y)
            idx_br = max(
                range(len(raw_world_points)),
                key=lambda i: (raw_world_points[i][0], -raw_world_points[i][1])
            )
            br_x, br_y = raw_world_points[idx_br]
            self.get_logger().info(
                f'[Layer 2] Bottom-right-ish DT candidate: ({br_x:.2f}, {br_y:.2f})'
            )

            # Log up to first 5 points for sanity
            sample_n = min(5, len(raw_world_points))
            msg_pts = ', '.join(
                f'({raw_world_points[i][0]:.2f},{raw_world_points[i][1]:.2f})'
                for i in range(sample_n)
            )
            self.get_logger().info(
                f'[Layer 2] Sample raw DT waypoints (first {sample_n}): {msg_pts}'
            )

        # Prune near-duplicates in world space
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

        # Route ordering: nearest neighbor + 2-opt
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

        # Route length just for logging
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

        # Extra debug: print ordered waypoints
        lines = []
        for i, (wx, wy) in enumerate(self.patrol_waypoints):
            lines.append(f'  #{i:02d}: ({wx:.2f}, {wy:.2f})')
        self.get_logger().info(
            '[Layer 2] Final ordered patrol waypoints:\n' + '\n'.join(lines)
        )

    def _prune_waypoints_by_distance(self, points, d_min):
        pruned = []
        for (x, y) in points:
            keep = True
            for (px, py) in pruned:
                if math.hypot(x - px, y - py) < d_min:
                    keep = False
                    break
            if keep:
                pruned.append((x, y))
        return pruned

    def _nearest_neighbor_route(self, points, start_xy):
        n = len(points)
        if n == 0:
            return []

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

        return route

    def _two_opt_improvement(self, points, route, max_iters):
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
            f'Route length over points ≈ {best_length:.2f}'
        )

        return best_route

    def _publish_waypoints_path(self, waypoints):
        if not waypoints:
            return

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
            return

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
    # Subscriber callbacks (Layer 0)
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

    def image_callback(self, msg: Image):
        self.latest_image = msg

    # ----------------------------------------------------------------------
    # Timer callback / state machine skeleton
    # ----------------------------------------------------------------------
    def timer_cb(self):
        self.get_logger().info(
            f'[Layer 0-2] State = {self.state}, '
            f'map_loaded={self.map_loaded}, '
            f'pose_received={self.current_pose is not None}, '
            f'scan_received={self.latest_scan is not None}, '
            f'image_received={self.latest_image is not None}, '
            f'waypoints_generated={self.waypoints_generated}',
            throttle_duration_sec=1.0
        )

        self._generate_patrol_waypoints_if_needed()
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
