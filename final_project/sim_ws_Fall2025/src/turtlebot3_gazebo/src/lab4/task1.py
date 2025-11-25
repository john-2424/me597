#!/usr/bin/env python3

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
    main()



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