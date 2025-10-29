#!/usr/bin/env python3

import sys
import os
from math import sqrt, atan2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as os_path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped, Pose, Twist, Point
from std_msgs.msg import Float32

from task_4 import MapProcessor, AStar, WayPoints, PID
from task_4.utils.etc import *

class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  node_name.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Path planner/follower related variables
        self.path = None
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0
        self.goal_updated = False

        # ===== Parameters (declare + defaults) =====
        self.declare_parameter('map_name', 'sync_classroom_map')          # Name of the map to navigate
        self.declare_parameter('kernel_size', 12)          # Size of the kernel, to configure how much you want to inflate map/obstacles
        # ===== Get params =====
        self.map_name   = str(self.get_parameter('map_name').value)
        self.kernel_size   = int(self.get_parameter('kernel_size').value)

        # speed estimate state
        self._speed_hist = []  # list of (t, x, y)
        self._ema_speed = 0.0

        self.speed_max = 0.5
        self.speed_min = 0.0 
        self.heading_max = 1.2
        self.speed_db  = 0.02
        self.heading_db  = 0.02
        self.slow_k = 1.2     # slow down when misaligned
        self.stop_tol = 0.3  # stop when close to waypoint
        self.ema_alpha = 0.35 # smoother speed estimate
        self.yaw_tol = 0.05            # ~3 degrees; stop when aligned
        self.last_ctrl_time = None     # for dt in final heading PID

        # PIDs (start conservative; tune later)
        self.pid_speed = PID(
            kp=2.0, ki=0.02, kd=0.01, 
            i_limit=0.8, 
            out_limit=(-self.speed_max, self.speed_max)
        )   # output = linear.x (m/s)
        self.pid_heading = PID(
            kp=2.0, ki=0.02, kd=0.10, 
            i_limit=0.8, 
            out_limit=(-self.heading_max, self.heading_max)
        )  # output = angular.z (rad/s)
        self.pid_yaw_final = PID(
            kp=2.2, ki=0.02, kd=0.10,
            i_limit=0.6,
            out_limit=(-self.heading_max, self.heading_max)
        )  # output = angular.z (rad/s)

        # Generate Graph from Map
        map_file_path = os.path.join(os_path(os.path.abspath(__file__)).resolve().parent.parent, 'maps', self.map_name)
        self.mp = MapProcessor(map_file_path)
        kr = self.mp.rect_kernel(self.kernel_size, 1)
        self.mp.inflate_map(kr, True)
        self.mp.get_graph_from_map()
        self.map_res = self.mp.map.map_df.resolution[0]
        self.map_origin = self.mp.map.map_df.origin[0]
        self.map_img_array_shape = self.mp.map.image_array.shape
        self.start_real_pose = 0, 0
        self.start_img_pose = self.__map_pose_real_to_img(*self.start_real_pose)
        self.get_logger().info(f'Start Pose [Img]: {self.start_img_pose}')

        # self.flw = Follower()
        self.wps = None

        # Subscribers
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.calc_time_pub = self.create_publisher(Float32, 'astar_time',10) #DO NOT MODIFY

        # Node rate
        self.rate = self.create_rate(10)

    def __goal_pose_cbk(self, data):
        """! Callback to catch the goal pose.
        @param  data    PoseStamped object from RVIZ.
        @return None.
        """
        self.goal_pose = data
        self.goal_updated = True
        self.get_logger().info(
            'goal_pose: {:.4f}, {:.4f}'.format(self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))

    def __ttbot_pose_cbk(self, data):
        """! Callback to catch the position of the vehicle.
        @param  data    PoseWithCovarianceStamped object from amcl.
        @return None.
        """
        self.ttbot_pose = data.pose
        self.get_logger().info(
            'ttbot_pose: {:.4f}, {:.4f}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y))
    
    def __map_pose_real_to_img(self, real_x, real_y):
        return int(self.map_img_array_shape[0] - 1 - ((real_y - self.map_origin[1])/self.map_res)), int(((real_x - self.map_origin[0])/self.map_res))

    def __map_pose_img_to_real(self, img_u, img_v):
        return (int(img_v)*self.map_res) + self.map_origin[0], ((self.map_img_array_shape[0] - int(img_u) - 1)*self.map_res) + self.map_origin[1]

    def _as_ps(self, src, frame="map"):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = frame

        if isinstance(src, PoseStamped):
            # Optionally refresh header/frame to your current clock/frame
            return src

        if isinstance(src, PoseWithCovarianceStamped):
            pose = src.pose.pose
            ps.pose = Pose()
            ps.pose.position.x = float(pose.position.x)
            ps.pose.position.y = float(pose.position.y)
            ps.pose.position.z = float(pose.position.z)
            ps.pose.orientation = pose.orientation
            return ps

        if isinstance(src, PoseWithCovariance):
            pose = src.pose  # <-- the inner Pose
            ps.pose = Pose()
            ps.pose.position.x = float(pose.position.x)
            ps.pose.position.y = float(pose.position.y)
            ps.pose.position.z = float(pose.position.z)
            ps.pose.orientation = pose.orientation
            return ps

        if isinstance(src, Pose):
            ps.pose = src
            return ps

        if isinstance(src, tuple) and len(src) == 2:
            x, y = src
            ps.pose = Pose(position=Point(x=float(x), y=float(y)))
            return ps

        raise TypeError(f"Cannot coerce {type(src)} to PoseStamped")

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        self.path = Path()
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.header.frame_id = "map"

        self.get_logger().info('A* Planner ->')
        # self.get_logger().info(
        #     'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)

        # Normalize end points
        start_ps = self._as_ps(start_pose)
        end_ps   = self._as_ps(end_pose)
        self.path.poses.append(start_ps)
        
        start_img_pose_u, start_img_pose_v = self.__map_pose_real_to_img(start_pose.pose.position.x, start_pose.pose.position.y)
        start_pose_u_v = f'{start_img_pose_u},{start_img_pose_v}'
        self.get_logger().info(f'[START] Real: {start_pose.pose.position.x},{start_pose.pose.position.y} :: Img: {start_pose_u_v}')
        
        if start_pose_u_v not in self.mp.map_graph.g:
            # Find the nearest node on the grid to the bot
            u0, v0 = int(round(start_img_pose_u)), int(round(start_img_pose_v))
            found = False
            max_radius = 25

            for r in range(1, max_radius + 1):
                for du in range(-r, r + 1):
                    for dv in range(-r, r + 1):
                        u, v = u0 + du, v0 + dv
                        candidate = f"{u},{v}"
                        if candidate in self.mp.map_graph.g:
                            start_pose_u_v = candidate
                            found = True
                            self.get_logger().warn(f"[START snapped] to nearby node {candidate} at radius {r}")
                            break
                    if found:
                        break
                if found:
                    break

            if not found:
                self.get_logger().error("[START] No nearby node found within search radius!")
        
        spxy_mp_node = self.mp.map_graph.g[start_pose_u_v]
        self.mp.map_graph.root = start_pose_u_v
        end_img_pose_u, end_img_pose_v = self.__map_pose_real_to_img(end_pose.pose.position.x, end_pose.pose.position.y)
        end_pose_u_v = f'{end_img_pose_u},{end_img_pose_v}'
        self.get_logger().info(f'[END] Real: {end_pose.pose.position.x},{end_pose.pose.position.y} :: Img: {end_pose_u_v}')
        if end_pose_u_v in self.mp.map_graph.g:
            epxy_mp_node = self.mp.map_graph.g[end_pose_u_v]
            self.mp.map_graph.end = end_pose_u_v
            
            astar_graph = AStar(self.mp.map_graph)
            astar_graph.solve(spxy_mp_node, epxy_mp_node)
            try:
                path_as, dist_as = astar_graph.reconstruct_path(spxy_mp_node, epxy_mp_node)
            except KeyError:
                self.get_logger().warn(f'Goal is not reachable!')
                self.path = None
                return
            self.get_logger().info(f'[Distance]: {dist_as}')
            for path_taken in path_as[1:-1]:
                path_taken_u, path_taken_v = path_taken.split(',')
                path_taken_real_pose_x, path_taken_real_pose_y = self.__map_pose_img_to_real(path_taken_u, path_taken_v)
                self.get_logger().info(f'[PATH] Real: {path_taken_real_pose_x},{path_taken_real_pose_y} :: Img: {path_taken}')
                path_taken_pose = PoseStamped(
                    pose=Pose(
                        position=Point(
                            x=path_taken_real_pose_x,
                            y=path_taken_real_pose_y
                        )
                    )
                )
                path_taken_pose = self._as_ps(
                    (path_taken_real_pose_x, path_taken_real_pose_y)
                )
                self.path.poses.append(
                    path_taken_pose
                )
            
            # map_with_path_as = self.mp.draw_path(path_as)
            # fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi=300, sharex=True, sharey=True)
            # ax.imshow(map_with_path_as)
            # ax.set_title('Path A*')
            # plt.show()
            # # plt.show(block=False)
        else:
            self.get_logger().warn(f'Goal position does not exist!')
            self.path = None
            return
        
        self.path.poses.append(end_ps)
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        # for i, p in enumerate(self.path.poses):
        #     self.get_logger().info(
        #         f"[poses[{i}]] type={p.__class__.__module__}.{p.__class__.__name__}"
        #     )

        return self.path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        # idx = 0
        # now_sec = self.get_clock().now().nanoseconds * 1e-9
        # idx = self.flw.get_path_idx(path, vehicle_pose, now_sec)

        self.wps.choose_next(vehicle_pose)

        # return idx

    def _final_heading_controller(self, vehicle_pose, goal_pose):
        """Rotate in place to match the goal's yaw. Returns angular.z.
        Linear.x should be 0 when calling this.
        """
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self.last_ctrl_time is None:
            dt = 1.0 / 10.0  # same as node rate fallback
        else:
            dt = max(1e-3, now_sec - self.last_ctrl_time)
        self.last_ctrl_time = now_sec

        # Current yaw and target yaw
        ryaw = yaw_from_quat(vehicle_pose.pose.orientation)
        gyaw = yaw_from_quat(goal_pose.pose.orientation)

        err = wrap_pi(gyaw - ryaw)
        if abs(err) < self.yaw_tol:
            self.pid_yaw_final.reset()
            return 0.0

        # PID purely on yaw error; only angular.z is commanded at the end
        ang_out = self.pid_yaw_final.step(err, dt)
        return float(np.clip(ang_out, -self.heading_max, self.heading_max))

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        speed = 0.0
        heading = 0.0

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        rx = vehicle_pose.pose.position.x
        ry = vehicle_pose.pose.position.y
        ryaw = yaw_from_quat(vehicle_pose.pose.orientation)

        gx = current_goal_pose.pose.position.x
        gy = current_goal_pose.pose.position.y

        dx, dy = gx - rx, gy - ry
        dist = sqrt(dx*dx + dy*dy)
        if dist <= self.stop_tol:
            # reset integrators to avoid "creep"
            self.pid_speed.reset()
            self.pid_heading.reset()
            # self.get_logger().info('Reset integrators to avoid creep')
            return 0.0, 0.0

        goal_bearing = atan2(dy, dx)
        heading_err = wrap_pi(goal_bearing - ryaw)  # desired heading is goal_bearing → error to zero

        # keep ~0.5–1.0 s of history; here we auto-size to timer rate
        self._speed_hist.append((now_sec, rx, ry))
        # keep last ~1 s
        while len(self._speed_hist) > 0 and (now_sec - self._speed_hist[0][0]) > 1.0:
            self._speed_hist.pop(0)

        speed_meas = 0.0
        if len(self._speed_hist) >= 2:
            t0, x0, y0 = self._speed_hist[0]
            dt_hist = max(1e-3, now_sec - t0)
            speed_raw = dist2d((rx, ry), (x0, y0)) / dt_hist
            self._ema_speed = self.ema_alpha*speed_raw + (1.0-self.ema_alpha)*self._ema_speed
            speed_meas = self._ema_speed

        # speed setpoint: proportional to distance, reduced when misaligned
        speed_sp = (dist) / (1.0 + self.slow_k*abs(heading_err))
        speed_sp = float(np.clip(speed_sp, self.speed_min, self.speed_max))

        # Speed PID tracks speed_sp using measured speed speed_meas.
        # error = speed_sp - speed_meas
        # dt from last control step ~ timer period; here use last two timestamps if possible
        if len(self._speed_hist) >= 2:
            dt = now_sec - self._speed_hist[-2][0]
        else:
            dt = 1.0 / 10.0  # fallback 10 Hz

        speed_out = self.pid_speed.step(speed_sp - speed_meas, dt)
        heading_out = self.pid_heading.step(heading_err, dt)  # error = 0 - heading_err

        # ---------- saturate for safety ----------
        speed = float(np.clip(speed_out, -self.speed_max, self.speed_max))
        heading = float(np.clip(heading_out, -self.heading_max, self.heading_max))

        # if abs(speed) < self.speed_db: speed = 0.0
        # if abs(heading) < self.heading_db: heading = 0.0

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()

        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        # Shows the nodes on the Map
        # map_with_nodes = self.mp.draw_nodes()
        # fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi=300, sharex=True, sharey=True)
        # ax.imshow(map_with_nodes)
        # ax.set_title('Nodes')
        # plt.show()
        # # plt.show(block=False)
        
        while rclpy.ok():
            # Call the spin_once to handle callbacks
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks without blocking

            # 1. Create the path to follow, when either goal is updated or when the ttbot_pose is not within the tolerane of the previously generated A* path
            if self.goal_updated or (self.wps is not None and not self.wps.bot_on_path(self.ttbot_pose)):
                self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                # self.get_logger().info(f'Path Generated: {type(self.path)}')
                if self.path is not None:
                    self.path_pub.publish(self.path)
                    self.wps = WayPoints(self.path)
                self.goal_updated = False
            
            # 2. Loop through the path and move the robot
            if self.wps is not None and self.wps.bot_reached(self.ttbot_pose):
                if not self.wps.final_idx:
                    self.get_path_idx(self.path, self.ttbot_pose)
                # idx = min(max(idx, 0), len(self.path.poses) - 1)   # clamp
            
            if self.wps is not None and self.path is not None:
                current_goal = self.path.poses[self.wps.last_idx]

                if self.wps.final_idx and self.wps.bot_reached(self.ttbot_pose):
                    orient = self._final_heading_controller(self.ttbot_pose, self.goal_pose)
                    self.get_logger().info(f'Final Heading/Orientation: {orient}')
                    self.move_ttbot(0.0, orient)

                    # Full stop
                    if orient == 0.0:
                        self.move_ttbot(0.0, 0.0)
                        self.wps = None
                        self.get_logger().info(f'You have arrived at your destination!! \nGoal Pose: Position: x -> {self.goal_pose.pose.position.x} ; y -> {self.goal_pose.pose.position.y} ;; Orientation: z -> {self.goal_pose.pose.orientation.z} \nTTBot Pose: Position: x -> {self.ttbot_pose.pose.position.x} ; y -> {self.ttbot_pose.pose.position.y} ;; Orientation: z -> {self.ttbot_pose.pose.orientation.z}')
                else:
                    # self.get_logger().info(f'Current Goal: {current_goal}')
                    speed, heading = self.path_follower(self.ttbot_pose, current_goal)
                    self.get_logger().info(f'Speed: {speed}; Heading: {heading}')
                    self.move_ttbot(speed, heading)

            # self.rate.sleep()
            # Sleep for the rate to control loop timing


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'Navigation' class node
    nav = Navigation(node_name='Navigation')

    # Spin up the created node
    try:
        # rclpy.spin(nav)
        nav.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Explicit declaration to destroy the node object and shutdown rclpy
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()