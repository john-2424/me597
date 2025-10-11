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
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist, Point
from std_msgs.msg import Float32

from task_4 import MapProcessor, AStar, Follower, PID
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
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0
        self.goal_updated = True

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
        self.stop_tol = 0.12  # stop when close to waypoint
        self.ema_alpha = 0.35 # smoother speed estimate

        # PIDs (start conservative; tune later)
        self.pid_speed = PID(kp=1.0, ki=0.2, kd=0.05, i_limit=0.6, out_limit=(-0.7, 0.7))   # output = linear.x (m/s)
        self.pid_heading = PID(kp=2.0, ki=0.0, kd=0.10, i_limit=0.8, out_limit=(-1.5, 1.5))  # output = angular.z (rad/s)

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

        self.flw = Follower()

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

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path = Path()
        self.get_logger().info('A* Planner ->')
        # self.get_logger().info(
        #     'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        path.poses.append(start_pose)
        
        start_img_pose_u, start_img_pose_v = self.__map_pose_real_to_img(start_pose.pose.position.x, start_pose.pose.position.y)
        start_pose_u_v = f'{start_img_pose_u},{start_img_pose_v}'
        self.get_logger().info(f'[START] Real: {start_pose.pose.position.x},{start_pose.pose.position.y} :: Img: {start_pose_u_v}')
        self.mp.map_graph.root = start_pose_u_v
        spxy_mp_node = self.mp.map_graph.g[start_pose_u_v]
        end_img_pose_u, end_img_pose_v = self.__map_pose_real_to_img(end_pose.pose.position.x, end_pose.pose.position.y)
        end_pose_u_v = f'{end_img_pose_u},{end_img_pose_v}'
        self.get_logger().info(f'[END] Real: {end_pose.pose.position.x},{end_pose.pose.position.y} :: Img: {end_pose_u_v}')
        if end_pose_u_v in self.mp.map_graph.g:
            self.mp.map_graph.end = end_pose_u_v
            epxy_mp_node = self.mp.map_graph.g[end_pose_u_v]
            
            astar_graph = AStar(self.mp.map_graph)
            astar_graph.solve(spxy_mp_node, epxy_mp_node)
            path_as, dist_as = astar_graph.reconstruct_path(spxy_mp_node, epxy_mp_node)
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
                path.poses.append(
                    path_taken_pose
                )
            
            map_with_path_as = self.mp.draw_path(path_as)
            fig, ax = plt.subplots(nrows = 1, ncols = 1, dpi=300, sharex=True, sharey=True)
            ax.imshow(map_with_path_as)
            ax.set_title('Path A*')
            plt.show()
            # plt.show(block=False)
        else:
            self.get_logger().warn(f'Goal position does not exist!')
        
        path.poses.append(end_pose)
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        return path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  vehicle_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        idx = 0
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        idx = self.flw.get_path_idx(path, vehicle_pose, now_sec)
        return idx

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

        # heading setpoint: we want heading_err → 0 (so setpoint is 0)
        heading_err_sp = 0.0  # track zero heading error

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

        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()

        speed = float(np.clip(speed,  -self.speed_max, self.speed_max))
        heading = float(np.clip(heading, -self.heading_max, self.heading_max))

        if abs(speed) < self.speed_db: speed = 0.0
        if abs(heading) < self.heading_db: heading = 0.0

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

            # 1. Create the path to follow
            if self.goal_updated:
                self.path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                self.goal_updated = False
            if len(self.path.poses) >= 1:
                # 2. Loop through the path and move the robot
                idx = self.get_path_idx(self.path, self.ttbot_pose)
                idx = min(max(idx, 0), len(self.path.poses) - 1)   # clamp
                current_goal = self.path.poses[idx]
                speed, heading = self.path_follower(self.ttbot_pose, current_goal)
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