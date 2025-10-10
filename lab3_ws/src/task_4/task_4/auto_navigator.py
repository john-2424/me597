#!/usr/bin/env python3

import sys
import os
import numpy as np
from pathlib import Path as os_path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist, Point
from std_msgs.msg import Float32

from task_4 import MapProcessor
from task_4 import AStar


class Navigation(Node):
    """! Navigation node class.
    This class should serve as a template to implement the path planning and
    path follower components to move the turtlebot from position A to B.
    """

    def __init__(self, node_name='Navigation'):
        """! Class constructor.
        @param  None.
        @return An instance of the Navigation class.
        """
        super().__init__(node_name)
        # Path planner/follower related variables
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.start_time = 0.0
        self.start_real_pose = 0, 0
        self.start_img_pose = self.__map_pose_real_to_img(self.start_real_pose)

        # ===== Parameters (declare + defaults) =====
        self.declare_parameter('map_name', 'sync_classroom_map')          # Name of the map to navigate
        self.declare_parameter('kernel_size', 5)          # Size of the kernel, to configure how much you want to inflate map/obstacles
        # ===== Get params =====
        self.map_name   = float(self.get_parameter('map_name').value)
        self.kernel_size   = float(self.get_parameter('kernel_size').value)

        # Generate Graph from Map
        map_file_path = os.path.join(os_path(os.path.abspath(__file__)).resolve().parent.parent, 'maps', self.map_name)
        self.mp = MapProcessor(map_file_path)
        kr = self.mp.rect_kernel(self.kernel_size, 1)
        self.mp.inflate_map(kr, True)
        self.mp.get_graph_from_map()
        self.map_res = self.mp.map.map_df.resolution[0]
        self.map_origin = self.mp.map.map_df.origin[0]
        self.map_img_array_shape = self.mp.map.image_array.shape

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
        return self.map_img_array_shape[0] - ((real_x + abs(self.map_origin[0]))//self.map_res), self.map_img_array_shape[1] - ((real_y + abs(self.map_origin[1]))//self.map_res)

    def __map_pose_img_to_real(self, img_x, img_y):
        return ((self.map_img_array_shape[0] - img_x)*self.map_res) - abs(self.map_origin[0]), ((self.map_img_array_shape[1] - img_y)*self.map_res) - abs(self.map_origin[1])

    def a_star_path_planner(self, start_pose, end_pose):
        """! A Start path planner.
        @param  start_pose    PoseStamped object containing the start of the path to be created.
        @param  end_pose      PoseStamped object containing the end of the path to be created.
        @return path          Path object containing the sequence of waypoints of the created path.
        """
        path = Path()
        self.get_logger().info(
            'A* planner.\n> start: {},\n> end: {}'.format(start_pose.pose.position, end_pose.pose.position))
        self.start_time = self.get_clock().now().nanoseconds*1e-9 #Do not edit this line (required for autograder)
        # TODO: IMPLEMENTATION OF THE A* ALGORITHM
        path.poses.append(start_pose)
        
        start_img_pose_x, start_img_pose_y = self.__map_pose_real_to_img(start_pose.pose.position.x, start_pose.pose.position.y)
        start_pose_x_y = f'{start_img_pose_x},{start_img_pose_y}'
        self.mp.map_graph.root = start_pose_x_y
        spxy_mp_node = self.mp.map_graph.g[start_pose_x_y]
        end_img_pose_x, end_img_pose_y = self.__map_pose_real_to_img(end_pose.pose.position.x, end_pose.pose.position.y)
        end_pose_x_y = f'{end_img_pose_x},{end_img_pose_y}'
        self.mp.map_graph.end = end_pose_x_y
        epxy_mp_node = self.mp.map_graph.g[end_pose_x_y]
        
        astar_graph = AStar(self.mp.map_graph)
        astar_graph.solve(spxy_mp_node, epxy_mp_node)
        path_as, dist_as = astar_graph.reconstruct_path(spxy_mp_node, epxy_mp_node)
        for path_taken in path_as[1:-1]:
            path_taken_x, path_taken_y = path_taken.split(',')
            path_taken_real_pose_x, path_taken_real_pose_y = self.__map_pose_img_to_real(path_taken_x, path_taken_y)
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

        path.poses.append(end_pose)
        # Do not edit below (required for autograder)
        self.astarTime = Float32()
        self.astarTime.data = float(self.get_clock().now().nanoseconds*1e-9-self.start_time)
        self.calc_time_pub.publish(self.astarTime)
        
        return path

    def get_path_idx(self, path, vehicle_pose):
        """! Path follower.
        @param  path                  Path object containing the sequence of waypoints of the created path.
        @param  current_goal_pose     PoseStamped object containing the current vehicle position.
        @return idx                   Position in the path pointing to the next goal pose to follow.
        """
        idx = 0
        # TODO: IMPLEMENT A MECHANISM TO DECIDE WHICH POINT IN THE PATH TO FOLLOW idx <= len(path)
        return idx

    def path_follower(self, vehicle_pose, current_goal_pose):
        """! Path follower.
        @param  vehicle_pose           PoseStamped object containing the current vehicle pose.
        @param  current_goal_pose      PoseStamped object containing the current target from the created path. This is different from the global target.
        @return path                   Path object containing the sequence of waypoints of the created path.
        """
        speed = 0.0
        heading = 0.0
        # TODO: IMPLEMENT PATH FOLLOWER
        return speed, heading

    def move_ttbot(self, speed, heading):
        """! Function to move turtlebot passing directly a heading angle and the speed.
        @param  speed     Desired speed.
        @param  heading   Desired yaw angle.
        @return path      object containing the sequence of waypoints of the created path.
        """
        cmd_vel = Twist()
        # TODO: IMPLEMENT YOUR LOW-LEVEL CONTROLLER
        cmd_vel.linear.x = speed
        cmd_vel.angular.z = heading

        self.cmd_vel_pub.publish(cmd_vel)

    def run(self):
        """! Main loop of the node. You need to wait until a new pose is published, create a path and then
        drive the vehicle towards the final pose.
        @param none
        @return none
        """
        while rclpy.ok():
            # Call the spin_once to handle callbacks
            rclpy.spin_once(self, timeout_sec=0.1)  # Process callbacks without blocking

            # 1. Create the path to follow
            # path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
            # 2. Loop through the path and move the robot
            # idx = self.get_path_idx(path, self.ttbot_pose)
            # current_goal = path.poses[idx]
            # speed, heading = self.path_follower(self.ttbot_pose, current_goal)
            # self.move_ttbot(speed, heading)

            self.rate.sleep()
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