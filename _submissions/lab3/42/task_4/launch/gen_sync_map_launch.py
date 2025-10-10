from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch_ros.actions import PushRosNamespace
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam_launch_dir = PathJoinSubstitution([FindPackageShare('turtlebot4_navigation'), 'launch'])
    rviz_launch_dir = PathJoinSubstitution([FindPackageShare('turtlebot4_viz'), 'launch'])
    return LaunchDescription([
        # DeclareLaunchArgument('arg', default_value='val'),

        # include SLAM launch file
        IncludeLaunchDescription(
            PathJoinSubstitution([slam_launch_dir, 'slam.launch.py'])
        ),

        # include RViz launch file
        IncludeLaunchDescription(
            PathJoinSubstitution([rviz_launch_dir, 'view_robot.launch.py'])
        ),
    ])