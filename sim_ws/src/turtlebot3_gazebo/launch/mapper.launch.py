import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get shared directories
    package_dir = get_package_share_directory('turtlebot3_gazebo')
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')

    # Launch file paths
    env_launch_file = os.path.join(package_dir, 'launch', 'turtlebot3_house.launch.py')
    bonus_launch_file = os.path.join(package_dir, 'launch', 'turtlebot3_bonus_world.launch.py')
    slam_params_file = os.path.join(package_dir, 'params', 'mapper_params_online_async.yaml')

    # Launch Configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    bonus = LaunchConfiguration('bonus')

    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument('bonus', default_value='false', description='Launch bonus world if true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),

        # Include default environment if bonus is false
        GroupAction([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(env_launch_file),
                launch_arguments={'use_rviz=': use_rviz}.items()
            )
        ], condition=UnlessCondition(bonus)),

        # Include bonus environment if bonus is true
        GroupAction([
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bonus_launch_file),
                launch_arguments={'use_rviz': use_rviz}.items()
            )
        ], condition=IfCondition(bonus)),

        # Include SLAM Toolbox
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'slam_params_file': slam_params_file
            }.items()
        ),

        # Launch mapping algorithm
        Node(
            package='turtlebot3_gazebo',
            executable='task1',
            name='task1_algorithm',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]
        ),
    ])
