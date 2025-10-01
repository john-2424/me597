import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, SetLaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Find the current package directory
    package_dir = get_package_share_directory('turtlebot3_gazebo')

    # Get the path to the environment launch file
    env_launch_file = os.path.join(package_dir, 'launch', 'turtlebot3_house.launch.py')
    slam_params_file = os.path.join(package_dir, 'params', 'mapper_params_online_async.yaml')

    # Define configurable parameters
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')

    return LaunchDescription([
        # Launch environment and turtlebot
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(env_launch_file),
            launch_arguments={'use_rviz': use_rviz}.items(),
        ),

        # Launch SLAM for mapping
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('slam_toolbox'),
                '/launch/online_async_launch.py'
            ]),
            launch_arguments={'use_sim_time': use_sim_time,
                              'slam_params_file': slam_params_file}.items()
        ),

        # Launch your mapping algorithm
        Node(
            package='turtlebot3_gazebo',
            executable='task1.py',
            name='task1_algorithm',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }]
        ),
    ])
