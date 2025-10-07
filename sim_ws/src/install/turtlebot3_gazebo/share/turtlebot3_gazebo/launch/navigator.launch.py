import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    # Set the directory paths
    package_dir = get_package_share_directory('turtlebot3_gazebo')
    map_file_path = os.path.join(package_dir, 'maps', 'map.yaml')

    # Declare launch arguments
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'map',
            default_value=map_file_path,
            description='Full path to map file to load'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to start RViz'
        ),
        DeclareLaunchArgument(
            'static_obstacles',
            default_value='false',
            description='Whether to enable static obstacles'
        ),
        DeclareLaunchArgument(
            'spawn_objects',
            default_value='true',
            description='Whether to spawn objects'
        ),

        # Start simulation environment and turtlebot
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(package_dir, 'launch', 'turtlebot3_house.launch.py')),
            launch_arguments={'use_rviz': LaunchConfiguration('use_rviz')}.items()
        ),

        # Conditionally enable dynamics obstacles if the argument is set to "true"
        Node(
            package='turtlebot3_gazebo',
            executable='static_obstacles.py',
            name='static_obstacles',
            output='screen',
            condition=IfCondition(LaunchConfiguration('static_obstacles'))
        ),

        Node(
            package='turtlebot3_gazebo',
            executable='spawn_objects.py',
            name='spawn_objects',
            output='screen',
            condition=IfCondition(LaunchConfiguration('spawn_objects'))
        ),

        # Map Server Node
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            # Neglect the map_server.yaml, which always requires absolute path for map file
            parameters=[{
                'yaml_filename': map_file_path,
                'frame_id': 'map',
                'topic_name': 'map',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }]
        ),

        # AMCL (Adaptive Monte Carlo Localization) Node
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(package_dir, 'launch', 'amcl.launch.py')),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
        ),

        # Lifecycle Manager Node for map server and amcl
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'autostart': True,
                'node_names': ['map_server', 'amcl']
            }]
        ),

        # Task2 Algorithm Node
        Node(
            package='turtlebot3_gazebo',
            executable='task2.py',
            name='task2_algorithm',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }],
            condition=IfCondition(LaunchConfiguration('static_obstacles'))
        ),

        # Task3 Algorithm Node (for dynamic obstacles)
        Node(
            package='turtlebot3_gazebo',
            executable='task3.py',
            name='task3_algorithm',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time')
            }],
            condition=IfCondition(LaunchConfiguration('spawn_objects'))
        )
    ])
