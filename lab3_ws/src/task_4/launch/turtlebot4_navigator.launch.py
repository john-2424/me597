from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    slam_launch_dir = PathJoinSubstitution([FindPackageShare('turtlebot4_navigation'), 'launch'])

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='src/task_4/maps/classroom_map.yaml'),
        DeclareLaunchArgument('namespace', default_value='/robot'),

        # include SLAM/Localization launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([slam_launch_dir, 'localization.launch.py']),
            ),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'namespace': LaunchConfiguration('namespace'),
            }.items(),
        ),

        # Include RViz visualization launch (to view map and robot)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                'view_robot.launch.py'
            ),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'namespace': LaunchConfiguration('namespace'),
            }.items(),
        ),

        # Run the auto_navigator node
        Node(
            package='task_4',
            executable='auto_navigator',
            name='auto_navigator',
            output='screen',
            parameters=[{
                'map': LaunchConfiguration('map'),
                'namespace': LaunchConfiguration('namespace'),
            }]
        ),
    ])