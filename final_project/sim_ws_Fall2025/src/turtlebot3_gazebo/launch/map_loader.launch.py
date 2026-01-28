from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    maps_share_directory = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'maps')
    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'yaml_filename': os.path.join(maps_share_directory, 'elmapa.yaml')
            }]
        )
    ])
