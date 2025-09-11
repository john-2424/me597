from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # start the basic_publisher node in a namespace
        Node(
            package='task_1',
            namespace='basics',
            executable='talker',
            name='talker'
        ),

        # start the basic_subscriber node in a namespace
        Node(
            package='task_1',
            namespace='basics',
            executable='listener',
            name='listener'
        ),
    ])