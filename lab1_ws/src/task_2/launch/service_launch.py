from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # start the basic_publisher node in a namespace
        Node(
            package='task_2',
            # namespace='basics',
            executable='talker',
            name='talker'
        ),

        # start the basic_service node in a namespace
        Node(
            package='task_2',
            # namespace='basics',
            executable='service',
            name='service'
        ),
    ])