from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # start the image_publisher node
        Node(
            package='task_5',
            executable='image_publisher',
            name='image_publisher'
        ),

        # start the object_detector node
        Node(
            package='task_5',
            executable='object_detector',
            name='object_detector'
        ),
    ])