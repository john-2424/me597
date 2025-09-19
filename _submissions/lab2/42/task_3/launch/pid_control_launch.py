from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('setpoint_reference', default_value='0.35'),
        DeclareLaunchArgument('tolerance', default_value='0.05'),
        DeclareLaunchArgument('hz', default_value='10.0'),
        DeclareLaunchArgument('kp', default_value='12.4'),
        DeclareLaunchArgument('ki', default_value='0.01'),
        DeclareLaunchArgument('kd', default_value='0.0005'),
        # DeclareLaunchArgument('i_limit', default_value='0.5'),
        DeclareLaunchArgument('max_speed', default_value='0.15'),
        DeclareLaunchArgument('min_speed', default_value='0.05'),
        DeclareLaunchArgument('sector_deg', default_value='12.0'),
        DeclareLaunchArgument('use_median', default_value='True'),
        DeclareLaunchArgument('reverse_ok', default_value='True'),

        # start the pid_controller node in a namespace
        Node(
            package='task_3',
            # namespace='basics',
            executable='pid_speed_controller',
            name='pid_speed_controller',
            parameters=[{
                'setpoint_reference': LaunchConfiguration('setpoint_reference'),
                'tolerance': LaunchConfiguration('tolerance'),
                'hz': LaunchConfiguration('hz'),
                'kp': LaunchConfiguration('kp'),
                'ki': LaunchConfiguration('ki'),
                'kd': LaunchConfiguration('kd'),
                # 'i_limit': LaunchConfiguration('i_limit'),
                'max_speed': LaunchConfiguration('max_speed'),
                'min_speed': LaunchConfiguration('min_speed'),
                'sector_deg': LaunchConfiguration('sector_deg'),
                'use_median': LaunchConfiguration('use_median'),
                'reverse_ok': LaunchConfiguration('reverse_ok'),
            }]
        ),
    ])