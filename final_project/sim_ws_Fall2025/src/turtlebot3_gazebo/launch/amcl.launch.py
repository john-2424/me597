import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the path to the params directory
    params_file = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'params',
        'amcl.yaml'
    )

    # Parameters to enable simulation time
    sim_time_param = {'use_sim_time': True}

    return LaunchDescription([
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params_file, sim_time_param]  # Include sim_time_param here
        )
    ])

if __name__ == '__main__':
    generate_launch_description()
