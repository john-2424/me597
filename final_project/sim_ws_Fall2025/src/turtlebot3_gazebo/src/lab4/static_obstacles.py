#!/usr/bin/env python3

import os
import rclpy

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SpawnEntity, SetEntityState
from geometry_msgs.msg import Pose
from rclpy.node import Node


class GazeboModelHandler(Node):

    def __init__(self):
        super().__init__('gazebo_model_handler')
        self.spawn_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.set_state_client = self.create_client(SetEntityState, 'set_entity_state')

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Spawn service not available, waiting again...')

        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set state service not available, waiting again...')
        
        self.spawn_model('obstacle0','trash_can')
        self.spawn_model('obstacle1','trash_can')

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.elapsed = 0

    def spawn_model(self, model_name,model):
        
        initial_pose = Pose()
        initial_pose.position.x = 0.0
        initial_pose.position.y = 0.0
        initial_pose.position.z = 1.0

        request = SpawnEntity.Request()
        request.name = model_name
        request.xml = self.load_model_xml_from_sdf(model)
        request.reference_frame = 'world'
        request.initial_pose = initial_pose

        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f'Model {model_name} spawned successfully')
        else:
            self.get_logger().error(f'Failed to spawn model {model_name}')

    def load_model_xml_from_sdf(self, model_name):
        xml_model = ""
        try:
            package_path = get_package_share_directory('turtlebot3_gazebo')
            model_path = os.path.join(package_path, 'models', model_name, 'model.sdf')
            
            with open(model_path, 'r') as xml_file:
                xml_model = xml_file.read().replace('\n', '')
        except Exception as e:
            self.get_logger().error(e)
        return xml_model
    
    def timer_callback(self):
        t = self.elapsed
        y0 = 0.5
        y1 = -0.7

 
        self.set_model_position(-3.75, y0, 'obstacle0')
        self.set_model_position(6.0, y1, 'obstacle1')

        self.elapsed = (self.elapsed + 1) % 201  # Increment and reset after reaching 200

    def set_model_position(self, x, y, model_name):
        """ Calls the set_state service to set the model_name position to x, y coordinates """
        state = EntityState()
        state.name = model_name
        state.pose.position.x = x
        state.pose.position.y = y
        
        request = SetEntityState.Request()
        request.state = state
        self.set_state_client.call_async(request)


def main(args=None):
    rclpy.init(args=args)

    gazeboModelHandlerNode = GazeboModelHandler()

    try:
        rclpy.spin(gazeboModelHandlerNode)
    except:
        pass
    finally:
        gazeboModelHandlerNode.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()  
