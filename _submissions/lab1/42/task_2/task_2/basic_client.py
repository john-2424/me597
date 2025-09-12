import sys

import rclpy
from rclpy.node import Node

from task_2_interfaces.srv import JointState


class BasicClientAsync(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('client')

        # Create client
        self.cli = self.create_client(JointState, 'joint_service')
        # Check if the defined service is available with a 1 second interval
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service is not available, waiting again...')
        
        # Initialize request object
        self.req = JointState.Request()

    def send_request(self, x, y, z):
        # Send the request and spin until it receives the response or fails
        self.req.x = x
        self.req.y = y
        self.req.z = z
        return self.cli.call_async(self.req)


def main():
    # Initialize rclpy library
    rclpy.init()

    # Create instance of the 'BasicClientAsync' class node
    basic_client_async = BasicClientAsync()
    future = basic_client_async.send_request(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    
    # Wait for the future result of the request
    rclpy.spin_until_future_complete(basic_client_async, future)
    response = future.result()

    # Log the result
    basic_client_async.get_logger().info(f'[Response] :: Valid: {response.valid}')

    # Explicit declaration to destroy the node object and shutdown rclpy
    basic_client_async.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
