import rclpy
from rclpy.node import Node

from task_2_interfaces.srv import JointState


class BasicService(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('service')

        # Create a service
        self.srv = self.create_service(JointState, 'joint_service', self.check_valid_callback)

    def check_valid_callback(self, request, response):
        # Log incoming request
        self.get_logger().info(f'[Incoming Request] :: x: {request.x}; y: {request.y}; z: {request.z};')
        
        # Logic for response
        sum = request.x + request.y + request.z
        if sum >= 0:
            response.valid = True
        else:
            response.valid = False
        
        # Log outgoing response
        self.get_logger().info(f'[Outgoing Response] :: Valid: {response.valid}')
        return response


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'BasicService' class node
    basic_service = BasicService()

    # Spin up the created node
    rclpy.spin(basic_service)

    # Explicit declaration to destroy the node object and shutdown rclpy
    basic_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
