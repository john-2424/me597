import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64


class BasicSubscriber(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('listener')

        # Subscribe to a topic
        self.subscription = self.create_subscription(
            Float64,
            'my_first_topic',
            self.listener_callback,
            20
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Logs the received messages from a topic
        self.get_logger().info(f'[Receiving]: {msg.data}; [Doubled]: {2*msg.data}')


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'BasicSubscriber' class node
    basic_subscriber = BasicSubscriber()

    # Spin up the created node
    rclpy.spin(basic_subscriber)

    # Explicit declaration to destroy the node object and shutdown rclpy
    basic_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
