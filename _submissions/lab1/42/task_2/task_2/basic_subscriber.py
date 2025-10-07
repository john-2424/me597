import rclpy
from rclpy.node import Node

# from std_msgs.msg import String
from task_2_interfaces.msg import JointData


class BasicSubscriber(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('listener')

        # Subscribe to a topic
        # self.subscription = self.create_subscription(
        #     String,
        #     'my_first_topic',
        #     self.listener_callback,
        #     20
        # )
        # Subscribe to a topic with custom message type
        self.subscription = self.create_subscription(
            JointData,
            'joint_topic',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Logs the received messages from a topic
        # self.get_logger().info('[Receiving]: ' + ' ;; '.join(2*[msg.data]))
        # Logs the received message from a topic of custom message type
        self.get_logger().info(f'[Receiving]: Center: {msg.center}; Vel: {msg.vel}')


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
