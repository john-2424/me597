# import time

import rclpy
from rclpy.node import Node

# from std_msgs.msg import String
from task_2_interfaces.msg import JointData


class BasicPublisher(Node):
    def __init__(self):
        # Initialize with 'Node' constructor
        super().__init__('basic_publisher')
        
        # Initialize node start time
        # self.node_started_at = time.time()

        # Topic publisher, with 'String' message type, 20 queue size
        # self.publisher_ = self.create_publisher(String, 'my_first_topic', 20)
        
        # Topic publisher, with custom 'JointData' message type, 10 queue size
        self.publisher_ = self.create_publisher(JointData, 'joint_topic', 10)
        
        # Topic timer to publish messages in an interval
        # timer_period = 0.05  # in seconds, for 20Hz
        timer_period = 0.5  # in seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Topic publish counter
        self.i = 0

    def timer_callback(self):
        # Create 'String' message object assigned with node's active/run time 
        # msg = String()
        # msg.data = f'Node Active for {int(time.time()-self.node_started_at)}s'  # Current time - Node start time (in seconds)
        
        # Create 'JointData' message object assigned with center and vel data
        msg = JointData()
        msg.center.x = float(self.i)
        msg.center.y = 0.0
        msg.center.z = 0.0
        msg.vel = float(self.i * 2)

        # Publish message to the topic
        self.publisher_.publish(msg)
        
        # Node logging to show message data published to the topic
        self.get_logger().info(f'[Publishing]: "Center: {msg.center}; Vel: {msg.vel}"')
        
        # Topic publish counter increment
        self.i += 1


def main(args=None):
    # Initialize rclpy library
    rclpy.init(args=args)

    # Create instance of the 'BasicPublisher' class node
    basic_publisher = BasicPublisher()

    # Spin up the created node
    rclpy.spin(basic_publisher)

    # Explicit declaration to destroy the node object and shutdown rclpy
    basic_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
