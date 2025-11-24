#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Import other python packages that you think necessary


class Task2(Node):
    """
    Environment localization and navigation task.
    """
    def __init__(self):
        super().__init__('task2_node')
        self.timer = self.create_timer(0.1, self.timer_cb)
        # Fill in the initialization member variables that you need

    def timer_cb(self):
        self.get_logger().info('Task2 node is alive.', throttle_duration_sec=1)
        # Feel free to delete this line, and write your algorithm in this callback function

    # Define function(s) that complete the (automatic) mapping task


def main(args=None):
    rclpy.init(args=args)

    task2 = Task2()

    try:
        rclpy.spin(task2)
    except KeyboardInterrupt:
        pass
    finally:
        task2.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()