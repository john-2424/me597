import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import sys
import select
import termios
import tty
import threading
import signal

keybindings = {'w': (0.0, 0.1),
               's': (0.0, -0.1),
               'a': (-0.1, 0.0),
               'd': (0.1, 0.0)}

class PositionNode(Node):
    def __init__(self):
        super().__init__('model_service_node')

        # Create a client for the SetEntityState service
        self.client = self.create_client(SetEntityState, '/set_entity_state')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Initialize the entity state message
        self.state_msg = EntityState()
        self.state_msg.pose.position.x = -1.0
        self.state_msg.pose.position.y = 1.0
        self.state_msg.name = 'RED_BALL'

        # Set up a timer to call the function periodically (10Hz)
        self.timer = self.create_timer(0.03, self.timer_callback)

        self.key = None
        self.running = True

        # Save the terminal settings
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)

        # Start a separate thread to read input
        self.input_thread = threading.Thread(target=self.read_input)
        self.input_thread.start()

    def read_input(self):
        # Set terminal to cbreak mode
        tty.setcbreak(self.fd)
        try:
            while self.running:
                dr, dw, de = select.select([sys.stdin], [], [], 0.1)
                if dr:
                    key = sys.stdin.read(1)
                    self.key = key
        except Exception as e:
            self.get_logger().error(f"Input thread error: {e}")
        finally:
            # Restore the terminal settings when the thread ends
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def timer_callback(self):
        if self.key:
            key = self.key
            self.key = None  # Reset the key
            print(f"Key pressed: {key}")
            if key in keybindings:
                dx, dy = keybindings[key]
                self.state_msg.pose.position.x += dx
                self.state_msg.pose.position.y += dy
                print(f"Moved entity to x: {self.state_msg.pose.position.x}, y: {self.state_msg.pose.position.y}")

                # Create a request and set the entity state
                request = SetEntityState.Request()
                request.state = self.state_msg

                # Call the service asynchronously
                self.future = self.client.call_async(request)
                self.future.add_done_callback(self.service_callback)

    def service_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Entity state updated successfully')
            else:
                self.get_logger().error('Failed to update entity state')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def destroy_node(self):
        # Stop the input thread
        self.running = False
        # Wait for the input thread to finish
        self.input_thread.join()
        # Restore terminal settings
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PositionNode()

    def signal_handler(sig, frame):
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
