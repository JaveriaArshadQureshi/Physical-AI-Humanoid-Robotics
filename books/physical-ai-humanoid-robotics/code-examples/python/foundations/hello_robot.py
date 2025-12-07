#!/usr/bin/env python3
"""
Hello Robot - Basic ROS2 Node Example
Chapter: Introduction to Physical AI & Humanoid Robotics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloRobotNode(Node):
    """
    A simple ROS2 node that publishes a "Hello Robot" message
    """
    def __init__(self):
        super().__init__('hello_robot_node')
        self.publisher = self.create_publisher(String, 'robot_greeting', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello Robot! Message #{self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    hello_robot_node = HelloRobotNode()

    try:
        rclpy.spin(hello_robot_node)
    except KeyboardInterrupt:
        pass
    finally:
        hello_robot_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()