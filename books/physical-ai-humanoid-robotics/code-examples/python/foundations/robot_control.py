#!/usr/bin/env python3
"""
Basic Robot Control Example
Chapter: Linux + ROS2 Foundations
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class RobotController(Node):
    """
    Basic robot controller that moves the robot forward until an obstacle is detected
    """
    def __init__(self):
        super().__init__('robot_controller')
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.safe_distance = 1.0  # meters
        self.moving_forward = True

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def scan_callback(self, msg):
        # Check the front-facing range (index around the middle of the array)
        front_scan = msg.ranges[len(msg.ranges) // 2]

        if front_scan < self.safe_distance:
            self.moving_forward = False
        else:
            self.moving_forward = True

    def control_loop(self):
        cmd_vel_msg = Twist()

        if self.moving_forward:
            cmd_vel_msg.linear.x = 0.5  # Move forward at 0.5 m/s
            cmd_vel_msg.angular.z = 0.0
        else:
            cmd_vel_msg.linear.x = 0.0  # Stop
            cmd_vel_msg.angular.z = 0.3  # Turn slowly

        self.cmd_vel_publisher.publish(cmd_vel_msg)


def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()