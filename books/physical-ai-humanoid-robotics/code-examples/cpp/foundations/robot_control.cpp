#include <iostream>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

class RobotController : public rclcpp::Node
{
public:
    RobotController() : Node("robot_controller")
    {
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&RobotController::scan_callback, this, std::placeholders::_1)
        );

        control_timer_ = this->create_wall_timer(
            100ms, std::bind(&RobotController::control_loop, this)
        );

        safe_distance_ = 1.0;
        moving_forward_ = true;
    }

private:
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        // Check the front-facing range (middle of the ranges array)
        size_t front_index = msg->ranges.size() / 2;
        float front_scan = msg->ranges[front_index];

        if (front_scan < safe_distance_) {
            moving_forward_ = false;
        } else {
            moving_forward_ = true;
        }
    }

    void control_loop()
    {
        auto cmd_vel_msg = geometry_msgs::msg::Twist();

        if (moving_forward_) {
            cmd_vel_msg.linear.x = 0.5;  // Move forward at 0.5 m/s
            cmd_vel_msg.angular.z = 0.0;
        } else {
            cmd_vel_msg.linear.x = 0.0;  // Stop
            cmd_vel_msg.angular.z = 0.3; // Turn slowly
        }

        cmd_vel_publisher_->publish(cmd_vel_msg);
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    double safe_distance_;
    bool moving_forward_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotController>());
    rclcpp::shutdown();
    return 0;
}