#include <iostream>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class HelloRobotNode : public rclcpp::Node
{
public:
    HelloRobotNode() : Node("hello_robot_node")
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("robot_greeting", 10);
        timer_ = this->create_wall_timer(
            1000ms, std::bind(&HelloRobotNode::timer_callback, this));
        count_ = 0;
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::String();
        message.data = "Hello Robot! Message #" + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HelloRobotNode>());
    rclcpp::shutdown();
    return 0;
}