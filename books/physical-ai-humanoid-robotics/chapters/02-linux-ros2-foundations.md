---
title: "Chapter 2: Linux + ROS2 Foundations"
description: "Setting up the Linux and ROS2 environment for robotics development"
---

# Chapter 2: Linux + ROS2 Foundations

## Overview

This chapter introduces the foundational tools and operating system environment required for robotics development. Linux provides the robust, real-time capable platform needed for robot control, while ROS2 (Robot Operating System 2) provides the middleware and tools for building distributed robot applications.

## Why Linux for Robotics?

Linux has become the de facto standard for robotics development due to several key advantages:

### Real-time Capabilities
- Preemptive multitasking for time-critical operations
- Low-latency kernel configurations available
- Deterministic behavior for control systems

### Open Source Ecosystem
- Extensive libraries and tools available
- Large community of developers and researchers
- Transparent and customizable system behavior

### Hardware Support
- Support for diverse hardware platforms (x86, ARM, etc.)
- Real-time kernel patches available
- Comprehensive device driver support

### Development Tools
- Powerful command-line tools for system management
- Integrated development environments
- Debugging and profiling tools

## Linux Distributions for Robotics

### Ubuntu LTS (Recommended)
Ubuntu Long-Term Support (LTS) versions are the most commonly used distribution for robotics development:

- **Stability**: 5-year support cycle ensures long-term stability
- **Community**: Extensive documentation and community support
- **Package Management**: APT package manager with extensive robotics libraries
- **ROS Compatibility**: Official ROS2 distributions target Ubuntu LTS

### Other Options
- **Real-time Linux**: For applications requiring deterministic timing
- **Yocto**: For custom embedded systems
- **Debian**: For stability-focused applications

## Introduction to ROS2

ROS2 is the next generation of the Robot Operating System, designed to address the limitations of ROS1 and provide a production-ready platform for robotics applications.

### Key Improvements in ROS2

1. **Real-time Support**: Better support for real-time systems
2. **Multi-robot Systems**: Native support for multiple robots
3. **Security**: Built-in security features and authentication
4. **Quality of Service**: Configurable communication policies
5. **Cross-platform**: Support for Windows, macOS, and Linux

### ROS2 Architecture

ROS2 uses a distributed system architecture based on the Data Distribution Service (DDS) standard:

#### Nodes
- Independent processes that perform computation
- Communicate with other nodes through topics, services, and actions
- Can run on the same or different machines

#### Topics
- Publish/subscribe communication pattern
- Asynchronous, one-to-many communication
- Used for sensor data, robot state, etc.

#### Services
- Request/response communication pattern
- Synchronous, one-to-one communication
- Used for actions that require confirmation

#### Actions
- Goal-based communication pattern
- Long-running tasks with feedback
- Used for navigation, manipulation, etc.

## Installing ROS2

### System Requirements
- Ubuntu 20.04 (Focal) or Ubuntu 22.04 (Jammy)
- 64-bit processor
- At least 4GB RAM (8GB+ recommended)
- 20GB free disk space

### Installation Steps

1. **Set locale**
   ```bash
   locale  # check for UTF-8
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US.UTF-8
   ```

2. **Add ROS2 GPG key and repository**
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. **Install ROS2**
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   ```

4. **Install colcon build tools**
   ```bash
   sudo apt install python3-colcon-common-extensions
   ```

5. **Setup environment**
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

## ROS2 Concepts and Tools

### Workspace Organization

ROS2 uses a workspace-based organization:

```
workspace/
├── src/           # Source code
├── build/         # Build artifacts
├── install/       # Install targets
└── log/           # Log files
```

### Creating a Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Essential ROS2 Commands

#### Node Management
- `ros2 run <package_name> <executable_name>`: Run a node
- `ros2 node list`: List running nodes
- `ros2 node info <node_name>`: Get information about a node

#### Topic Communication
- `ros2 topic list`: List available topics
- `ros2 topic echo <topic_name>`: Print messages from a topic
- `ros2 topic pub <topic_name> <msg_type> <args>`: Publish a message

#### Service Communication
- `ros2 service list`: List available services
- `ros2 service call <service_name> <service_type> <args>`: Call a service

#### Parameter Management
- `ros2 param list`: List parameters of a node
- `ros2 param get <node_name> <param_name>`: Get parameter value
- `ros2 param set <node_name> <param_name> <value>`: Set parameter value

## ROS2 Packages and Messages

### Package Structure

A typical ROS2 package includes:

```
package_name/
├── CMakeLists.txt     # Build configuration for C++
├── package.xml        # Package metadata
├── src/               # Source code
├── include/           # Header files
├── launch/            # Launch files
├── config/            # Configuration files
├── test/              # Test files
└── msg/               # Custom message definitions
```

### Creating a Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake my_robot_package
```

### Common Message Types

- `std_msgs`: Basic data types (Int32, Float64, String, etc.)
- `sensor_msgs`: Sensor data (Image, LaserScan, JointState, etc.)
- `geometry_msgs`: Geometric primitives (Point, Pose, Twist, etc.)
- `nav_msgs`: Navigation-related messages
- `action_msgs`: Action-related messages

## ROS2 Launch Files

Launch files allow you to start multiple nodes with a single command:

```xml
<launch>
  <node pkg="my_robot_package" exec="robot_controller" name="controller" output="screen">
    <param name="wheel_radius" value="0.1"/>
    <param name="base_width" value="0.5"/>
  </node>

  <node pkg="my_robot_package" exec="sensor_node" name="lidar" output="screen"/>
</launch>
```

## ROS2 Actions and Services

### Actions
Actions are used for long-running tasks with feedback:

```python
# Example action client
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

class NavigationClient:
    def __init__(self):
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
```

### Services
Services provide request-response communication:

```python
# Example service server
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response
```

## Best Practices for ROS2 Development

### Code Organization
- Use separate packages for different functionality
- Follow the ROS2 coding standards
- Use meaningful names for nodes, topics, and services

### Error Handling
- Implement proper error handling and logging
- Use ROS2's logging facilities
- Handle network and communication failures gracefully

### Performance Considerations
- Minimize message size and frequency
- Use appropriate Quality of Service (QoS) settings
- Consider real-time constraints in your design

### Security
- Use ROS2's security features when appropriate
- Validate all inputs from other nodes
- Consider network security for distributed systems

## Setting Up Your Development Environment

### IDE Recommendations
- **VS Code**: With ROS2 extensions
- **CLion**: For C++ development
- **PyCharm**: For Python development
- **Eclipse**: Traditional ROS IDE

### Version Control
- Use Git for version control
- Structure repositories to align with ROS2 packages
- Use appropriate .gitignore files for ROS2 projects

### Simulation Integration
- Integrate with Gazebo for physics simulation
- Use RViz for visualization
- Implement testing with GTest/gtest for C++ or pytest for Python

## Conclusion

Linux and ROS2 form the foundation of modern robotics development. Understanding these tools is essential for building sophisticated humanoid robots. ROS2's distributed architecture, real-time capabilities, and extensive ecosystem make it the ideal platform for Physical AI applications.

In the next chapter, we will explore simulation environments that allow you to test and develop your robotics algorithms in a safe, controlled environment before deploying them on real hardware.

## Exercises

1. Install ROS2 Humble Hawksbill on your Linux system and verify the installation by running the talker/listener example.

2. Create a new ROS2 workspace and build a simple "Hello World" package that publishes a message to a topic.

3. Research the differences between ROS1 and ROS2. What are the key architectural changes and why were they made?

4. Explain the publish/subscribe communication pattern in ROS2. When would you use topics vs. services vs. actions?

5. Set up a basic launch file that starts two nodes and configures parameters for each.