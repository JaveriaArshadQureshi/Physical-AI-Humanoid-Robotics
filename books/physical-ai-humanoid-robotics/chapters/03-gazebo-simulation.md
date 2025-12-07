---
title: "Chapter 3: Gazebo / Ignition Simulation"
description: "Physics-based simulation for robotics development and testing"
---

# Chapter 3: Gazebo / Ignition Simulation

## Overview

Simulation is a critical component of robotics development, allowing for safe, cost-effective testing and validation of algorithms before deployment on real hardware. Gazebo and its successor Ignition provide physics-based simulation environments that accurately model the dynamics of robotic systems in realistic environments.

## The Role of Simulation in Robotics

Simulation serves multiple purposes in the robotics development lifecycle:

### Development and Testing
- Test algorithms without risk of hardware damage
- Rapid iteration on control and perception algorithms
- Validation of system behavior under various conditions

### Training and Learning
- Generate synthetic data for machine learning
- Train robots in diverse environments
- Test edge cases that are difficult to reproduce in reality

### Integration and Debugging
- Test multi-robot systems without physical hardware
- Debug software components in isolation
- Validate sensor models and data processing pipelines

## Introduction to Gazebo

Gazebo is a physics-based simulation environment that provides:
- Realistic physics simulation using ODE, Bullet, or Simbody
- High-quality rendering with OGRE
- Support for various sensors (cameras, LIDAR, IMU, etc.)
- Plugin architecture for custom functionality
- Integration with ROS/ROS2

### Gazebo vs. Ignition

Ignition Gazebo (now called Ignition Fortress) is the next-generation simulation platform that offers:
- Improved performance and modularity
- Better real-time simulation capabilities
- Enhanced rendering and visualization
- More flexible plugin architecture
- Better integration with ROS2

## Installing Gazebo

### Gazebo Classic Installation

For ROS2 Humble, install Gazebo Classic:

```bash
sudo apt update
sudo apt install ros-humble-gazebo-*
sudo apt install gazebo11 gz-transport12
```

### Ignition Fortress Installation

For the latest Ignition:

```bash
curl -sSL http://get.gazebosim.org | sh
sudo apt install ignition-fortress
```

## Gazebo Architecture

### World Files
World files define the simulation environment in SDF (Simulation Description Format):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Your robot model -->
    <include>
      <uri>model://my_robot</uri>
    </include>

    <!-- Custom objects -->
    <model name="obstacle">
      <pose>1 0 0 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Robot Models
Robot models are defined using URDF (Unified Robot Description Format) or SDF:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Wheel joints and links -->
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Gazebo Plugins

Gazebo supports various plugins to extend functionality:

### Sensor Plugins
- Camera sensors
- LIDAR/Laser range finders
- IMU sensors
- Force/torque sensors
- GPS sensors

### Controller Plugins
- Joint controllers
- Model state publishers
- ROS2 interface plugins

### Custom Plugins
- Model plugins for custom behavior
- World plugins for environment modifications
- Sensor plugins for custom sensors

## ROS2 Integration with Gazebo

### Gazebo ROS2 Packages
- `gazebo_ros_pkgs`: Core ROS2-Gazebo integration
- `gazebo_plugins`: Pre-built simulation plugins
- `gazebo_dev`: Development tools and headers

### Launch Files for Simulation

Example launch file to start Gazebo with a robot:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ])
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'my_robot'],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_entity
    ])
```

## Physics Simulation

### Physics Engines
Gazebo supports multiple physics engines:
- **ODE**: Open Dynamics Engine (default)
- **Bullet**: Fast and robust
- **Simbody**: Accurate for complex systems

### Physics Parameters
Tuning physics parameters affects simulation accuracy and performance:
- Time step size
- Solver iterations
- Collision detection parameters

### Realism vs. Performance Trade-offs
- Smaller time steps: More accurate but slower
- More solver iterations: More stable but slower
- Complex collision meshes: More accurate but slower

## Sensor Simulation

### Camera Simulation
```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_frame</frame_name>
  </plugin>
</sensor>
```

### LIDAR Simulation
```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
    <frame_name>lidar_frame</frame_name>
  </plugin>
</sensor>
```

## Creating Custom Environments

### Building World Files
World files can include:
- Static objects and obstacles
- Dynamic objects
- Light sources
- Terrain models
- Weather effects

### Using Models from Fuel
Gazebo Fuel is an online repository of 3D models:
- Buildings and structures
- Vehicles
- Furniture
- Robots
- Natural environments

## Advanced Simulation Features

### Multi-Robot Simulation
Gazebo can simulate multiple robots simultaneously:
- Coordinate their movements
- Simulate communication between robots
- Test multi-robot algorithms

### Physics-based Interactions
- Object manipulation and grasping
- Collision response and contact simulation
- Deformable objects (in advanced versions)

### Real-time Simulation
- Synchronization with real-time clock
- Deterministic simulation for reproducible results
- Performance optimization techniques

## Performance Optimization

### Simulation Speed
- Adjust physics parameters for speed vs. accuracy
- Simplify collision meshes
- Reduce sensor update rates where appropriate

### Rendering Optimization
- Use simplified visual models
- Adjust rendering quality settings
- Disable rendering for headless simulation

### Model Optimization
- Use simplified meshes for collision detection
- Reduce the number of joints and links where possible
- Optimize URDF/SDF files for faster loading

## Troubleshooting Common Issues

### Physics Instability
- Increase solver iterations
- Reduce time step size
- Check mass and inertia parameters

### Performance Problems
- Simplify collision geometries
- Reduce sensor resolution
- Use faster physics engine

### Plugin Issues
- Check plugin dependencies
- Verify correct plugin paths
- Check ROS2 topic/service names

## Best Practices

### Model Design
- Create separate visual and collision geometries
- Use appropriate mass and inertia values
- Include proper joint limits and dynamics

### Simulation Design
- Start with simple environments
- Gradually increase complexity
- Validate simulation results against real-world data

### Integration with ROS2
- Use standard message types when possible
- Follow ROS2 naming conventions
- Implement proper error handling

## Conclusion

Gazebo and Ignition provide powerful simulation environments for robotics development, enabling safe and efficient testing of Physical AI algorithms. Understanding simulation is crucial for developing robust humanoid robots that can operate effectively in the real world.

Simulation allows you to test edge cases, validate control algorithms, and train machine learning models without the risks and costs associated with real hardware. The next chapter will explore NVIDIA Isaac Sim, which provides even more advanced simulation capabilities for humanoid robotics.

## Exercises

1. Install Gazebo and run a simple simulation with a basic robot model.

2. Create a custom world file with obstacles and simulate a robot navigating through it.

3. Add a camera sensor to your robot model and visualize the camera feed in RViz.

4. Research the differences between Gazebo Classic and Ignition Gazebo. What are the advantages of each?

5. Implement a simple controller plugin that moves a robot based on sensor input in simulation.