# Exercise Solutions: Introduction to Physical AI & Humanoid Robotics

## Chapter 1: Introduction to Physical AI & Humanoid Robotics

### Exercise 1.1: Physical AI vs Traditional AI
**Question**: What are the key differences between Physical AI and traditional AI?

**Solution**:
- Traditional AI operates primarily in digital domains (e.g., playing chess, processing text)
- Physical AI systems have a physical form that interacts with the real world
- Physical AI must handle uncertainty, sensor noise, and environmental unpredictability
- Physical AI requires real-time processing within the constraints of physical dynamics

### Exercise 1.2: Humanoid Robot Applications
**Question**: List 5 potential applications for humanoid robots.

**Solution**:
1. Elderly care and assistance
2. Education and therapy
3. Industrial collaboration with humans
4. Disaster response and rescue
5. Customer service and hospitality

## Chapter 2: Linux + ROS2 Foundations

### Exercise 2.1: ROS2 Node Communication
**Question**: Explain how nodes communicate in ROS2.

**Solution**:
In ROS2, nodes communicate through:
- Topics: Publish/subscribe messaging pattern
- Services: Request/response pattern
- Actions: Goal/feedback/result pattern with cancel/reject capabilities
Communication is facilitated by DDS (Data Distribution Service) middleware.

### Exercise 2.2: Package Creation
**Question**: Create a simple ROS2 package structure.

**Solution**:
```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── src/
│   └── my_node.cpp
├── include/
│   └── my_robot_package/
│       └── my_header.hpp
├── launch/
│   └── my_launch_file.py
└── config/
    └── params.yaml
```

## Chapter 3: Gazebo / Ignition Simulation

### Exercise 3.1: SDF Model Creation
**Question**: Create a simple SDF model for a wheeled robot.

**Solution**:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <link name="chassis">
      <pose>0 0 0.1 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box><size>0.5 0.3 0.2</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.5 0.3 0.2</size></box>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

## Chapter 4: NVIDIA Isaac Sim

### Exercise 4.1: Isaac Sim vs Gazebo
**Question**: Compare NVIDIA Isaac Sim and Gazebo for robotics simulation.

**Solution**:
- Isaac Sim: Better graphics, more realistic physics, NVIDIA ecosystem integration
- Gazebo: Open source, more mature, broader robot model support
- Isaac Sim: Better for vision-based tasks, Gazebo: Better for general robotics research

## Chapter 5: Real Robot Control Architecture

### Exercise 5.1: Control Architecture Design
**Question**: Design a control architecture for a 6-DOF robotic arm.

**Solution**:
The architecture should include:
- High-level motion planner
- Trajectory generator
- Joint-level controllers (PID)
- Hardware abstraction layer
- Safety monitors

## Chapter 6: Sensor Fusion + Localization (SLAM/IMU/LiDAR)

### Exercise 6.1: Kalman Filter Implementation
**Question**: Implement a simple Kalman filter for sensor fusion.

**Solution**:
```python
import numpy as np

class SimpleKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_input):
        # State prediction (simplified)
        self.state = self.state + control_input  # Simplified model
        # Covariance prediction
        self.covariance = self.covariance + self.process_noise

    def update(self, measurement):
        # Kalman gain
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)
        # State update
        self.state = self.state + kalman_gain * (measurement - self.state)
        # Covariance update
        self.covariance = (1 - kalman_gain) * self.covariance
```

## Chapter 7: Kinematics & Dynamics (FK, IK, Trajectory Planning)

### Exercise 7.1: Forward Kinematics
**Question**: Calculate the end-effector position for a 2-DOF planar arm with link lengths [1, 1] and joint angles [π/4, π/4].

**Solution**:
For a 2-DOF planar arm:
- Joint 1: x₁ = L₁ * cos(θ₁), y₁ = L₁ * sin(θ₁)
- End-effector: x₂ = x₁ + L₂ * cos(θ₁ + θ₂), y₂ = y₁ + L₂ * sin(θ₁ + θ₂)

With L₁=1, L₂=1, θ₁=π/4, θ₂=π/4:
- x₂ = cos(π/4) + cos(π/2) = √2/2 + 0 = 0.707
- y₂ = sin(π/4) + sin(π/2) = √2/2 + 1 = 1.707

### Exercise 7.2: Inverse Kinematics
**Question**: For a 2-DOF arm with link lengths [1, 1], find joint angles for end-effector position [1.5, 0.5].

**Solution**:
Using the law of cosines:
- r² = x² + y² = 1.5² + 0.5² = 2.5
- cos(θ₂) = (L₁² + L₂² - r²) / (2*L₁*L₂) = (1 + 1 - 2.5) / 2 = -0.25
- θ₂ = arccos(-0.25) ≈ 1.823 radians
- θ₁ = arctan2(y, x) - arctan2(L₂*sin(θ₂), L₁ + L₂*cos(θ₂))

## Chapter 8: Control Systems (PID, MPC, Whole-Body Control)

### Exercise 8.1: PID Controller Design
**Question**: Design a PID controller for a simple system.

**Solution**:
```python
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative

        # Store error for next iteration
        self.previous_error = error

        return p_term + i_term + d_term
```

## Chapter 9: Robot Perception (CV, LLM-Vision, Depth Estimation)

### Exercise 9.1: Feature Detection
**Question**: Explain the difference between SIFT and ORB feature detectors.

**Solution**:
- SIFT: Scale-invariant, rotation-invariant, patented (requires license)
- ORB: Faster, free to use, binary descriptors, good for real-time applications
- SIFT: More robust to changes in illumination, ORB: More sensitive to illumination changes

## Chapter 10: Vision-Language-Action Models (VLAs)

### Exercise 10.1: VLA Architecture Components
**Question**: List the main components of a Vision-Language-Action model.

**Solution**:
1. Visual encoder (e.g., CNN or Vision Transformer)
2. Language encoder (e.g., Transformer-based model)
3. Fusion mechanism to combine visual and language features
4. Action decoder to generate motor commands
5. Training framework for end-to-end learning

## Chapter 11: Reinforcement Learning for Robotics

### Exercise 11.1: Q-Learning Algorithm
**Question**: Implement the Q-learning update rule.

**Solution**:
```python
def q_learning_update(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Q-learning update rule: Q(s,a) = Q(s,a) + α[r + γmax Q(s',a') - Q(s,a)]
    """
    current_q = q_table[state][action]
    max_next_q = max(q_table[next_state])

    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    q_table[state][action] = new_q

    return q_table
```

## Chapter 12: Imitation Learning + Teleoperation

### Exercise 12.1: Behavioral Cloning
**Question**: Explain the difference between behavioral cloning and DAgger.

**Solution**:
- Behavioral cloning: Simple supervised learning from expert demonstrations
- DAgger: Iteratively collects new data by querying the expert for states the policy visits
- DAgger: Addresses distribution shift problem, behavioral cloning: Does not

## Chapter 13: Building a Humanoid (Actuators, Joints, Hardware Choices)

### Exercise 13.1: Actuator Selection
**Question**: Compare servo motors, stepper motors, and pneumatic actuators for humanoid joints.

**Solution**:
- Servo motors: Precise control, moderate speed, moderate power density
- Stepper motors: Precise positioning, holding torque, but limited speed
- Pneumatic: High power density, fast response, but requires compressor and is harder to control

## Chapter 14: Autonomous Navigation for Humanoids

### Exercise 14.1: Path Planning Algorithms
**Question**: Compare A* and RRT for humanoid navigation.

**Solution**:
- A*: Optimal path in known environment, complete, but computationally expensive
- RRT: Probabilistically complete, good for high-dimensional spaces, but not optimal
- A*: Better for static environments, RRT: Better for dynamic environments

## Chapter 15: Safety, Fail-safes, Edge Computing

### Exercise 15.1: Safety Architecture
**Question**: Design a safety architecture for a humanoid robot.

**Solution**:
- Hardware safety: Emergency stops, current limiting, temperature monitoring
- Software safety: Collision detection, joint limit checking, behavior monitoring
- Communication safety: Redundant systems, fail-safe states
- Physical safety: Rounded edges, safe materials, impact absorption

## Chapter 16: Capstone Project Guide

### Exercise 16.1: Project Planning
**Question**: Create a project plan for implementing a humanoid robot that can walk and pick up objects.

**Solution**:
**Phase 1: Simulation Development**
- Create robot model in simulation
- Implement basic walking gait
- Test in simulation environment

**Phase 2: Perception System**
- Object detection and localization
- Hand-eye coordination
- Grasp planning

**Phase 3: Integration**
- Combine walking and manipulation
- Implement safety features
- Test in simulation

**Phase 4: Real Robot Implementation**
- Transfer to physical robot
- Fine-tune parameters
- Test and validate

This document provides solutions to exercises throughout the book, helping students verify their understanding of Physical AI and Humanoid Robotics concepts.