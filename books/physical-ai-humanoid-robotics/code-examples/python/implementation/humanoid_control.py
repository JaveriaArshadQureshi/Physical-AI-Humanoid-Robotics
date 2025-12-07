#!/usr/bin/env python3
"""
Humanoid Robot Control Implementation
Chapter: Building a Humanoid (Actuators, Joints, Hardware Choices)
"""

import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class JointState:
    """Represents the state of a single joint"""
    position: float
    velocity: float
    effort: float
    desired_position: float = 0.0


class HumanoidController:
    """
    A controller for humanoid robot with multiple joints
    """
    def __init__(self, joint_names: List[str]):
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.joint_states = [JointState(0.0, 0.0, 0.0) for _ in range(self.num_joints)]
        self.control_loop_rate = 100  # Hz
        self.is_running = False
        self.control_thread = None

        # PID gains for each joint (simplified)
        self.kp = [100.0] * self.num_joints  # Proportional gain
        self.ki = [0.1] * self.num_joints   # Integral gain
        self.kd = [10.0] * self.num_joints  # Derivative gain

        self.integral_errors = [0.0] * self.num_joints
        self.previous_errors = [0.0] * self.num_joints

    def set_joint_positions(self, positions: List[float]):
        """Set desired joint positions"""
        if len(positions) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} positions, got {len(positions)}")

        for i, pos in enumerate(positions):
            self.joint_states[i].desired_position = pos

    def get_joint_positions(self) -> List[float]:
        """Get current joint positions"""
        return [state.position for state in self.joint_states]

    def compute_control_efforts(self) -> List[float]:
        """Compute control efforts using PID control"""
        efforts = []

        for i in range(self.num_joints):
            current_pos = self.joint_states[i].position
            desired_pos = self.joint_states[i].desired_position

            # Calculate error
            error = desired_pos - current_pos

            # Update integral and derivative terms
            self.integral_errors[i] += error / self.control_loop_rate
            derivative = (error - self.previous_errors[i]) * self.control_loop_rate

            # Apply PID formula
            effort = (self.kp[i] * error +
                     self.ki[i] * self.integral_errors[i] +
                     self.kd[i] * derivative)

            # Store current error for next derivative calculation
            self.previous_errors[i] = error

            efforts.append(effort)

        return efforts

    def control_loop(self):
        """Main control loop running in separate thread"""
        rate = 1.0 / self.control_loop_rate

        while self.is_running:
            start_time = time.time()

            # Compute control efforts
            efforts = self.compute_control_efforts()

            # Simulate applying efforts to joints (in a real robot, this would interface with hardware)
            for i in range(self.num_joints):
                # Simple simulation: update position based on effort
                # In a real implementation, this would interface with the robot's actuators
                self.joint_states[i].effort = efforts[i]

                # Update position (simplified simulation)
                dt = rate
                self.joint_states[i].velocity += efforts[i] * dt  # F = ma, simplified
                self.joint_states[i].position += self.joint_states[i].velocity * dt

            # Sleep to maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, rate - elapsed)
            time.sleep(sleep_time)

    def start(self):
        """Start the control loop in a separate thread"""
        if not self.is_running:
            self.is_running = True
            self.control_thread = threading.Thread(target=self.control_loop)
            self.control_thread.start()

    def stop(self):
        """Stop the control loop"""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()


def walk_trajectory_generator(duration: float = 10.0, dt: float = 0.01) -> List[List[float]]:
    """
    Generate a simple walking trajectory for demonstration
    This is a simplified example - real walking trajectories are much more complex
    """
    # For demonstration, we'll create a trajectory for a 6-DOF leg
    # In a real humanoid, we'd have many more joints
    num_joints = 6  # Simplified leg with 6 DOF
    steps = int(duration / dt)

    trajectory = []

    for t in np.linspace(0, duration, steps):
        # Generate a simple periodic trajectory
        # This is a very simplified walking pattern
        joint_positions = [
            0.0,                                    # Hip yaw
            0.2 * np.sin(2 * np.pi * t / 2.0),     # Hip pitch (walking motion)
            0.1 * np.sin(2 * np.pi * t / 2.0),     # Hip roll
            0.3 * np.sin(2 * np.pi * t / 2.0),     # Knee pitch
            0.1 * np.cos(2 * np.pi * t / 2.0),     # Ankle pitch
            0.05 * np.sin(2 * np.pi * t / 2.0)     # Ankle roll
        ]
        trajectory.append(joint_positions)

    return trajectory


def main():
    """
    Main function demonstrating humanoid control
    """
    # Create a controller for a simplified humanoid (6 joints for one leg)
    joint_names = [
        "left_hip_yaw", "left_hip_pitch", "left_hip_roll",
        "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll"
    ]

    controller = HumanoidController(joint_names)

    print("Starting humanoid controller...")
    controller.start()

    try:
        # Generate a simple walking trajectory
        trajectory = walk_trajectory_generator(duration=5.0)

        print(f"Generated trajectory with {len(trajectory)} steps")

        # Execute the trajectory
        for i, joint_positions in enumerate(trajectory):
            controller.set_joint_positions(joint_positions)

            if i % 50 == 0:  # Print every 50 steps
                current_positions = controller.get_joint_positions()
                print(f"Step {i}: Desired={joint_positions[1]:.3f}, Actual={current_positions[1]:.3f}")

            time.sleep(0.01)  # 10ms delay between trajectory points

        print("Trajectory execution completed")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        controller.stop()
        print("Controller stopped")


if __name__ == '__main__':
    main()