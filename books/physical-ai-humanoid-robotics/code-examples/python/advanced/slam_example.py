#!/usr/bin/env python3
"""
SLAM (Simultaneous Localization and Mapping) Example
Chapter: Sensor Fusion + Localization (SLAM/IMU/LiDAR)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class SimpleSLAM:
    """
    A simplified SLAM implementation for educational purposes
    """
    def __init__(self):
        # Robot state [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])

        # Covariance matrix
        self.covariance = np.eye(3) * 0.1

        # Landmarks (known positions in the environment)
        self.landmarks = []

        # Measurements (relative to robot)
        self.measurements = []

    def predict(self, control_input, dt=0.1):
        """
        Predict the next state based on control input
        control_input = [v, omega] where v is linear velocity and omega is angular velocity
        """
        v, omega = control_input

        # Update state based on motion model
        if abs(omega) < 1e-5:  # Straight line motion
            self.state[0] += v * dt * np.cos(self.state[2])
            self.state[1] += v * dt * np.sin(self.state[2])
        else:  # Circular motion
            self.state[0] += (v / omega) * (np.sin(self.state[2] + omega * dt) - np.sin(self.state[2]))
            self.state[1] += (v / omega) * (np.cos(self.state[2]) - np.cos(self.state[2] + omega * dt))
            self.state[2] += omega * dt

        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

    def update(self, landmark_id, range_measurement, bearing_measurement):
        """
        Update state based on landmark measurement
        """
        # Expected measurement based on current state and landmark position
        if landmark_id < len(self.landmarks):
            landmark_x, landmark_y = self.landmarks[landmark_id]

            # Expected range and bearing
            dx = landmark_x - self.state[0]
            dy = landmark_y - self.state[1]
            expected_range = np.sqrt(dx**2 + dy**2)
            expected_bearing = np.arctan2(dy, dx) - self.state[2]
            expected_bearing = np.arctan2(np.sin(expected_bearing), np.cos(expected_bearing))

            # Measurement Jacobian
            H = np.zeros((2, 3))
            H[0, 0] = -dx / expected_range  # partial of range w.r.t. x
            H[0, 1] = -dy / expected_range  # partial of range w.r.t. y
            H[1, 0] = dy / (expected_range**2)  # partial of bearing w.r.t. x
            H[1, 1] = -dx / (expected_range**2)  # partial of bearing w.r.t. y
            H[1, 2] = -1  # partial of bearing w.r.t. theta

            # Measurement noise covariance
            R = np.diag([0.1**2, 0.05**2])  # range and bearing noise

            # Kalman gain
            S = H @ self.covariance @ H.T + R
            K = self.covariance @ H.T @ np.linalg.inv(S)

            # Innovation
            z = np.array([range_measurement, bearing_measurement])
            z_pred = np.array([expected_range, expected_bearing])
            y = z - z_pred
            y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # Normalize angle difference

            # Update state and covariance
            self.state += K @ y
            self.covariance = (np.eye(3) - K @ H) @ self.covariance


def main():
    """
    Main function to demonstrate SLAM
    """
    slam = SimpleSLAM()

    # Add some landmarks
    slam.landmarks = [
        (5.0, 0.0),   # Landmark 0
        (10.0, 5.0),  # Landmark 1
        (5.0, 10.0),  # Landmark 2
        (0.0, 5.0),   # Landmark 3
    ]

    # Simulate robot movement and measurements
    true_states = []
    estimated_states = []

    for t in range(100):
        # Simulate control input (move in square)
        if t < 25:
            control = [0.5, 0.0]  # Move forward
        elif t < 50:
            control = [0.0, 0.5]  # Turn
        elif t < 75:
            control = [0.5, 0.0]  # Move forward
        else:
            control = [0.0, 0.5]  # Turn

        # Predict
        slam.predict(control)

        # Add some noise to measurements
        if t % 5 == 0:  # Take measurement every 5 steps
            landmark_id = t % len(slam.landmarks)
            true_landmark = slam.landmarks[landmark_id]

            # True range and bearing with noise
            dx = true_landmark[0] - slam.state[0]
            dy = true_landmark[1] - slam.state[1]
            true_range = np.sqrt(dx**2 + dy**2)
            true_bearing = np.arctan2(dy, dx) - slam.state[2]

            # Add noise
            range_measurement = true_range + np.random.normal(0, 0.05)
            bearing_measurement = true_bearing + np.random.normal(0, 0.02)

            # Update
            slam.update(landmark_id, range_measurement, bearing_measurement)

        # Store states
        estimated_states.append(slam.state.copy())

    # Plot results
    estimated_states = np.array(estimated_states)

    plt.figure(figsize=(10, 8))
    plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'b-', label='Estimated Path')
    plt.scatter([lm[0] for lm in slam.landmarks], [lm[1] for lm in slam.landmarks],
                c='red', s=100, marker='^', label='Landmarks')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Simple SLAM Example')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()