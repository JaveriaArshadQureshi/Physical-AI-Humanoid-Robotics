---
title: "Chapter 6: Sensor Fusion + Localization (SLAM/IMU/LiDAR)"
description: "Techniques for combining sensor data and determining robot position"
---

# Chapter 6: Sensor Fusion + Localization (SLAM/IMU/LiDAR)

## Overview

Sensor fusion and localization are fundamental capabilities for humanoid robots operating in real-world environments. This chapter explores the mathematical foundations, algorithms, and implementation techniques for combining data from multiple sensors to accurately estimate the robot's state and map its surroundings.

## Introduction to Sensor Fusion

### What is Sensor Fusion?

Sensor fusion is the process of combining data from multiple sensors to achieve more accurate and reliable estimates than could be obtained from any individual sensor. In humanoid robotics, this is critical due to the complexity of the robot's environment and the need for precise state estimation.

### Why Sensor Fusion is Critical for Humanoid Robots

- **Redundancy**: Multiple sensors provide backup if one fails
- **Accuracy**: Combining sensors reduces overall uncertainty
- **Robustness**: Different sensors work well in different conditions
- **Completeness**: Different sensors provide complementary information

### Types of Sensor Fusion

#### Data-Level Fusion
- Raw sensor data is combined before processing
- Requires synchronized, time-aligned data
- Computationally intensive but preserves all information

#### Feature-Level Fusion
- Extracted features from different sensors are combined
- Reduces data dimensionality
- Requires feature matching algorithms

#### Decision-Level Fusion
- Individual sensor estimates are combined
- Most computationally efficient
- May lose information during preprocessing

## Sensor Types and Characteristics

### Inertial Measurement Units (IMU)

#### Components
- **Accelerometers**: Measure linear acceleration
- **Gyroscopes**: Measure angular velocity
- **Magnetometers**: Measure magnetic field (compass)

#### Advantages
- High update rates (100Hz-1000Hz)
- Self-contained, no external references
- Works in any environment

#### Limitations
- Drift over time (especially for position)
- Noise accumulation
- Biases and temperature effects

#### IMU Data Processing
```cpp
// Example IMU integration
void integrateIMU(const ImuData& imu, RobotState& state) {
    // Update orientation using gyroscope data
    state.orientation += imu.angular_velocity * dt;

    // Transform acceleration to world frame
    Vector3 world_accel = state.orientation.rotate(imu.linear_acceleration);

    // Integrate acceleration to get velocity and position
    state.velocity += (world_accel - Vector3(0, 0, 9.81)) * dt;  // Subtract gravity
    state.position += state.velocity * dt;
}
```

### LiDAR Sensors

#### Working Principle
- Time-of-flight measurement of laser pulses
- Generates 2D or 3D point clouds
- High accuracy and precision

#### Advantages
- Accurate distance measurements
- Works in low-light conditions
- Reliable geometric features

#### Limitations
- Expensive and heavy
- Limited in dusty/misty conditions
- Lower resolution than cameras

#### LiDAR Processing
- Point cloud filtering and segmentation
- Feature extraction (planes, corners, edges)
- Scan matching for localization

### Cameras

#### Types
- **Monocular**: Single camera, depth from motion
- **Stereo**: Two cameras, triangulation-based depth
- **RGB-D**: Color + depth information

#### Advantages
- Rich semantic information
- Texture and color data
- Relatively inexpensive

#### Limitations
- Performance varies with lighting
- Limited range accuracy
- Computationally intensive processing

### Other Sensors

#### Wheel Encoders
- Relative position estimation
- High accuracy for short distances
- Subject to slippage and drift

#### GPS
- Absolute positioning (outdoor)
- Poor accuracy (typically 1-3m)
- No signal indoors

## Mathematical Foundations

### Probability and Uncertainty

#### Gaussian Distributions
Most sensor fusion algorithms assume Gaussian noise:
- Mean represents the estimate
- Covariance represents uncertainty
- Linear transformations preserve Gaussian nature

#### Bayes' Rule
The foundation of most estimation algorithms:
```
P(state|measurement) ∝ P(measurement|state) × P(state)
```

### State Estimation

#### State Vector
For a humanoid robot, the state might include:
- Position (x, y, z)
- Orientation (roll, pitch, yaw or quaternion)
- Velocities (linear and angular)
- Joint positions and velocities

#### Process Model
Describes how the state evolves over time:
```
x_k = f(x_{k-1}, u_k) + w_k
```
where w_k is process noise

#### Measurement Model
Relates the state to sensor measurements:
```
z_k = h(x_k) + v_k
```
where v_k is measurement noise

## Kalman Filter Family

### Extended Kalman Filter (EKF)

The EKF linearizes nonlinear models around the current estimate:

#### Prediction Step
```
x_pred = f(x_prev, u)
P_pred = F*P_prev*F^T + Q
```

#### Update Step
```
K = P_pred*H^T*(H*P_pred*H^T + R)^(-1)
x_new = x_pred + K*(z - h(x_pred))
P_new = (I - K*H)*P_pred
```

#### Implementation Example
```cpp
class ExtendedKalmanFilter {
public:
    void predict(const Control& u) {
        // Linearize process model
        Matrix F = jacobian_process(state, u);

        // Predict state and covariance
        state = process_model(state, u);
        covariance = F * covariance * F.transpose() + process_noise;
    }

    void update(const Measurement& z) {
        // Linearize measurement model
        Matrix H = jacobian_measurement(state);

        // Compute Kalman gain
        Matrix S = H * covariance * H.transpose() + measurement_noise;
        Matrix K = covariance * H.transpose() * S.inverse();

        // Update state and covariance
        state += K * (z - measurement_model(state));
        covariance = (Matrix::Identity() - K * H) * covariance;
    }

private:
    Vector state;
    Matrix covariance;
    Matrix process_noise;
    Matrix measurement_noise;
};
```

### Unscented Kalman Filter (UKF)

The UKF uses the unscented transform to better handle nonlinearities:

#### Sigma Points
- Generate 2n+1 sigma points around the mean
- Propagate through nonlinear functions
- Reconstruct mean and covariance

#### Advantages over EKF
- Better handling of nonlinearities
- No need to compute Jacobians
- More accurate for highly nonlinear systems

### Particle Filter

A Monte Carlo approach using random samples:

#### Algorithm
1. Generate random particles representing possible states
2. Weight particles based on measurement likelihood
3. Resample particles based on weights
4. Repeat for each time step

#### Advantages
- Handles non-Gaussian distributions
- Works with any nonlinear model
- Robust to outliers

#### Disadvantages
- Computationally expensive
- May suffer from particle depletion
- Requires many particles for high-dimensional problems

## Simultaneous Localization and Mapping (SLAM)

### SLAM Problem

SLAM estimates:
- Robot trajectory over time
- Map of the environment
- Both simultaneously, without prior knowledge

### Mathematical Formulation

The SLAM problem can be formulated as:
```
P(x_{0:t}, m | z_{1:t}, u_{1:t})
```
where x is robot pose, m is map, z is measurements, u is controls

### SLAM Approaches

#### EKF SLAM
- Maintains state vector with robot poses and landmark positions
- Quadratic complexity with map size
- Suitable for small environments

#### Graph-based SLAM
- Formulates as optimization problem
- Nodes: robot poses
- Edges: constraints between poses
- More efficient for large environments

#### Particle Filter SLAM
- Multiple hypotheses about map and pose
- Handles multi-modal distributions
- Computationally expensive

### LiDAR SLAM

#### Scan Matching
- Align consecutive LiDAR scans
- Estimate relative motion
- Build local map

#### Loop Closure
- Detect when robot returns to known location
- Correct accumulated drift
- Optimize global map consistency

#### Common Algorithms
- **Hector SLAM**: Featureless scan matching
- **Gmapping**: Grid-based with particle filter
- **Cartographer**: Real-time mapping with submaps
- **LOAM**: LiDAR Odometry and Mapping

### Visual SLAM

#### Feature-based Methods
- Extract and track visual features
- Estimate motion from feature correspondences
- Examples: ORB-SLAM, LSD-SLAM

#### Direct Methods
- Use raw pixel intensities
- No feature extraction required
- Examples: DSO, LSD-SLAM

## Multi-Sensor Fusion Frameworks

### Robot Operating System (ROS) Integration

#### Robot State Publisher
- Maintains transform tree
- Combines odometry and IMU data
- Publishes TF transforms

#### robot_localization Package
- Extended and Unscented Kalman Filters
- Handles multiple sensor sources
- Supports sensor diagnostics

### Example Configuration
```xml
<!-- Example robot_localization configuration -->
<node pkg="robot_localization" type="ekf_localization_node" name="ekf_se">
  <param name="frequency" value="50"/>
  <param name="sensor_timeout" value="0.1"/>
  <param name="two_d_mode" value="false"/>

  <!-- Process noise -->
  <param name="process_noise_covariance" value="[0.05, 0,    0,    0,    0,    0,
                                                  0,    0.05, 0,    0,    0,    0,
                                                  0,    0,    0.06, 0,    0,    0,
                                                  0,    0,    0,    0.03, 0,    0,
                                                  0,    0,    0,    0,    0.03, 0,
                                                  0,    0,    0,    0,    0,    0.06]"/>

  <!-- IMU configuration -->
  <rosparam param="imu0_config">[false, false, false,
                                 true,  true,  true,
                                 false, false, false,
                                 true,  true,  true,
                                 true,  true,  true]</rosparam>
</node>
```

## Humanoid Robot Specific Considerations

### Bipedal Challenges

#### Dynamic Balance
- Center of mass constantly moving
- Zero moment point (ZMP) considerations
- Need for rapid state estimation

#### Sensor Placement
- IMUs on multiple body segments
- Force/torque sensors in feet
- Integration with whole-body control

### Whole-Body State Estimation

#### Kinematic Constraints
- Joint angle measurements
- Kinematic chain relationships
- Contact state estimation

#### Dynamic State Estimation
- Incorporating dynamics models
- Contact force integration
- Momentum-based estimation

### Example: Humanoid State Estimation
```cpp
class HumanoidStateEstimator {
public:
    void updateState(const std::vector<SensorData>& sensors) {
        // Fuse IMU data for orientation
        updateOrientation(sensors.imu);

        // Use forward kinematics with joint encoders
        updatePositionFromKinematics(sensors.encoders);

        // Integrate with force/torque sensors for contact state
        updateContactState(sensors.ft_sensors);

        // Apply kinematic constraints
        applyConstraints();
    }

private:
    void updateOrientation(const ImuData& imu) {
        // Use complementary filter or EKF to fuse gyroscope and accelerometer
        orientation_integrator.update(imu.gyro, imu.accel);
    }

    void updatePositionFromKinematics(const std::vector<double>& joint_angles) {
        // Forward kinematics to get end-effector positions
        // Use kinematic constraints to estimate base position
        kinematics_solver.solve(joint_angles);
    }

    Vector3 orientation;
    Vector3 position;
    Matrix6 covariance;
    std::vector<ContactState> contact_states;
};
```

## Advanced Techniques

### Factor Graphs

#### Mathematical Foundation
- Nodes: Variables to estimate
- Factors: Constraints between variables
- Optimization to minimize constraint violations

#### Libraries
- **GTSAM**: Georgia Tech Smoothing and Mapping
- **Ceres Solver**: Google's optimization library
- **g2o**: General graph optimization

### Deep Learning Integration

#### Learning-based Fusion
- Neural networks for sensor fusion
- End-to-end learning of state estimation
- Handling sensor failures adaptively

#### Uncertainty Quantification
- Bayesian neural networks
- Monte Carlo dropout
- Deep ensembles

## Performance Evaluation

### Metrics

#### Accuracy Metrics
- Absolute trajectory error (ATE)
- Relative pose error (RPE)
- Root mean square error (RMSE)

#### Computational Metrics
- Processing time per step
- Memory usage
- CPU utilization

### Benchmarking

#### Standard Datasets
- KITTI dataset for outdoor SLAM
- EuRoC MAV for aerial robots
- Custom humanoid datasets

#### Evaluation Protocols
- Cross-validation
- Real-world testing
- Simulation-to-reality transfer

## Implementation Considerations

### Real-time Requirements

#### Computational Efficiency
- Efficient data structures
- Approximation algorithms
- Parallel processing

#### Memory Management
- Fixed-size data structures
- Memory pools
- Avoid dynamic allocation in critical loops

### Robustness

#### Sensor Failure Handling
- Detection of sensor failures
- Graceful degradation
- Sensor validation

#### Outlier Rejection
- RANSAC for outlier removal
- Statistical validation
- Adaptive thresholding

## Troubleshooting Common Issues

### Sensor Synchronization
- Hardware vs. software timestamping
- Communication delays
- Buffer management

### Drift and Bias
- Long-term drift compensation
- Bias estimation and correction
- Calibration procedures

### Computational Bottlenecks
- Algorithm optimization
- Parallel processing
- Approximation techniques

## Conclusion

Sensor fusion and localization are critical capabilities for humanoid robots to operate effectively in unstructured environments. The combination of multiple sensors through sophisticated algorithms enables robots to maintain accurate state estimates despite sensor noise, environmental challenges, and dynamic motion.

Modern approaches combine classical estimation theory with advanced optimization techniques and, increasingly, machine learning methods. The choice of approach depends on the specific requirements of the application, including accuracy needs, computational constraints, and environmental conditions.

The next chapter will explore kinematics and dynamics, which are essential for understanding and controlling the motion of humanoid robots.

## Exercises

1. Implement a simple Extended Kalman Filter to fuse IMU and wheel encoder data for position estimation.

2. Research and compare different SLAM algorithms (EKF SLAM, Graph SLAM, etc.) in terms of accuracy, computational requirements, and robustness.

3. Design a sensor fusion architecture for a humanoid robot that includes IMU, LiDAR, cameras, and joint encoders.

4. Implement a basic particle filter for robot localization in a known map.

5. Research how deep learning is being integrated with traditional sensor fusion approaches.