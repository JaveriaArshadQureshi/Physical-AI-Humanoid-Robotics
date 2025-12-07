---
title: "Chapter 7: Kinematics & Dynamics (FK, IK, Trajectory Planning)"
description: "Mathematical foundations for robot motion and force analysis"
---

# Chapter 7: Kinematics & Dynamics (FK, IK, Trajectory Planning)

## Overview

Kinematics and dynamics form the mathematical foundation for understanding and controlling robot motion. This chapter covers forward and inverse kinematics for determining robot configurations, dynamic modeling for understanding forces and motion, and trajectory planning for generating smooth, feasible movements for humanoid robots.

## Introduction to Robot Kinematics

### What is Kinematics?

Kinematics is the study of motion without considering the forces that cause it. In robotics, kinematics deals with the relationship between joint variables and the position and orientation of robot links.

### Coordinate Systems and Representations

#### Homogeneous Transformations

Homogeneous transformation matrices combine rotation and translation:
```
T = [R  p]
    [0  1]
```
where R is a 3x3 rotation matrix and p is a 3x1 position vector.

#### Rotation Representations

##### Rotation Matrices
- 3x3 orthogonal matrices
- 9 parameters with 6 constraints (orthonormality)
- No singularities but over-parameterized

##### Euler Angles
- 3 parameters (e.g., ZYX convention)
- Intuitive interpretation
- Singularity at gimbal lock

##### Quaternions
- 4 parameters with unit norm constraint
- No singularities
- Efficient for composition and interpolation

```cpp
// Quaternion implementation for rotations
class Quaternion {
public:
    double w, x, y, z;

    // Quaternion multiplication (composition of rotations)
    Quaternion operator*(const Quaternion& q) const {
        return {
            w*q.w - x*q.x - y*q.y - z*q.z,  // w
            w*q.x + x*q.w + y*q.z - z*q.y,  // x
            w*q.y - x*q.z + y*q.w + z*q.x,  // y
            w*q.z + x*q.y - y*q.x + z*q.w   // z
        };
    }

    // Convert quaternion to rotation matrix
    Matrix3 toRotationMatrix() const {
        Matrix3 R;
        double xx = x*x, yy = y*y, zz = z*z;
        double xy = x*y, xz = x*z, yz = y*z;
        double wx = w*x, wy = w*y, wz = w*z;

        R(0,0) = 1 - 2*(yy + zz); R(0,1) = 2*(xy - wz);  R(0,2) = 2*(xz + wy);
        R(1,0) = 2*(xy + wz);   R(1,1) = 1 - 2*(xx + zz); R(1,2) = 2*(yz - wx);
        R(2,0) = 2*(xz - wy);   R(2,1) = 2*(yz + wx);   R(2,2) = 1 - 2*(xx + yy);

        return R;
    }
};
```

## Forward Kinematics (FK)

### Definition

Forward kinematics computes the end-effector pose given the joint angles:
```
T_end_effector = f(q1, q2, ..., qn)
```

### Denavit-Hartenberg (DH) Convention

The DH convention provides a systematic way to define coordinate frames for robot links:

#### DH Parameters
- **a_i**: Link length (distance along x_i from z_{i-1} to z_i)
- **α_i**: Link twist (angle from z_{i-1} to z_i about x_i)
- **d_i**: Link offset (distance along z_{i-1} from x_{i-1} to x_i)
- **θ_i**: Joint angle (angle from x_{i-1} to x_i about z_{i-1})

#### Transformation Matrix
```
A_i = [cos(θ_i)   -sin(θ_i)*cos(α_i)   sin(θ_i)*sin(α_i)   a_i*cos(θ_i)]
      [sin(θ_i)    cos(θ_i)*cos(α_i)   -cos(θ_i)*sin(α_i)   a_i*sin(θ_i)]
      [0           sin(α_i)            cos(α_i)             d_i        ]
      [0           0                   0                    1          ]
```

### Forward Kinematics Implementation

```cpp
class RobotKinematics {
public:
    RobotKinematics(const std::vector<DHParams>& dh_params)
        : dh_params_(dh_params) {}

    Transform forwardKinematics(const std::vector<double>& joint_angles) {
        Transform T = Transform::Identity();

        for (size_t i = 0; i < joint_angles.size(); ++i) {
            Transform A_i = computeDHTransform(dh_params_[i], joint_angles[i]);
            T = T * A_i;
        }

        return T;
    }

    std::vector<Transform> forwardKinematicsAllLinks(
        const std::vector<double>& joint_angles) {
        std::vector<Transform> link_poses;
        Transform T = Transform::Identity();

        link_poses.push_back(T);  // Base frame

        for (size_t i = 0; i < joint_angles.size(); ++i) {
            Transform A_i = computeDHTransform(dh_params_[i], joint_angles[i]);
            T = T * A_i;
            link_poses.push_back(T);
        }

        return link_poses;
    }

private:
    struct DHParams {
        double a, alpha, d, theta_offset;
    };

    Transform computeDHTransform(const DHParams& params, double joint_angle) {
        double th = params.theta_offset + joint_angle;

        Transform T;
        T(0,0) = cos(th); T(0,1) = -sin(th)*cos(params.alpha); T(0,2) = sin(th)*sin(params.alpha); T(0,3) = params.a*cos(th);
        T(1,0) = sin(th); T(1,1) = cos(th)*cos(params.alpha);  T(1,2) = -cos(th)*sin(params.alpha); T(1,3) = params.a*sin(th);
        T(2,0) = 0;       T(2,1) = sin(params.alpha);         T(2,2) = cos(params.alpha);         T(2,3) = params.d;
        T(3,0) = 0;       T(3,1) = 0;                         T(3,2) = 0;                         T(3,3) = 1;

        return T;
    }

    std::vector<DHParams> dh_params_;
};
```

## Inverse Kinematics (IK)

### Definition

Inverse kinematics computes the joint angles required to achieve a desired end-effector pose:
```
q = f^(-1)(T_desired)
```

### Analytical vs. Numerical Solutions

#### Analytical Solutions
- Closed-form solutions for specific robot geometries
- Fast computation
- Limited to robots with special structures

#### Numerical Solutions
- General approach for any robot geometry
- Iterative computation
- May converge to local minima

### Jacobian-based Methods

#### Jacobian Matrix

The Jacobian relates joint velocities to end-effector velocities:
```
v_ee = J(q) * q_dot
```

Where:
- v_ee: End-effector velocity (linear + angular)
- q_dot: Joint velocity vector
- J(q): Jacobian matrix

#### Jacobian Computation

```cpp
class InverseKinematics {
public:
    // Compute geometric Jacobian
    Matrix6xN computeJacobian(const std::vector<double>& joint_angles) {
        std::vector<Transform> link_poses = kinematics_.forwardKinematicsAllLinks(joint_angles);

        Matrix6xN J = Matrix6xN::Zero(6, joint_angles.size());

        Transform T_end = link_poses.back();
        Vector3 p_end = T_end.translation();

        for (int i = 0; i < joint_angles.size(); ++i) {
            Transform T_i = link_poses[i];
            Vector3 z_i = T_i.rotation() * Vector3::UnitZ();  // Joint axis
            Vector3 p_i = T_i.translation();                  // Joint position

            // Linear velocity component
            J.block<3,1>(0, i) = z_i.cross(p_end - p_i);

            // Angular velocity component
            J.block<3,1>(3, i) = z_i;
        }

        return J;
    }

    // Solve IK using Jacobian transpose method
    std::vector<double> solveIK(const Transform& target_pose,
                                const std::vector<double>& initial_joints,
                                double tolerance = 1e-4,
                                int max_iterations = 1000) {
        std::vector<double> q = initial_joints;

        for (int iter = 0; iter < max_iterations; ++iter) {
            Transform current_pose = kinematics_.forwardKinematics(q);
            Transform error_transform = target_pose * current_pose.inverse();

            // Extract position and orientation errors
            Vector3 position_error = error_transform.translation();
            Vector3 orientation_error = rotationMatrixToAxisAngle(error_transform.rotation());

            Vector6 error;
            error << position_error, orientation_error;

            if (error.norm() < tolerance) {
                break;  // Converged
            }

            // Compute Jacobian
            Matrix6xN J = computeJacobian(q);

            // Compute joint velocity
            std::vector<double> q_dot = pseudoInverse(J) * error * 0.1;  // Damping factor

            // Update joint angles
            for (size_t i = 0; i < q.size(); ++i) {
                q[i] += q_dot[i];
            }
        }

        return q;
    }

private:
    RobotKinematics kinematics_;

    // Pseudo-inverse using SVD
    std::vector<double> pseudoInverse(const Matrix6xN& J, const Vector6& error) {
        // Using SVD to compute pseudo-inverse: J^+ = V * S^+ * U^T
        Eigen::JacobiSVD<Matrix6xN> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
        double tolerance = 1e-6 * std::max(J.cols(), J.rows());

        VectorN s_inv = svd.singularValues();
        for (int i = 0; i < s_inv.size(); ++i) {
            s_inv[i] = (s_inv[i] > tolerance) ? 1.0 / s_inv[i] : 0.0;
        }

        MatrixNx6 J_inv = svd.matrixV() * s_inv.asDiagonal() * svd.matrixU().transpose();
        return (J_inv * error).eval();
    }

    Vector3 rotationMatrixToAxisAngle(const Matrix3& R) {
        // Convert rotation matrix to axis-angle representation
        double angle = acos(std::min(1.0, std::max(-1.0, (R(0,0) + R(1,1) + R(2,2) - 1) / 2)));

        if (angle < 1e-6) {
            return Vector3::Zero();  // Identity rotation
        }

        Vector3 axis;
        axis[0] = R(2,1) - R(1,2);
        axis[1] = R(0,2) - R(2,0);
        axis[2] = R(1,0) - R(0,1);

        axis = axis / (2 * sin(angle));
        return axis * angle;
    }
};
```

### Common IK Algorithms

#### Jacobian Transpose Method
- Simple but may be slow to converge
- J_dot = α * J^T * e
- Good for redundant manipulators

#### Jacobian Pseudo-inverse Method
- More accurate than transpose method
- J_dot = α * J^+ * e
- Better convergence properties

#### Damped Least Squares (DLS)
- Adds damping to handle singularities
- J_dot = α * (J^T * J + λ² * I)^(-1) * J^T * e
- Robust to singular configurations

## Humanoid-Specific Kinematics

### Multi-Chain Kinematics

Humanoid robots have multiple kinematic chains (arms, legs, spine):
- Each chain has its own IK solver
- Need coordination between chains
- Balance constraints for bipedal locomotion

### Whole-Body IK

#### Task-Based Formulation
- Multiple tasks with priorities
- Primary tasks (end-effector pose) vs. secondary tasks (posture)
- Optimization-based approach

#### Example: Humanoid Whole-Body IK
```cpp
class HumanoidWholeBodyIK {
public:
    struct Task {
        int link_id;
        Transform desired_pose;
        Vector6 weights;  // Task weights for position/orientation
        int priority;     // Task priority (0 = highest)
    };

    std::vector<double> solve(const std::vector<Task>& tasks,
                             const std::vector<double>& initial_joints,
                             const std::vector<double>& posture_weights) {
        // Hierarchical optimization approach
        std::vector<double> q = initial_joints;

        // Solve for each priority level
        for (int priority = 0; priority < max_priority_; ++priority) {
            std::vector<Task> priority_tasks = getTasksByPriority(tasks, priority);

            if (!priority_tasks.empty()) {
                q = solvePriorityLevel(priority_tasks, q, posture_weights);
            }
        }

        return q;
    }

private:
    std::vector<double> solvePriorityLevel(const std::vector<Task>& tasks,
                                          const std::vector<double>& current_joints,
                                          const std::vector<double>& posture_weights) {
        // Formulate as constrained optimization problem
        // min ||J * dq - desired_velocities||^2 + regularization
        // subject to joint limits and constraints

        // Use quadratic programming solver
        return solveQP(tasks, current_joints, posture_weights);
    }

    int max_priority_ = 5;
};
```

## Robot Dynamics

### Dynamic Equations

The equations of motion for a robot manipulator are given by the Euler-Lagrange equations:

```
M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = τ
```

Where:
- M(q): Mass/inertia matrix
- C(q, q_dot): Coriolis and centrifugal forces
- g(q): Gravity forces
- τ: Joint torques
- q: Joint positions
- q_dot: Joint velocities
- q_ddot: Joint accelerations

### Dynamic Parameters

#### Mass Matrix
- Represents inertial properties
- Depends on joint configuration
- Positive definite

#### Coriolis Matrix
- Represents velocity-coupling effects
- Depends on joint positions and velocities

#### Gravity Vector
- Represents gravitational forces
- Depends on joint configuration

### Recursive Newton-Euler Algorithm (RNEA)

Efficient algorithm to compute inverse dynamics:

```cpp
class RobotDynamics {
public:
    struct LinkParams {
        double mass;
        Vector3 com;      // Center of mass
        Matrix3 inertia;  // Inertia tensor at COM
    };

    // Inverse dynamics: compute torques for given motion
    std::vector<double> inverseDynamics(
        const std::vector<double>& joint_positions,
        const std::vector<double>& joint_velocities,
        const std::vector<double>& joint_accelerations,
        const std::vector<LinkParams>& link_params,
        const std::vector<double>& external_forces) {

        int n = joint_positions.size();
        std::vector<double> torques(n);

        // Forward pass: compute velocities and accelerations
        std::vector<Vector3> v(n + 1);      // Linear velocities
        std::vector<Vector3> omega(n + 1);  // Angular velocities
        std::vector<Vector3> a(n + 1);      // Linear accelerations
        std::vector<Vector3> alpha(n + 1);  // Angular accelerations

        // Initialize base conditions (assuming fixed base)
        v[0] = Vector3::Zero();
        omega[0] = Vector3::Zero();
        a[0] = -gravity_;  // Gravity acceleration
        alpha[0] = Vector3::Zero();

        // Forward recursion
        for (int i = 0; i < n; ++i) {
            // Compute transformation from link i-1 to i
            Transform T_i = computeLinkTransform(joint_positions[i], i);

            // Joint axis in local frame
            Vector3 z_i = T_i.rotation() * Vector3::UnitZ();

            // Link velocity and acceleration
            omega[i+1] = omega[i] + z_i * joint_velocities[i];
            alpha[i+1] = alpha[i] + z_i * joint_accelerations[i] +
                        omega[i+1].cross(z_i * joint_velocities[i]);

            // Linear velocity and acceleration of link i origin
            Vector3 r_i = T_i.translation();  // Position of joint i+1 in frame i
            v[i+1] = v[i] + omega[i].cross(r_i) + z_i * joint_velocities[i];
            a[i+1] = a[i] + alpha[i].cross(r_i) + omega[i].cross(omega[i].cross(r_i)) +
                     2 * omega[i].cross(z_i * joint_velocities[i]) +
                     z_i * joint_accelerations[i];

            // Velocity and acceleration of center of mass
            Vector3 r_com = link_params[i].com;
            Vector3 v_com = v[i+1] + omega[i+1].cross(r_com);
            Vector3 a_com = a[i+1] + alpha[i+1].cross(r_com) +
                           omega[i+1].cross(omega[i+1].cross(r_com));

            // Compute forces and torques on link i
            Vector3 f_i = link_params[i].mass * a_com;
            Vector3 tau_i = link_params[i].inertia * alpha[i+1] +
                           omega[i+1].cross(link_params[i].inertia * omega[i+1]);

            // Joint torque
            torques[i] = tau_i.dot(z_i) + external_forces[i];
        }

        return torques;
    }

    // Forward dynamics: compute accelerations for given torques
    std::vector<double> forwardDynamics(
        const std::vector<double>& joint_positions,
        const std::vector<double>& joint_velocities,
        const std::vector<double>& joint_torques,
        const std::vector<LinkParams>& link_params) {

        // Compute mass matrix, Coriolis terms, and gravity
        MatrixNd M = computeMassMatrix(joint_positions, link_params);
        VectorNd C = computeCoriolisAndGravity(joint_positions, joint_velocities, link_params);

        // Solve M*q_ddot = tau - C
        VectorNd q_ddot = M.inverse() * (Eigen::Map<VectorNd>(joint_torques.data(), joint_torques.size()) - C);

        return std::vector<double>(q_ddot.data(), q_ddot.data() + q_ddot.size());
    }

private:
    Vector3 gravity_ = Vector3(0, 0, -9.81);  // Gravity vector

    MatrixNd computeMassMatrix(const std::vector<double>& q,
                              const std::vector<LinkParams>& link_params) {
        // Compute mass matrix using composite rigid body algorithm
        // This is a simplified implementation
        int n = q.size();
        MatrixNd M = MatrixNd::Zero(n, n);

        // For each pair of joints, compute the mass matrix element
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                M(i, j) = computeMassMatrixElement(i, j, q, link_params);
            }
        }

        return M;
    }

    double computeMassMatrixElement(int i, int j,
                                   const std::vector<double>& q,
                                   const std::vector<LinkParams>& link_params) {
        // Implementation of mass matrix element computation
        // This involves complex calculations of partial velocities
        // For brevity, returning a placeholder
        return 0.0;
    }

    VectorNd computeCoriolisAndGravity(const std::vector<double>& q,
                                      const std::vector<double>& q_dot,
                                      const std::vector<LinkParams>& link_params) {
        // Compute Coriolis/centrifugal and gravity terms
        // This is a simplified implementation
        int n = q.size();
        VectorNd C = VectorNd::Zero(n);

        // Placeholder implementation
        for (int i = 0; i < n; ++i) {
            // Add Coriolis and gravity effects
            C[i] = 0.0;  // Placeholder
        }

        return C;
    }

    Transform computeLinkTransform(double joint_angle, int link_index) {
        // Compute transformation matrix for a joint
        Transform T = Transform::Identity();
        T.rotate(AngleAxis(joint_angle, Vector3::UnitZ()));
        return T;
    }
};
```

## Trajectory Planning

### Overview

Trajectory planning generates smooth, feasible paths for robot motion that satisfy kinematic and dynamic constraints.

### Path vs. Trajectory

- **Path**: Geometric route (position only)
- **Trajectory**: Path with timing information (position, velocity, acceleration)

### Polynomial Trajectories

#### Cubic Polynomials
For smooth motion with specified start/end positions and velocities:

```
q(t) = a0 + a1*t + a2*t^2 + a3*t^3
```

Boundary conditions:
- q(0) = q_start, q(T) = q_end
- q_dot(0) = v_start, q_dot(T) = v_end

#### Quintic Polynomials
For smoother motion with specified accelerations:

```
q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
```

```cpp
class PolynomialTrajectory {
public:
    struct BoundaryConditions {
        double start_pos, start_vel, start_acc;
        double end_pos, end_vel, end_acc;
        double duration;
    };

    PolynomialTrajectory(const BoundaryConditions& bc) {
        // Solve for polynomial coefficients
        double T = bc.duration;
        double T2 = T*T, T3 = T2*T, T4 = T3*T, T5 = T4*T;

        // Matrix form: A * coeffs = boundary_conditions
        Matrix6d A;
        A << 1, 0,   0,    0,     0,      0,
             0, 1,   0,    0,     0,      0,
             0, 0,   2,    0,     0,      0,
             1, T,   T2,   T3,    T4,     T5,
             0, 1,   2*T,  3*T2,  4*T3,   5*T4,
             0, 0,   2,    6*T,   12*T2,  20*T3;

        Vector6d b;
        b << bc.start_pos, bc.start_vel, bc.start_acc,
             bc.end_pos, bc.end_vel, bc.end_acc;

        Vector6d coeffs = A.inverse() * b;

        a0 = coeffs[0]; a1 = coeffs[1]; a2 = coeffs[2];
        a3 = coeffs[3]; a4 = coeffs[4]; a5 = coeffs[5];
    }

    double position(double t) const {
        return a0 + a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t;
    }

    double velocity(double t) const {
        return a1 + 2*a2*t + 3*a3*t*t + 4*a4*t*t*t + 5*a5*t*t*t*t;
    }

    double acceleration(double t) const {
        return 2*a2 + 6*a3*t + 12*a4*t*t + 20*a5*t*t*t;
    }

private:
    double a0, a1, a2, a3, a4, a5;
};
```

### Joint-Space vs. Cartesian-Space Trajectories

#### Joint-Space Trajectories
- Plan in joint angle space
- Direct control of actuators
- Avoids kinematic singularities

#### Cartesian-Space Trajectories
- Plan in task space (end-effector position/orientation)
- Intuitive for task-oriented motion
- Requires IK at each time step

### Humanoid-Specific Trajectory Planning

#### Walking Pattern Generation
- Zero Moment Point (ZMP) planning
- Capture Point based control
- Footstep planning

#### Whole-Body Motion Planning
- Coordinated motion of multiple chains
- Balance and stability constraints
- Collision avoidance

### Real-time Trajectory Generation

#### Online Trajectory Modification
- Adjust trajectories based on sensor feedback
- Handle dynamic obstacles
- Maintain stability constraints

#### Model Predictive Control (MPC)
- Receding horizon optimization
- Constraint handling
- Feedback integration

## Implementation in ROS2

### MoveIt! Integration

MoveIt! provides high-level motion planning capabilities:

```cpp
// Example MoveIt! trajectory planning
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

class HumanoidMotionPlanner {
public:
    HumanoidMotionPlanner(const std::string& group_name) {
        move_group_interface_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(group_name);
        planning_scene_interface_ = std::make_shared<moveit::planning_scene_interface::PlanningSceneInterface>();
    }

    bool planToPose(const geometry_msgs::msg::Pose& target_pose) {
        move_group_interface_->setPoseTarget(target_pose);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = (move_group_interface_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (success) {
            // Execute the plan
            move_group_interface_->execute(plan);
        }

        return success;
    }

    bool planToJointValues(const std::vector<double>& joint_values) {
        move_group_interface_->setJointValueTarget(joint_values);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        bool success = (move_group_interface_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (success) {
            move_group_interface_->execute(plan);
        }

        return success;
    }

private:
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface_;
    std::shared_ptr<moveit::planning_scene_interface::PlanningSceneInterface> planning_scene_interface_;
};
```

## Performance Considerations

### Computational Complexity

#### Kinematics
- Forward kinematics: O(n) for n joints
- Inverse kinematics: O(n³) for numerical methods
- Optimization-based IK: Depends on solver

#### Dynamics
- Inverse dynamics (RNEA): O(n)
- Forward dynamics: O(n³) due to matrix inversion
- Mass matrix computation: O(n²) to O(n³)

### Real-time Implementation

#### Code Optimization
- Efficient data structures
- Pre-computed values where possible
- Avoid dynamic memory allocation in control loops

#### Parallel Processing
- Multi-threading for sensor processing
- GPU acceleration for perception
- Parallel kinematics computation

## Troubleshooting Common Issues

### Kinematic Singularities
- Detect and avoid singular configurations
- Use damped pseudo-inverse methods
- Implement joint limit avoidance

### Dynamic Model Errors
- Parameter identification procedures
- Adaptive control techniques
- Robust control design

### Trajectory Feasibility
- Velocity and acceleration limits
- Torque constraints
- Real-time replanning capabilities

## Conclusion

Kinematics and dynamics provide the mathematical foundation for understanding and controlling robot motion. Forward kinematics allows us to determine end-effector positions from joint angles, while inverse kinematics solves the reverse problem. Dynamic modeling enables the understanding of forces and motion, essential for controlled robot behavior.

Trajectory planning bridges the gap between high-level goals and low-level control, generating smooth, feasible paths that respect robot constraints. For humanoid robots, these concepts become even more complex due to the multi-chain nature and balance requirements.

The next chapter will explore control systems in detail, building on the kinematic and dynamic foundations established here to create responsive, stable robot controllers.

## Exercises

1. Implement forward kinematics for a 6-DOF manipulator using the DH convention.

2. Derive and implement the Jacobian matrix for a simple 2-link planar manipulator.

3. Implement a trajectory planner that generates smooth quintic polynomial trajectories between two points.

4. Research and implement an inverse kinematics solver for a humanoid robot arm using the Jacobian pseudoinverse method.

5. Investigate how dynamic models are used in operational space control for humanoid robots.