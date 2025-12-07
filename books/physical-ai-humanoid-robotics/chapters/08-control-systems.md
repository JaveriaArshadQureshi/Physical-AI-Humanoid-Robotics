---
title: "Chapter 8: Control Systems (PID, MPC, Whole-Body Control)"
description: "Designing control systems for stable and responsive robot behavior"
---

# Chapter 8: Control Systems (PID, MPC, Whole-Body Control)

## Overview

Control systems are the nervous system of humanoid robots, translating high-level goals into precise motor commands that enable stable, responsive, and purposeful behavior. This chapter explores various control strategies, from classical PID controllers to advanced model predictive and whole-body control approaches, with specific focus on the unique challenges of humanoid robotics.

## Introduction to Robot Control

### Control System Fundamentals

A robot control system typically consists of:
- **Controller**: Computes desired actions based on state and reference
- **Plant**: The physical robot system being controlled
- **Sensors**: Measure robot state and environment
- **Actuators**: Execute control commands

### Control Architecture

#### Hierarchical Control Structure
```
High-Level Planner → Trajectory Generator → Low-Level Controller → Robot
```

- **High-Level**: Task planning, path planning
- **Mid-Level**: Trajectory generation, feedback control
- **Low-Level**: Joint control, safety systems

### Control System Properties

#### Stability
- System returns to equilibrium after disturbances
- Bounded input, bounded output (BIBO) stability
- Lyapunov stability for nonlinear systems

#### Performance
- Response speed and accuracy
- Disturbance rejection
- Robustness to model uncertainties

#### Real-time Requirements
- Deterministic execution times
- High-frequency control loops (typically 100Hz-1kHz)
- Predictable computational complexity

## Classical Control: PID Controllers

### PID Fundamentals

The Proportional-Integral-Derivative (PID) controller is the most widely used control strategy:

```
u(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de(t)/dt
```

Where:
- u(t): Control output
- e(t): Error signal (reference - measurement)
- Kp: Proportional gain
- Ki: Integral gain
- Kd: Derivative gain

### PID Implementation

```cpp
class PIDController {
public:
    PIDController(double kp, double ki, double kd, double dt)
        : kp_(kp), ki_(ki), kd_(kd), dt_(dt), integral_(0), prev_error_(0) {}

    double compute(double error) {
        // Proportional term
        double p_term = kp_ * error;

        // Integral term with anti-windup
        integral_ += error * dt_;
        // Clamp integral to prevent windup
        integral_ = std::clamp(integral_, -integral_limit_, integral_limit_);
        double i_term = ki_ * integral_;

        // Derivative term (using measured value to avoid derivative kick)
        double derivative = (error - prev_error_) / dt_;
        double d_term = kd_ * derivative;

        prev_error_ = error;

        double output = p_term + i_term + d_term;
        return std::clamp(output, output_min_, output_max_);
    }

    void reset() {
        integral_ = 0;
        prev_error_ = 0;
    }

    void setOutputLimits(double min, double max) {
        output_min_ = min;
        output_max_ = max;
    }

    void setIntegralLimits(double limit) {
        integral_limit_ = limit;
    }

private:
    double kp_, ki_, kd_;
    double dt_;
    double integral_, prev_error_;
    double output_min_ = -std::numeric_limits<double>::infinity();
    double output_max_ = std::numeric_limits<double>::infinity();
    double integral_limit_ = 100.0;
};
```

### PID Tuning Methods

#### Ziegler-Nichols Method
1. Set Ki = Kd = 0
2. Increase Kp until system oscillates
3. Record critical gain (Kc) and oscillation period (Pc)
4. Apply tuning rules:
   - P: Kp = 0.5 * Kc
   - PI: Kp = 0.45 * Kc, Ki = 0.81 * Kc / Pc
   - PID: Kp = 0.6 * Kc, Ki = 1.2 * Kc / Pc, Kd = 0.075 * Kc * Pc

#### Cohen-Coon Method
Better for systems with time delays:
- More conservative than Ziegler-Nichols
- Better stability margins

#### Manual Tuning Strategy
1. Start with P-only control
2. Increase Kp until response is fast but not oscillating
3. Add small Ki to eliminate steady-state error
4. Add Kd to reduce overshoot if needed

### PID in Robotics Applications

#### Joint Position Control
```cpp
class JointController {
public:
    JointController(int joint_id, double kp, double ki, double kd)
        : joint_id_(joint_id), pid_(kp, ki, kd, 0.01) {
        // Initialize with appropriate gains for the specific joint
    }

    double update(double desired_position, double current_position) {
        double error = desired_position - current_position;
        double control_effort = pid_.compute(error);

        // Apply control effort to joint (e.g., as torque or current)
        applyTorque(control_effort);

        return control_effort;
    }

private:
    int joint_id_;
    PIDController pid_;

    void applyTorque(double torque) {
        // Send torque command to joint actuator
        // Implementation depends on hardware interface
    }
};
```

#### Trajectory Following
- Feedforward terms for known trajectory
- Feedback for error correction
- Cascaded control loops

## Advanced Control Techniques

### Model Predictive Control (MPC)

MPC solves an optimization problem at each time step to determine the optimal control sequence:

#### Mathematical Formulation
```
min Σ(k=0 to N-1) [x(k)ᵀQx(k) + u(k)ᵀRu(k)] + x(N)ᵀP x(N)
s.t. x(k+1) = f(x(k), u(k))
     x_min ≤ x(k) ≤ x_max
     u_min ≤ u(k) ≤ u_max
```

#### MPC Implementation for Robotics
```cpp
class ModelPredictiveController {
public:
    struct State {
        std::vector<double> positions;
        std::vector<double> velocities;
    };

    struct ControlInput {
        std::vector<double> torques;
    };

    ModelPredictiveController(
        const std::function<State(State, ControlInput, double)>& dynamics,
        int prediction_horizon, double dt)
        : dynamics_func_(dynamics), N_(prediction_horizon), dt_(dt) {}

    ControlInput computeControl(const State& current_state,
                               const std::vector<State>& reference_trajectory) {
        // Solve optimization problem using quadratic programming
        // This is a simplified outline - full implementation requires QP solver

        // Define cost function
        auto cost_function = [&](const std::vector<ControlInput>& control_sequence) -> double {
            double cost = 0.0;
            State state = current_state;

            for (int i = 0; i < N_; ++i) {
                // State cost
                cost += computeStateCost(state, reference_trajectory[i]);

                // Control cost
                cost += computeControlCost(control_sequence[i]);

                // Predict next state
                state = dynamics_func_(state, control_sequence[i], dt_);
            }

            // Terminal cost
            cost += computeTerminalCost(state, reference_trajectory.back());

            return cost;
        };

        // Optimize control sequence (would use QP solver in practice)
        std::vector<ControlInput> optimal_sequence = optimize(cost_function);

        // Return first control in sequence (receding horizon)
        return optimal_sequence[0];
    }

private:
    std::function<State(State, ControlInput, double)> dynamics_func_;
    int N_;  // Prediction horizon
    double dt_;  // Time step

    double computeStateCost(const State& state, const State& reference) {
        // Compute quadratic cost for state tracking
        double cost = 0.0;
        for (size_t i = 0; i < state.positions.size(); ++i) {
            double pos_error = state.positions[i] - reference.positions[i];
            double vel_error = state.velocities[i] - reference.velocities[i];
            cost += pos_error * pos_error * position_weight_;
            cost += vel_error * vel_error * velocity_weight_;
        }
        return cost;
    }

    double position_weight_ = 1.0;
    double velocity_weight_ = 0.1;
    double control_weight_ = 0.01;
    double terminal_weight_ = 10.0;
};
```

### Linear Quadratic Regulator (LQR)

LQR provides optimal control for linear systems with quadratic costs:

#### Continuous-time LQR
For system: `dx/dt = Ax + Bu`
Cost: `J = ∫(x^T Q x + u^T R u) dt`

Optimal control: `u = -Kx` where `K = R^(-1) B^T P`
P is found by solving the Algebraic Riccati Equation.

```cpp
class LQRController {
public:
    LQRController(const MatrixXd& A, const MatrixXd& B,
                  const MatrixXd& Q, const MatrixXd& R) {
        // Solve Algebraic Riccati Equation: A^T P + P A - P B R^(-1) B^T P + Q = 0
        MatrixXd P = solveRiccatiEquation(A, B, Q, R);

        // Compute optimal gain matrix
        K_ = R.inverse() * B.transpose() * P;
    }

    VectorXd computeControl(const VectorXd& state) {
        return -K_ * state;  // u = -Kx
    }

private:
    MatrixXd K_;  // Optimal gain matrix

    MatrixXd solveRiccatiEquation(const MatrixXd& A, const MatrixXd& B,
                                  const MatrixXd& Q, const MatrixXd& R) {
        // Implementation of Riccati equation solver
        // This would typically use numerical methods like Schur decomposition
        // For brevity, returning a placeholder
        return MatrixXd::Identity(A.rows(), A.cols());
    }
};
```

## Humanoid-Specific Control Challenges

### Balance and Stability

#### Zero Moment Point (ZMP)
ZMP is a crucial concept for bipedal locomotion:

```
ZMP_x = (M*g*x_com - I*theta_ddot) / (M*g + M*z_ddot)
```

Where M is mass, g is gravity, and other terms relate to center of mass and angular acceleration.

#### Linear Inverted Pendulum Model (LIPM)
Simplified model for walking:
```
x_com_ddot = ω²(x_com - x_zmp)
```

Where `ω² = g / h` (h is height of center of mass).

### Whole-Body Control

#### Operational Space Control
Control tasks in operational space while maintaining secondary objectives:

```
τ = J^T * F_task + (I - J^T * J^#) * τ_null
```

Where J^# is the damped pseudo-inverse of the Jacobian.

#### Prioritized Task Control
```cpp
class WholeBodyController {
public:
    struct Task {
        std::string name;
        int priority;  // 0 = highest
        std::function<VectorXd(const RobotState&)> error_func;
        std::function<MatrixXd(const RobotState&)> jacobian_func;
        double weight;
    };

    VectorXd computeControl(const RobotState& state,
                           const std::vector<Task>& tasks) {
        // Sort tasks by priority
        std::vector<Task> sorted_tasks = sortTasksByPriority(tasks);

        MatrixXd N = MatrixXd::Identity(state.joints.size(), state.joints.size());  // Null space projector
        VectorXd tau = VectorXd::Zero(state.joints.size());

        for (const auto& task : sorted_tasks) {
            // Compute task Jacobian in current null space
            MatrixXd J_task = task.jacobian_func(state) * N;

            // Compute damped pseudo-inverse
            MatrixXd J_damped = computeDampedPseudoInverse(J_task);

            // Compute required acceleration
            VectorXd error = task.error_func(state);
            VectorXd x_ddot = -error * kp_ - state.joint_velocities * kv_;

            // Compute joint torques for this task
            VectorXd tau_task = J_damped * x_ddot;

            // Add to total control with null space projection
            tau += tau_task;

            // Update null space projector
            MatrixXd I = MatrixXd::Identity(state.joints.size(), state.joints.size());
            N = N * (I - J_damped * J_task);
        }

        return tau;
    }

private:
    double kp_ = 10.0;  // Proportional gain
    double kv_ = 2.0 * sqrt(kp_);  // Derivative gain for critical damping

    MatrixXd computeDampedPseudoInverse(const MatrixXd& J) {
        double damping = 0.01;
        return J.transpose() * (J * J.transpose() + damping * MatrixXd::Identity(J.rows(), J.rows())).inverse();
    }
};
```

## Adaptive and Learning-Based Control

### Adaptive Control

Adaptive controllers adjust their parameters based on system behavior:

#### Model Reference Adaptive Control (MRAC)
```cpp
class MRACController {
public:
    MRACController(const VectorXd& reference_model_params,
                   const VectorXd& initial_controller_params)
        : theta_ref_(reference_model_params),
          theta_ctrl_(initial_controller_params) {}

    double update(const double reference_input, const double actual_output) {
        // Model error
        double error = reference_input - actual_output;

        // Parameter update law (gradient descent)
        VectorXd phi = regressorVector(actual_output);
        theta_ctrl_ += learning_rate_ * error * phi;

        // Compute control output
        return computeControlOutput(reference_input);
    }

private:
    VectorXd theta_ref_;   // Reference model parameters
    VectorXd theta_ctrl_;  // Controller parameters
    double learning_rate_ = 0.01;

    VectorXd regressorVector(double y) {
        // Regressor vector for parameter estimation
        // Implementation depends on system structure
        return VectorXd::Zero(1);  // Placeholder
    }

    double computeControlOutput(double reference) {
        // Compute control based on estimated parameters
        return 0.0;  // Placeholder
    }
};
```

### Reinforcement Learning in Control

#### Policy-Based Methods
- Direct optimization of control policy
- Continuous action spaces
- Sample-efficient for complex behaviors

#### Value-Based Methods
- Q-learning for discrete actions
- Actor-critic for continuous actions
- Deep Q-Networks (DQN) for high-dimensional state spaces

## Implementation Considerations

### Real-time Constraints

#### Control Loop Timing
```cpp
class RealTimeController {
public:
    RealTimeController(double frequency)
        : period_(1.0 / frequency), last_time_(std::chrono::steady_clock::now()) {}

    void runControlLoop() {
        while (running_) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = current_time - last_time_;
            double dt = std::chrono::duration<double>(elapsed).count();

            if (dt >= period_) {
                // Perform control computation
                performControlUpdate();

                // Update timing
                last_time_ = current_time;
            } else {
                // Sleep to maintain timing (or use real-time scheduling)
                std::this_thread::sleep_for(
                    std::chrono::nanoseconds(static_cast<long>((period_ - dt) * 1e9)));
            }
        }
    }

private:
    double period_;
    std::chrono::steady_clock::time_point last_time_;
    bool running_ = true;
};
```

### Safety and Limit Handling

#### Saturation and Constraints
```cpp
class ConstrainedController {
public:
    double applyLimits(double command, double min_val, double max_val) {
        return std::clamp(command, min_val, max_val);
    }

    double antiWindupPID(double error, double max_output) {
        double raw_output = pid_controller_.compute(error);

        // Check if output is saturated
        bool saturated = (std::abs(raw_output) >= max_output);

        if (saturated) {
            // Reduce integral accumulation when saturated
            pid_controller_.reduceIntegralGrowth();
        }

        return applyLimits(raw_output, -max_output, max_output);
    }
};
```

### Filtering and Noise Handling

#### Kalman Filters for Control
```cpp
class KalmanFilterController {
public:
    KalmanFilterController() {
        // Initialize state covariance
        P_ = MatrixXd::Identity(4, 4) * 10.0;  // Initial uncertainty
    }

    double filterMeasurement(double measurement) {
        // Prediction step
        x_pred_ = A_ * x_hat_;  // Predict state
        P_pred_ = A_ * P_ * A_.transpose() + Q_;  // Predict covariance

        // Update step
        MatrixXd K = P_pred_ * H_.transpose() * (H_ * P_pred_ * H_.transpose() + R_).inverse();
        x_hat_ = x_pred_ + K * (measurement - H_ * x_pred_);
        P_ = (MatrixXd::Identity(4, 4) - K * H_) * P_pred_;

        return x_hat_(0);  // Return filtered position
    }

private:
    VectorXd x_hat_ = VectorXd::Zero(4);  // State estimate [pos, vel, acc, jerk]
    MatrixXd P_;  // State covariance
    MatrixXd A_ = MatrixXd::Identity(4, 4);  // State transition
    MatrixXd H_ = MatrixXd::Zero(1, 4);      // Measurement matrix
    MatrixXd Q_ = MatrixXd::Identity(4, 4) * 0.1;  // Process noise
    MatrixXd R_ = MatrixXd::Identity(1, 1) * 1.0;   // Measurement noise

    void initializeMatrices() {
        H_(0, 0) = 1.0;  // Measure position
    }
};
```

## Multi-Rate Control Systems

### Cascade Control

```cpp
class CascadeController {
public:
    CascadeController() : position_ctrl_(1.0, 0.1, 0.01, 0.01),  // Position PID
                         velocity_ctrl_(2.0, 0.2, 0.02, 0.01) {  // Velocity PID
    }

    double computeControl(double desired_pos, double current_pos) {
        // Inner loop: velocity control
        double desired_vel = position_ctrl_.compute(desired_pos - current_pos);
        double actual_vel = differentiatePosition(current_pos);

        // Outer loop: position control
        double control_output = velocity_ctrl_.compute(desired_vel - actual_vel);

        return control_output;
    }

private:
    PIDController position_ctrl_;
    PIDController velocity_ctrl_;

    double differentiatePosition(double pos) {
        // Compute velocity by differentiating position
        static double prev_pos = 0;
        static auto prev_time = std::chrono::steady_clock::now();

        auto current_time = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(current_time - prev_time).count();

        double vel = (pos - prev_pos) / dt;

        prev_pos = pos;
        prev_time = current_time;

        return vel;
    }
};
```

## Control System Design Process

### Step 1: System Modeling
- Identify inputs, outputs, and states
- Develop mathematical model (linearized if needed)
- Validate model against experimental data

### Step 2: Control Objectives
- Define performance requirements
- Identify constraints and limitations
- Consider trade-offs between competing objectives

### Step 3: Controller Selection
- Choose appropriate control strategy
- Consider implementation constraints
- Plan for robustness and safety

### Step 4: Parameter Tuning
- Use analytical methods where possible
- Simulate performance before implementation
- Tune experimentally with safety measures

### Step 5: Validation and Testing
- Test in simulation first
- Gradual deployment with safety protocols
- Monitor performance and adapt as needed

## Case Studies

### Humanoid Walking Control
- ZMP-based controllers for balance
- Preview control for gait planning
- Compliance control for foot contact

### Manipulation Control
- Cartesian impedance control for compliant interaction
- Force control for precise contact tasks
- Multi-task optimization for whole-body manipulation

### Whole-Body Motion Control
- Prioritized task execution
- Dynamic balance maintenance
- Real-time motion generation

## Troubleshooting Common Issues

### Oscillations and Instability
- Check gain scheduling
- Verify sensor noise levels
- Examine sampling rate adequacy

### Steady-State Errors
- Increase integral gain (carefully)
- Add feedforward terms
- Check for unmodeled disturbances

### Actuator Saturation
- Implement anti-windup mechanisms
- Use gain scheduling
- Consider model predictive control

## Integration with ROS2

### Control Frameworks

#### ros2_control
```yaml
# Controller manager configuration
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    joint_trajectory_controller:
      ros__parameters:
        joints:
          - joint1
          - joint2
          - joint3
        command_interfaces:
          - position
        state_interfaces:
          - position
          - velocity
```

#### Custom Controller Implementation
```cpp
#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "hardware_interface/loaned_state_interface.hpp"

namespace my_robot_controllers
{
class CustomPIDController : public controller_interface::ControllerInterface
{
public:
    controller_interface::InterfaceConfiguration command_interface_configuration() const override
    {
        controller_interface::InterfaceConfiguration conf;
        conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
        conf.names.push_back("joint1/position");
        return conf;
    }

    controller_interface::InterfaceConfiguration state_interface_configuration() const override
    {
        controller_interface::InterfaceConfiguration conf;
        conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
        conf.names.push_back("joint1/position");
        conf.names.push_back("joint1/velocity");
        return conf;
    }

    controller_interface::return_type update(
        const rclcpp::Time& time, const rclcpp::Duration& period) override
    {
        // Get current state
        double current_pos = joint_state_pos_.get().get_value();
        double current_vel = joint_state_vel_.get().get_value();

        // Compute control
        double error = desired_position_ - current_pos;
        double command = pid_controller_.compute(error);

        // Apply command
        joint_command_pos_.get().set_value(command);

        return controller_interface::return_type::OK;
    }

private:
    double desired_position_ = 0.0;
    PIDController pid_controller_{10.0, 0.1, 0.01, 0.01};

    // Interfaces
    std::vector<hardware_interface::LoanedCommandInterface> joint_command_pos_;
    std::vector<hardware_interface::LoanedStateInterface> joint_state_pos_;
    std::vector<hardware_interface::LoanedStateInterface> joint_state_vel_;
};
}  // namespace my_robot_controllers
```

## Performance Evaluation

### Metrics

#### Time-Domain Metrics
- Rise time, settling time, overshoot
- Steady-state error
- Integral of absolute error (IAE), integral of squared error (ISE)

#### Frequency-Domain Metrics
- Gain and phase margins
- Bandwidth
- Resonant peak

#### Robustness Metrics
- Sensitivity functions
- Disturbance rejection
- Parameter variation tolerance

### Benchmarking

#### Standard Tests
- Step response tests
- Frequency response analysis
- Disturbance rejection tests

#### Real-world Validation
- Task performance metrics
- Energy efficiency
- Safety and reliability measures

## Conclusion

Control systems are fundamental to achieving stable, responsive, and purposeful behavior in humanoid robots. From classical PID controllers to advanced model predictive and whole-body control approaches, each technique has its place in the roboticist's toolkit.

The challenge in humanoid robotics lies in managing the complex interactions between multiple control objectives (balance, manipulation, locomotion) while maintaining system stability and safety. Modern approaches often combine multiple control strategies hierarchically, with high-level planners generating references for lower-level controllers.

The next chapter will explore robot perception systems, which provide the sensory input necessary for intelligent control decisions.

## Exercises

1. Implement a PID controller for joint position control and tune it using the Ziegler-Nichols method.

2. Research and implement a simple model predictive controller for a point mass system.

3. Design a whole-body controller for a humanoid robot that can maintain balance while tracking a Cartesian trajectory.

4. Implement a cascade controller with position and velocity loops and compare its performance to a single PID controller.

5. Investigate how machine learning techniques can be integrated with classical control methods for humanoid robots.