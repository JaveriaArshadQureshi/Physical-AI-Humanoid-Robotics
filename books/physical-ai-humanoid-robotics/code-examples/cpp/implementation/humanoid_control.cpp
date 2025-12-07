#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <mutex>
#include <algorithm>

/**
 * @brief Joint state structure for humanoid robot
 * Chapter: Building a Humanoid (Actuators, Joints, Hardware Choices)
 */
struct JointState {
    double position;
    double velocity;
    double effort;
    double desired_position;

    JointState() : position(0.0), velocity(0.0), effort(0.0), desired_position(0.0) {}
};

class HumanoidController {
public:
    HumanoidController(const std::vector<std::string>& joint_names)
        : joint_names_(joint_names), num_joints_(joint_names.size()), is_running_(false) {

        joint_states_.resize(num_joints_);

        // Initialize PID gains
        kp_.resize(num_joints_, 100.0);
        ki_.resize(num_joints_, 0.1);
        kd_.resize(num_joints_, 10.0);

        integral_errors_.resize(num_joints_, 0.0);
        previous_errors_.resize(num_joints_, 0.0);
    }

    void setJointPositions(const std::vector<double>& positions) {
        if (positions.size() != num_joints_) {
            throw std::invalid_argument("Incorrect number of joint positions");
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        for (size_t i = 0; i < num_joints_; ++i) {
            joint_states_[i].desired_position = positions[i];
        }
    }

    std::vector<double> getJointPositions() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        std::vector<double> positions(num_joints_);
        for (size_t i = 0; i < num_joints_; ++i) {
            positions[i] = joint_states_[i].position;
        }
        return positions;
    }

    void start() {
        if (!is_running_) {
            is_running_ = true;
            control_thread_ = std::thread(&HumanoidController::controlLoop, this);
        }
    }

    void stop() {
        is_running_ = false;
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
    }

private:
    void controlLoop() {
        const double control_rate = 100.0; // Hz
        const std::chrono::duration<double> dt(1.0 / control_rate);

        while (is_running_) {
            auto start_time = std::chrono::high_resolution_clock::now();

            computeControlEfforts();

            // Simulate applying efforts to joints
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (size_t i = 0; i < num_joints_; ++i) {
                    joint_states_[i].effort = control_efforts_[i];

                    // Simple simulation: update position based on effort
                    joint_states_[i].velocity += joint_states_[i].effort * dt.count();
                    joint_states_[i].position += joint_states_[i].velocity * dt.count();
                }
            }

            // Sleep to maintain control rate
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            auto sleep_time = dt - elapsed;
            if (sleep_time.count() > 0) {
                std::this_thread::sleep_for(sleep_time);
            }
        }
    }

    void computeControlEfforts() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        control_efforts_.resize(num_joints_);

        for (size_t i = 0; i < num_joints_; ++i) {
            double current_pos = joint_states_[i].position;
            double desired_pos = joint_states_[i].desired_position;

            // Calculate error
            double error = desired_pos - current_pos;

            // Update integral and derivative terms
            integral_errors_[i] += error / (1.0 / (dt_.count()));
            double derivative = (error - previous_errors_[i]) / dt_.count();

            // Apply PID formula
            control_efforts_[i] = kp_[i] * error +
                                 ki_[i] * integral_errors_[i] +
                                 kd_[i] * derivative;

            // Store current error for next derivative calculation
            previous_errors_[i] = error;
        }
    }

    std::vector<std::string> joint_names_;
    size_t num_joints_;
    std::vector<JointState> joint_states_;
    std::vector<double> kp_, ki_, kd_;
    std::vector<double> integral_errors_, previous_errors_;
    std::vector<double> control_efforts_;
    std::chrono::duration<double> dt_ = std::chrono::duration<double>(0.01);

    std::thread control_thread_;
    std::atomic<bool> is_running_;
    std::mutex state_mutex_;
};

int main() {
    // Create a controller for a simplified humanoid (6 joints for one leg)
    std::vector<std::string> joint_names = {
        "left_hip_yaw", "left_hip_pitch", "left_hip_roll",
        "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll"
    };

    HumanoidController controller(joint_names);

    std::cout << "Starting humanoid controller..." << std::endl;
    controller.start();

    try {
        // Generate a simple walking trajectory
        const double duration = 5.0;
        const double dt = 0.01;
        const int steps = static_cast<int>(duration / dt);

        std::cout << "Executing walking trajectory for " << duration << " seconds" << std::endl;

        for (int i = 0; i < steps && controller.is_running(); ++i) {
            double t = i * dt;

            // Generate a simple periodic trajectory
            std::vector<double> joint_positions = {
                0.0,                                    // Hip yaw
                0.2 * sin(2 * M_PI * t / 2.0),         // Hip pitch (walking motion)
                0.1 * sin(2 * M_PI * t / 2.0),         // Hip roll
                0.3 * sin(2 * M_PI * t / 2.0),         // Knee pitch
                0.1 * cos(2 * M_PI * t / 2.0),         // Ankle pitch
                0.05 * sin(2 * M_PI * t / 2.0)         // Ankle roll
            };

            controller.setJointPositions(joint_positions);

            if (i % 50 == 0) {  // Print every 50 steps
                auto current_positions = controller.getJointPositions();
                std::cout << "Step " << i << ": Desired hip pitch="
                         << joint_positions[1] << ", Actual hip pitch="
                         << current_positions[1] << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::cout << "Trajectory execution completed" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    controller.stop();
    std::cout << "Controller stopped" << std::endl;

    return 0;
}