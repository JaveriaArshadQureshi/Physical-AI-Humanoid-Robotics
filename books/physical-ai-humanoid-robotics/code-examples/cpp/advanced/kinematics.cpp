#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief Simple 2D robot arm kinematics example
 * Chapter: Kinematics & Dynamics (FK, IK, Trajectory Planning)
 */
class RobotArm {
public:
    RobotArm(std::vector<double> link_lengths) : link_lengths_(link_lengths) {
        num_joints_ = link_lengths.size();
    }

    /**
     * @brief Forward kinematics: compute end-effector position from joint angles
     * @param joint_angles Vector of joint angles in radians
     * @return End-effector position [x, y]
     */
    std::vector<double> forward_kinematics(const std::vector<double>& joint_angles) {
        double x = 0.0, y = 0.0;
        double cumulative_angle = 0.0;

        for (size_t i = 0; i < num_joints_; ++i) {
            cumulative_angle += joint_angles[i];
            x += link_lengths_[i] * cos(cumulative_angle);
            y += link_lengths_[i] * sin(cumulative_angle);
        }

        return {x, y};
    }

    /**
     * @brief Simple inverse kinematics for 2-DOF planar arm
     * @param target_x X coordinate of target position
     * @param target_y Y coordinate of target position
     * @return Joint angles [theta1, theta2] or empty vector if no solution
     */
    std::vector<double> inverse_kinematics(double target_x, double target_y) {
        if (link_lengths_.size() != 2) {
            std::cout << "This IK solver only works for 2-DOF arms" << std::endl;
            return {};
        }

        double l1 = link_lengths_[0];
        double l2 = link_lengths_[1];

        // Distance from origin to target
        double r = sqrt(target_x * target_x + target_y * target_y);

        // Check if target is reachable
        if (r > l1 + l2) {
            std::cout << "Target is out of reach" << std::endl;
            return {};
        }
        if (r < abs(l1 - l2)) {
            std::cout << "Target is inside workspace" << std::endl;
            return {};
        }

        // Compute second joint angle
        double cos_theta2 = (l1*l1 + l2*l2 - r*r) / (2*l1*l2);
        double theta2 = acos(std::max(-1.0, std::min(1.0, cos_theta2)));

        // Compute first joint angle
        double k1 = l1 + l2 * cos_theta2;
        double k2 = l2 * sin(theta2);
        double theta1 = atan2(target_y, target_x) - atan2(k2, k1);

        return {theta1, theta2};
    }

private:
    std::vector<double> link_lengths_;
    size_t num_joints_;
};

int main() {
    // Create a 2-DOF robot arm with link lengths [1.0, 1.0]
    RobotArm arm({1.0, 1.0});

    // Example: Forward kinematics
    std::vector<double> joint_angles = {M_PI/4, M_PI/4}; // 45 degrees each
    auto ee_pos = arm.forward_kinematics(joint_angles);
    std::cout << "Forward Kinematics:" << std::endl;
    std::cout << "Joint angles: [" << joint_angles[0] << ", " << joint_angles[1] << "]" << std::endl;
    std::cout << "End-effector position: [" << ee_pos[0] << ", " << ee_pos[1] << "]" << std::endl;

    // Example: Inverse kinematics
    double target_x = 1.0, target_y = 1.0;
    auto ik_solution = arm.inverse_kinematics(target_x, target_y);

    if (!ik_solution.empty()) {
        std::cout << "\nInverse Kinematics:" << std::endl;
        std::cout << "Target position: [" << target_x << ", " << target_y << "]" << std::endl;
        std::cout << "Joint angles: [" << ik_solution[0] << ", " << ik_solution[1] << "]" << std::endl;

        // Verify with forward kinematics
        auto verification = arm.forward_kinematics(ik_solution);
        std::cout << "Verification (should be close to target): ["
                  << verification[0] << ", " << verification[1] << "]" << std::endl;
    }

    return 0;
}