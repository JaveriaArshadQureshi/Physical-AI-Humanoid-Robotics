---
title: "Chapter 13: Building a Humanoid (Actuators, Joints, Hardware Choices)"
description: "Designing and constructing the physical components of humanoid robots"
---

# Chapter 13: Building a Humanoid (Actuators, Joints, Hardware Choices)

## Overview

Building a humanoid robot requires careful consideration of mechanical design, actuator selection, and hardware integration. This chapter explores the engineering challenges and solutions involved in creating physical humanoid robots, from actuator technologies to joint designs, and provides guidance on hardware selection for different applications and budgets.

## Humanoid Robot Design Principles

### Anthropomorphic Design Considerations

#### Size and Proportions
- **Height**: Typically ranges from 1m (small research platforms) to 1.8m (full-size humanoids)
- **Weight**: Critical for mobility and battery life (20-150kg depending on size)
- **Degrees of Freedom**: 20-30+ joints for human-like mobility
- **Center of Mass**: Critical for balance and stability

#### Degrees of Freedom Distribution
```
Lower Body: 6-8 DOF per leg (hip: 3 DOF, knee: 1 DOF, ankle: 2-3 DOF)
Upper Body: 6-7 DOF per arm (shoulder: 3 DOF, elbow: 1 DOF, wrist: 2-3 DOF)
Torso: 2-6 DOF (waist, chest)
Head: 2-3 DOF (neck, eyes)
```

### Structural Design Requirements

#### Load Considerations
- Static loads (robot weight distribution)
- Dynamic loads (acceleration forces during motion)
- Impact loads (falling, collisions)
- Payload loads (carrying objects)

#### Material Selection
- **Aluminum**: Good strength-to-weight ratio, machinable
- **Carbon Fiber**: Excellent stiffness-to-weight ratio, expensive
- **Titanium**: High strength, corrosion resistance, expensive
- **Engineering Plastics**: For non-critical components, lightweight

```cpp
#include <vector>
#include <string>

struct JointSpec {
    std::string name;
    std::string type;  // "rotary", "linear", "spherical"
    double range_min;
    double range_max;
    double max_velocity;
    double max_torque;
    double gear_ratio;
    std::string actuator_type;  // "servo", "brushless", "pneumatic", etc.
};

struct LinkSpec {
    std::string name;
    double mass;
    double length;
    double diameter;
    std::string material;
    double com_x, com_y, com_z;  // Center of mass
    double inertia_xx, inertia_yy, inertia_zz;  // Moments of inertia
};

class HumanoidSpecifications {
public:
    std::vector<JointSpec> joints;
    std::vector<LinkSpec> links;

    double total_height;
    double total_weight;
    double max_payload;
    double battery_life;

    HumanoidSpecifications() {
        // Typical specifications for a medium-sized humanoid
        total_height = 1.5;  // meters
        total_weight = 60.0;  // kg
        max_payload = 5.0;   // kg
        battery_life = 2.0;  // hours
    }
};
```

## Actuator Technologies

### Servo Motors

#### Advantages
- Integrated control electronics
- Precise position control
- Wide availability
- Cost-effective for small robots

#### Disadvantages
- Limited torque for size
- Heat generation during continuous operation
- Resolution limits in position feedback

#### Popular Servo Types

##### Hobby RC Servos
```cpp
class RCServo {
public:
    RCServo(int pin, double min_pulse=500, double max_pulse=2500)
        : pin_(pin), min_pulse_us_(min_pulse), max_pulse_us_(max_pulse) {
        // Initialize PWM control
    }

    void setPosition(double angle_degrees) {
        // Convert angle to pulse width
        double pulse_width = min_pulse_us_ +
                            (angle_degrees / 180.0) * (max_pulse_us_ - min_pulse_us_);

        // Send PWM signal
        setPWMSignal(pin_, pulse_width);
    }

    double getPosition() {
        // Read current position from potentiometer or encoder
        return readAnalogPin(pin_);
    }

private:
    int pin_;
    double min_pulse_us_;
    double max_pulse_us_;
};
```

##### Smart Servos (Dynamixel, Herkulex)
- Digital position feedback
- Higher torque ratings
- Communication protocols (RS485, CAN)
- Built-in protection features

### Brushless DC Motors

#### Advantages
- High power density
- Efficient operation
- Long lifespan (no brushes)
- Precise control with proper drivers

#### Disadvantages
- Requires complex control electronics
- Higher cost
- Need for encoders for position feedback

```cpp
class BLDCMotor {
public:
    BLDCMotor(int pole_pairs, double torque_constant, double max_current)
        : pole_pairs_(pole_pairs), kt_(torque_constant), max_current_(max_current) {}

    void setCurrent(double current) {
        // Limit current to maximum
        current = std::min(current, max_current_);

        // Apply current control (FOC - Field Oriented Control)
        applyFieldOrientedControl(current);
    }

    void setVelocity(double velocity_rpm) {
        // Implement velocity control loop
        double target_current = velocity_to_current(velocity_rpm);
        setCurrent(target_current);
    }

    void setPosition(double position_radians) {
        // Implement position control loop
        double current_error = position_radians - getCurrentPosition();
        double velocity_command = position_pid_.compute(current_error);
        setVelocity(velocity_command * 60.0 / (2 * M_PI));  // Convert to RPM
    }

    double getTorque() {
        return current_ * kt_;
    }

private:
    int pole_pairs_;
    double kt_;  // Torque constant (Nm/A)
    double max_current_;
    double current_ = 0.0;

    PIDController position_pid_{10.0, 0.1, 0.01, 0.001};  // kp, ki, kd, dt
    PIDController velocity_pid_{5.0, 0.05, 0.005, 0.001};

    void applyFieldOrientedControl(double current) {
        // Simplified FOC implementation
        // In practice, this would involve complex trigonometric calculations
        // and phase control
    }

    double velocity_to_current(double velocity_rpm) {
        // Convert velocity error to current demand
        // This is simplified - actual implementation involves motor dynamics
        return velocity_rpm * 0.1;  // Simplified conversion
    }

    double getCurrentPosition() {
        // Read position from encoder
        return 0.0;  // Placeholder
    }
};
```

### Series Elastic Actuators (SEA)

#### Concept
Series Elastic Actuators include a spring in series with the motor, providing:
- Back-drivability
- Force control capability
- Shock absorption
- Safe human interaction

```cpp
class SeriesElasticActuator {
public:
    SeriesElasticActuator(double spring_constant, double motor_gear_ratio)
        : k_spring_(spring_constant), gear_ratio_(motor_gear_ratio) {
        // Initialize motor and encoder
        motor_ = std::make_unique<BLDCMotor>(7, 0.15, 10.0);  // Example motor
    }

    void setForce(double desired_force) {
        // Convert force to spring deflection
        double spring_deflection = desired_force / k_spring_;

        // Calculate motor position needed to achieve deflection
        double motor_position = joint_position_ + (spring_deflection * gear_ratio_);

        // Command motor to position
        motor_->setPosition(motor_position);
    }

    void setJointPosition(double desired_position) {
        // Calculate motor position to achieve joint position with spring deflection
        double spring_deflection = joint_load_ / k_spring_;
        double motor_position = desired_position + (spring_deflection * gear_ratio_);

        motor_->setPosition(motor_position);
    }

    double getForce() {
        // Measure spring deflection to calculate force
        double current_motor_pos = motor_->getPosition();
        double current_joint_pos = getJointPosition();
        double spring_deflection = (current_motor_pos - current_joint_pos) / gear_ratio_;

        return spring_deflection * k_spring_;
    }

    double getJointPosition() {
        // Read joint encoder directly
        return readJointEncoder();
    }

private:
    double k_spring_;      // Spring constant (N/m or Nm/rad)
    double gear_ratio_;    // Motor to joint gear ratio
    double joint_position_ = 0.0;
    double joint_load_ = 0.0;

    std::unique_ptr<BLDCMotor> motor_;
    std::unique_ptr<Encoder> joint_encoder_;
    std::unique_ptr<Encoder> motor_encoder_;

    double readJointEncoder() {
        // Read joint position from encoder
        return 0.0;  // Placeholder
    }
};
```

### Pneumatic Actuators

#### Advantages
- High power-to-weight ratio
- Compliance and shock absorption
- Clean operation (no electrical hazards)
- High force capability

#### Disadvantages
- Compressor requirements
- Less precise control
- Air compressibility affects response
- Noise from compressor

```cpp
class PneumaticActuator {
public:
    PneumaticActuator(double cylinder_area, double max_pressure)
        : area_(cylinder_area), max_pressure_(max_pressure) {}

    void setPosition(double position) {
        // Use proportional valves for position control
        // This is a simplified model
        double current_position = getPosition();
        double error = position - current_position;

        // Control valve opening based on error
        double valve_opening = position_pid_.compute(error);
        setValveOpening(valve_opening);
    }

    void setForce(double force) {
        // Calculate required pressure
        double pressure = force / area_;
        pressure = std::min(pressure, max_pressure_);

        setPressure(pressure);
    }

    double getMaxForce() {
        return area_ * max_pressure_;
    }

private:
    double area_;           // Cylinder area (m²)
    double max_pressure_;   // Maximum pressure (Pa)

    PIDController position_pid_{2.0, 0.01, 0.05, 0.001};

    void setValveOpening(double opening) {
        // Control proportional valve
    }

    void setPressure(double pressure) {
        // Set pressure regulator
    }

    double getPosition() {
        // Read position from linear encoder
        return 0.0;  // Placeholder
    }
};
```

## Joint Design and Mechanics

### Joint Types and Configurations

#### Revolute Joints
- Most common in humanoid robots
- Allow rotation around single axis
- Can be implemented with various actuator types

#### Spherical Joints
- Multi-axis rotation capability
- Used for hip and shoulder joints
- More complex mechanical design

#### Prismatic Joints
- Linear motion
- Less common in humanoids but used in some designs
- High precision applications

### Gearbox Selection

#### Harmonic Drive Gears
- High reduction ratios (50:1 to 320:1)
- Compact size
- High precision
- Expensive

#### Planetary Gears
- Moderate reduction ratios
- Good efficiency
- Cost-effective
- Moderate precision

#### Cycloidal Gears
- High reduction ratios
- High torque capacity
- Good backlash characteristics
- Medium cost

```cpp
class Gearbox {
public:
    enum Type { HARMONIC_DRIVE, PLANETARY, CYCLOIDAL };

    Gearbox(Type type, double reduction_ratio, double efficiency = 0.85)
        : type_(type), reduction_(reduction_ratio), efficiency_(efficiency) {}

    double getOutputTorque(double input_torque) {
        return input_torque * reduction_ * efficiency_;
    }

    double getOutputSpeed(double input_speed) {
        return input_speed / reduction_;
    }

    double getBacklash() {
        switch (type_) {
            case HARMONIC_DRIVE: return 0.001;  // 0.1 degrees
            case PLANETARY: return 0.01;        // 1 degree
            case CYCLOIDAL: return 0.005;       // 0.5 degrees
        }
        return 0.01;  // Default
    }

private:
    Type type_;
    double reduction_;
    double efficiency_;
};
```

### Joint Assembly Design

#### Key Components
```cpp
class JointAssembly {
public:
    JointAssembly(std::unique_ptr<Actuator> actuator,
                  std::unique_ptr<Gearbox> gearbox,
                  std::unique_ptr<Encoder> encoder)
        : actuator_(std::move(actuator)),
          gearbox_(std::move(gearbox)),
          encoder_(std::move(encoder)) {
        // Initialize joint parameters
        max_torque_ = actuator_->getMaxTorque() * gearbox_->getOutputTorque(1.0);
    }

    void setPosition(double position_rad) {
        // Calculate required motor position considering gear ratio
        double motor_position = position_rad * gearbox_->getReductionRatio();
        actuator_->setPosition(motor_position);
    }

    double getPosition() {
        double motor_pos = actuator_->getPosition();
        return motor_pos / gearbox_->getReductionRatio();
    }

    void setTorque(double torque_nm) {
        // Calculate required motor torque considering gearbox efficiency
        double motor_torque = torque_nm / (gearbox_->getReductionRatio() * gearbox_->getEfficiency());
        actuator_->setTorque(motor_torque);
    }

    double getTorque() {
        double motor_torque = actuator_->getTorque();
        return motor_torque * gearbox_->getReductionRatio() * gearbox_->getEfficiency();
    }

    double getVelocity() {
        double motor_vel = actuator_->getVelocity();
        return motor_vel / gearbox_->getReductionRatio();
    }

private:
    std::unique_ptr<Actuator> actuator_;
    std::unique_ptr<Gearbox> gearbox_;
    std::unique_ptr<Encoder> encoder_;

    double max_torque_;
    double max_velocity_;
};
```

## Sensing and Feedback Systems

### Position Sensing

#### Absolute Encoders
- Know position immediately at startup
- No need for homing routine
- Higher cost
- Critical for safety systems

#### Incremental Encoders
- Measure position changes from reference point
- Lower cost
- Require homing procedure
- High resolution available

```cpp
class Encoder {
public:
    enum Type { ABSOLUTE, INCREMENTAL };

    Encoder(Type type, int resolution_bits)
        : type_(type), resolution_bits_(resolution_bits), counts_per_rev_(1 << resolution_bits) {}

    virtual double getPosition() = 0;
    virtual double getVelocity() = 0;

    double getResolution() {
        return 2 * M_PI / counts_per_rev_;
    }

protected:
    Type type_;
    int resolution_bits_;
    int counts_per_rev_;
};

class MagneticEncoder : public Encoder {
public:
    MagneticEncoder(int resolution_bits) : Encoder(ABSOLUTE, resolution_bits) {}

    double getPosition() override {
        // Read from magnetic sensor (e.g., AS5048B)
        uint16_t raw_count = readMagneticSensor();
        return (double)raw_count / counts_per_rev_ * 2 * M_PI;
    }

    double getVelocity() override {
        // Calculate from position changes
        static double prev_pos = 0;
        static auto prev_time = std::chrono::high_resolution_clock::now();

        auto curr_time = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(curr_time - prev_time).count();

        double curr_pos = getPosition();
        double velocity = (curr_pos - prev_pos) / dt;

        prev_pos = curr_pos;
        prev_time = curr_time;

        return velocity;
    }

private:
    uint16_t readMagneticSensor() {
        // Interface with magnetic encoder IC
        // This is platform-specific
        return 0;  // Placeholder
    }
};
```

### Force/Torque Sensing

#### Six-Axis Force/Torque Sensors
- Measure forces (Fx, Fy, Fz) and torques (Tx, Ty, Tz)
- Critical for manipulation and balance
- Expensive but essential for advanced behaviors

#### Strain Gauge Sensors
- Measure deformation to infer forces
- Can be integrated into structure
- Require calibration

```cpp
class ForceTorqueSensor {
public:
    struct Wrench {
        double fx, fy, fz;  // Forces in Newtons
        double tx, ty, tz;  // Torques in Newton-meters
    };

    ForceTorqueSensor() {
        // Initialize calibration matrix
        initializeCalibration();
    }

    Wrench getWrench() {
        // Read raw strain gauge values
        std::array<int, 6> raw_readings = readStrainGauges();

        // Apply calibration matrix
        Wrench wrench;
        for (int i = 0; i < 6; i++) {
            double calibrated_value = 0;
            for (int j = 0; j < 6; j++) {
                calibrated_value += calibration_matrix_[i][j] * raw_readings[j];
            }

            switch (i) {
                case 0: wrench.fx = calibrated_value; break;
                case 1: wrench.fy = calibrated_value; break;
                case 2: wrench.fz = calibrated_value; break;
                case 3: wrench.tx = calibrated_value; break;
                case 4: wrench.ty = calibrated_value; break;
                case 5: wrench.tz = calibrated_value; break;
            }
        }

        return wrench;
    }

    void calibrate() {
        // Perform calibration procedure
        // Apply known forces and torques, record readings
    }

private:
    std::array<std::array<double, 6>, 6> calibration_matrix_;

    std::array<int, 6> readStrainGauges() {
        // Read from 6 strain gauges
        return std::array<int, 6>{0, 0, 0, 0, 0, 0};  // Placeholder
    }

    void initializeCalibration() {
        // Load pre-computed calibration matrix
        // In practice, this would come from calibration procedure
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                calibration_matrix_[i][j] = (i == j) ? 1.0 : 0.0;  // Identity matrix placeholder
            }
        }
    }
};
```

## Power Systems

### Battery Selection

#### Lithium Polymer (LiPo) Batteries
- High energy density
- Good discharge rates
- Require careful management
- Fire hazard if damaged

#### Lithium Iron Phosphate (LiFePO4)
- Safer chemistry
- Longer cycle life
- Lower energy density
- More stable voltage

#### Nickel Metal Hydride (NiMH)
- Proven technology
- Moderate energy density
- Safe operation
- Heavier than lithium options

```cpp
class BatterySystem {
public:
    enum Chemistry { LITHIUM_POLYMER, LIFEP04, NIMH };

    BatterySystem(Chemistry chemistry, double capacity_Ah, double voltage)
        : chemistry_(chemistry), capacity_Ah_(capacity_Ah), voltage_(voltage) {
        current_soc_ = 1.0;  // Start fully charged
    }

    double getRemainingCapacity() {
        return capacity_Ah_ * current_soc_;
    }

    double getRuntimeEstimate(double current_draw_A) {
        return (capacity_Ah_ * current_soc_) / current_draw_A;  // Hours
    }

    double getVoltage() {
        // Voltage varies with SOC and load
        return voltage_ * getSOCVoltageFactor() - getLoadDrop(current_draw_);
    }

    double getSOC() {  // State of Charge
        return current_soc_;
    }

    void updateSOC(double current_draw_A, double dt_hours) {
        double charge_change = current_draw_A * dt_hours / capacity_Ah_;
        current_soc_ = std::max(0.0, std::min(1.0, current_soc_ - charge_change));
    }

private:
    Chemistry chemistry_;
    double capacity_Ah_;
    double voltage_;
    double current_soc_;

    double current_draw_ = 0.0;

    double getSOCVoltageFactor() {
        // Simplified SOC-voltage curve
        if (current_soc_ > 0.8) return 1.0;
        if (current_soc_ > 0.2) return 0.95;
        return 0.9;
    }

    double getLoadDrop(double current) {
        // Voltage drop due to internal resistance
        double internal_resistance = getInternalResistance();
        return current * internal_resistance;
    }

    double getInternalResistance() {
        switch (chemistry_) {
            case LITHIUM_POLYMER: return 0.05;  // Ohms
            case LIFEP04: return 0.1;
            case NIMH: return 0.2;
        }
        return 0.1;
    }
};
```

### Power Distribution

```cpp
class PowerDistribution {
public:
    PowerDistribution() {
        // Initialize power rails
        power_rails_["5V"] = PowerRail(5.0, 10.0);    // 10A available
        power_rails_["12V"] = PowerRail(12.0, 5.0);   // 5A available
        power_rails_["24V"] = PowerRail(24.0, 3.0);   // 3A available
    }

    bool requestPower(const std::string& rail_name, double current_A) {
        auto it = power_rails_.find(rail_name);
        if (it != power_rails_.end()) {
            return it->second.allocateCurrent(current_A);
        }
        return false;
    }

    void releasePower(const std::string& rail_name, double current_A) {
        auto it = power_rails_.find(rail_name);
        if (it != power_rails_.end()) {
            it->second.releaseCurrent(current_A);
        }
    }

    double getAvailableCurrent(const std::string& rail_name) {
        auto it = power_rails_.find(rail_name);
        if (it != power_rails_.end()) {
            return it->second.getAvailableCurrent();
        }
        return 0.0;
    }

private:
    struct PowerRail {
        double voltage;
        double max_current;
        double allocated_current;

        PowerRail(double v, double max_i) : voltage(v), max_current(max_i), allocated_current(0) {}

        bool allocateCurrent(double current) {
            if (allocated_current + current <= max_current) {
                allocated_current += current;
                return true;
            }
            return false;
        }

        void releaseCurrent(double current) {
            allocated_current = std::max(0.0, allocated_current - current);
        }

        double getAvailableCurrent() {
            return max_current - allocated_current;
        }
    };

    std::map<std::string, PowerRail> power_rails_;
};
```

## Control Electronics

### Motor Controllers

#### Brushless Motor Controllers
```cpp
class BLDCController {
public:
    BLDCController(int poles, double max_current, double switching_freq)
        : poles_(poles), max_current_(max_current), switching_freq_(switching_freq) {
        initializePWM();
    }

    void setPhaseVoltages(double va, double vb, double vc) {
        // Convert phase voltages to PWM duty cycles
        double dc_a = voltageToDutyCycle(va);
        double dc_b = voltageToDutyCycle(vb);
        double dc_c = voltageToDutyCycle(vc);

        setPWMDuties(dc_a, dc_b, dc_c);
    }

    void setTorque(double torque) {
        // Field-oriented control implementation
        double iq_ref = torqueToCurrent(torque);
        performFOC(iq_ref, 0.0);  // id = 0 control
    }

    void setVelocity(double velocity_electrical) {
        // Implement velocity control loop
        double torque_cmd = velocity_loop_.compute(velocity_electrical - getElectricalVelocity());
        setTorque(torque_cmd);
    }

private:
    int poles_;
    double max_current_;
    double switching_freq_;

    PIDController velocity_loop_{1.0, 0.1, 0.01, 1.0/switching_freq_};

    double voltageToDutyCycle(double voltage) {
        // Convert voltage to duty cycle (0-1)
        return (voltage / bus_voltage_ + 1.0) / 2.0;
    }

    double torqueToCurrent(double torque) {
        // Simple torque-current relationship: T = k * I
        return torque / torque_constant_;
    }

    void performFOC(double iq_ref, double id_ref) {
        // Simplified FOC implementation
        // In practice, this involves Park/Clarke transforms
        // and current regulation
    }

    double getElectricalVelocity() {
        // Convert mechanical velocity to electrical
        return getMechanicalVelocity() * poles_ / 2;
    }

    void initializePWM() {
        // Setup 3-phase PWM generation
    }

    void setPWMDuties(double da, double db, double dc) {
        // Set PWM duty cycles for 3 phases
    }

    double bus_voltage_ = 24.0;  // V
    double torque_constant_ = 0.1;  // Nm/A
};
```

### Microcontroller Selection

#### Processing Requirements
- Real-time control (1-10 kHz loop rates)
- Multiple sensor interfaces
- Communication protocols (CAN, SPI, I2C)
- Sufficient RAM for control algorithms

#### Popular Platforms
- **Arduino Mega**: Simple projects, limited performance
- **Raspberry Pi**: Good for high-level control, not real-time
- **BeagleBone Black**: Real-time capable with PRU
- **Custom STM32/FPGA boards**: High performance, complex

## Safety Systems

### Emergency Stop Systems

```cpp
class SafetySystem {
public:
    SafetySystem() {
        // Initialize safety circuits
        initializeEmergencyStops();
        initializeTorqueLimits();
        initializeCollisionDetection();
    }

    void checkSafety() {
        // Check all safety conditions
        if (checkEmergencyStop()) {
            emergencyStop();
            return;
        }

        if (checkTorqueLimits()) {
            reduceTorque();
        }

        if (checkCollision()) {
            stopMotion();
        }

        if (checkTemperature()) {
            thermalProtection();
        }
    }

    void registerEmergencyStop(int pin) {
        emergency_stops_.push_back(pin);
    }

    void enableSafety(bool enable) {
        safety_enabled_ = enable;
    }

private:
    std::vector<int> emergency_stops_;
    bool safety_enabled_ = true;

    bool checkEmergencyStop() {
        for (int pin : emergency_stops_) {
            if (digitalRead(pin) == LOW) {  // Active low emergency stop
                return true;
            }
        }
        return false;
    }

    void emergencyStop() {
        // Immediately cut power to all motors
        for (auto& controller : motor_controllers_) {
            controller.emergencyStop();
        }

        // Set all joints to passive (no holding torque)
        setPassiveMode();

        // Log incident
        logSafetyEvent("EMERGENCY_STOP_ACTIVATED");
    }

    void checkTorqueLimits() {
        // Monitor joint torques
        for (size_t i = 0; i < joint_torques_.size(); i++) {
            if (std::abs(joint_torques_[i]) > torque_limits_[i]) {
                return true;
            }
        }
        return false;
    }

    void reduceTorque() {
        // Reduce torque commands gradually
        torque_reduction_factor_ *= 0.95;  // Reduce by 5%
        if (torque_reduction_factor_ < 0.1) {
            emergencyStop();  // Too much reduction needed
        }
    }

    void stopMotion() {
        // Command zero velocity to all joints
        for (auto& controller : motor_controllers_) {
            controller.setVelocity(0.0);
        }
    }

    bool checkCollision() {
        // Check force/torque sensors for collision
        for (const auto& sensor : force_torque_sensors_) {
            auto wrench = sensor.getWrench();
            if (std::abs(wrench.fz) > collision_threshold_) {
                return true;
            }
        }
        return false;
    }

    bool checkTemperature() {
        // Check motor and electronics temperatures
        for (const auto& temp : temperatures_) {
            if (temp > temperature_limit_) {
                return true;
            }
        }
        return false;
    }

    void thermalProtection() {
        // Reduce power to prevent overheating
        for (auto& controller : motor_controllers_) {
            controller.reducePower(0.8);  // 80% power
        }
    }

    std::vector<BLDCController> motor_controllers_;
    std::vector<ForceTorqueSensor> force_torque_sensors_;
    std::vector<double> joint_torques_;
    std::vector<double> torque_limits_;
    std::vector<double> temperatures_;

    double collision_threshold_ = 50.0;  // N
    double temperature_limit_ = 80.0;   // Celsius
    double torque_reduction_factor_ = 1.0;
};
```

## Hardware Integration and Assembly

### CAD Design Considerations

#### Tolerance Analysis
- Account for manufacturing tolerances
- Consider thermal expansion
- Allow for assembly and maintenance access

#### Stress Analysis
- Use FEA to verify structural integrity
- Check fatigue life for cyclic loads
- Optimize for weight while maintaining strength

### Assembly Procedures

#### Pre-Assembly Checks
- Verify all components match specifications
- Check motor and encoder calibration
- Test individual joint functionality

#### Integration Testing
- Test communication between components
- Verify safety systems function
- Characterize actual performance vs. design

## Cost Analysis and Budgeting

### Typical Cost Breakdown for Humanoid Robot

#### High-Performance Humanoid (e.g., Atlas-level)
- Actuators and gearboxes: 40-50%
- Structural components: 15-20%
- Electronics: 15-20%
- Sensors: 10-15%
- Miscellaneous: 5-10%

#### Research-Grade Humanoid
- Actuators and gearboxes: 30-40%
- Structural components: 20-25%
- Electronics: 20-25%
- Sensors: 10-15%
- Miscellaneous: 5-10%

### Cost Optimization Strategies

#### Volume Production Benefits
- Actuator costs decrease significantly with quantity
- PCB fabrication becomes economical
- Assembly automation possibilities

#### Design for Manufacturing
- Standard components where possible
- Modular design for easier assembly
- Maintenance-friendly design

## Real-World Examples

### Boston Dynamics Atlas
- Hydraulic actuators for high power density
- Sophisticated safety and control systems
- Advanced perception capabilities

### Honda ASIMO
- Electric servo actuators
- Focus on human interaction
- Long-term development approach

### SoftBank Pepper/Nao
- Consumer-focused design
- Cost-effective manufacturing
- Educational and service applications

## Troubleshooting Common Hardware Issues

### Actuator Problems
- Overheating: Check thermal management and duty cycles
- Position drift: Verify encoder calibration and mechanical play
- Noise/vibration: Check mechanical alignment and control parameters

### Electrical Issues
- Communication failures: Check wiring, termination, and protocols
- Power fluctuations: Verify power supply capacity and regulation
- Ground loops: Ensure proper grounding architecture

### Mechanical Issues
- Joint binding: Check alignment, lubrication, and bearing wear
- Resonance: Verify structural stiffness and control parameters
- Wear: Schedule maintenance and replacement of consumables

## Future Trends

### Emerging Technologies

#### Artificial Muscles
- Electroactive polymers
- Shape memory alloys
- Pneumatic artificial muscles

#### Advanced Materials
- Carbon nanotube composites
- Metamaterials for specific properties
- Self-healing materials

#### Integrated Systems
- Actuators with built-in sensing
- Distributed control architectures
- Wireless power and communication

## ROS2 Integration

### Hardware Abstraction Layer

```cpp
#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/hardware_interface.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>

class HumanoidHardwareInterface : public hardware_interface::SystemInterface {
public:
    hardware_interface::CallbackReturn on_init(
        const hardware_interface::HardwareInfo & info) override {

        if (configure_default(info) != CallbackReturn::SUCCESS) {
            return CallbackReturn::ERROR;
        }

        // Initialize hardware components
        for (const auto & joint : info_.joints) {
            // Create joint interfaces based on joint type
            if (joint.command_interfaces.size() == 1 &&
                joint.command_interfaces[0].name == "position") {
                joint_modes_.push_back(ControlMode::POSITION);
            } else if (joint.command_interfaces.size() == 1 &&
                       joint.command_interfaces[0].name == "velocity") {
                joint_modes_.push_back(ControlMode::VELOCITY);
            } else if (joint.command_interfaces.size() == 1 &&
                       joint.command_interfaces[0].name == "effort") {
                joint_modes_.push_back(ControlMode::TORQUE);
            }
        }

        // Initialize actual hardware
        initializeHardware();

        return CallbackReturn::SUCCESS;
    }

    std::vector<hardware_interface::StateInterface> export_state_interfaces() override {
        std::vector<hardware_interface::StateInterface> state_interfaces;

        for (size_t i = 0; i < info_.joints.size(); i++) {
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_positions_[i]));
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &hw_velocities_[i]));
            state_interfaces.emplace_back(hardware_interface::StateInterface(
                info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_efforts_[i]));
        }

        return state_interfaces;
    }

    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override {
        std::vector<hardware_interface::CommandInterface> command_interfaces;

        for (size_t i = 0; i < info_.joints.size(); i++) {
            command_interfaces.emplace_back(hardware_interface::CommandInterface(
                info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]));
        }

        return command_interfaces;
    }

    hardware_interface::CallbackReturn on_activate(
        const rclcpp_lifecycle::State & previous_state) override {

        // Activate hardware
        for (auto& controller : joint_controllers_) {
            controller.enable();
        }

        return CallbackReturn::SUCCESS;
    }

    hardware_interface::CallbackReturn on_deactivate(
        const rclcpp_lifecycle::State & previous_state) override {

        // Deactivate hardware safely
        for (auto& controller : joint_controllers_) {
            controller.disable();
        }

        return CallbackReturn::SUCCESS;
    }

    hardware_interface::return_type read(
        const rclcpp::Time & time, const rclcpp::Duration & period) override {

        // Read joint states from hardware
        for (size_t i = 0; i < joint_controllers_.size(); i++) {
            hw_positions_[i] = joint_controllers_[i].getPosition();
            hw_velocities_[i] = joint_controllers_[i].getVelocity();
            hw_efforts_[i] = joint_controllers_[i].getTorque();
        }

        return hardware_interface::return_type::OK;
    }

    hardware_interface::return_type write(
        const rclcpp::Time & time, const rclcpp::Duration & period) override {

        // Write commands to hardware
        for (size_t i = 0; i < joint_controllers_.size(); i++) {
            joint_controllers_[i].setCommand(hw_commands_[i]);
        }

        return hardware_interface::return_type::OK;
    }

private:
    enum class ControlMode { POSITION, VELOCITY, TORQUE };

    std::vector<ControlMode> joint_modes_;
    std::vector<JointController> joint_controllers_;

    std::vector<double> hw_positions_;
    std::vector<double> hw_velocities_;
    std::vector<double> hw_efforts_;
    std::vector<double> hw_commands_;

    void initializeHardware() {
        // Initialize actual hardware components
        // This would interface with the physical robot
    }
};
```

## Performance Evaluation

### Metrics

#### Mechanical Performance
- Position accuracy and repeatability
- Torque and speed capabilities
- Efficiency and power consumption

#### System Performance
- Control loop timing and jitter
- Communication latency
- Overall system reliability

### Testing Procedures

#### Factory Testing
- Individual component verification
- Integration testing
- Safety system validation

#### Field Testing
- Real-world performance validation
- Durability and longevity testing
- User experience evaluation

## Conclusion

Building humanoid robots is a complex multidisciplinary endeavor that requires expertise in mechanical engineering, electrical engineering, and software development. Success depends on careful selection of components, proper integration, and thorough testing.

The choice of actuators, sensors, and control systems must be matched to the specific requirements of the application, considering factors such as payload capacity, speed requirements, precision needs, and budget constraints.

Modern humanoid development benefits from standardized interfaces, open-source software, and modular designs that allow for rapid prototyping and iteration. As the field advances, we can expect continued improvements in component miniaturization, efficiency, and cost-effectiveness.

The next chapter will explore autonomous navigation for humanoid robots, which builds upon the physical platform to enable mobile operation in complex environments.

## Exercises

1. Design a joint mechanism for a humanoid robot's knee, specifying actuator, gearbox, and sensor requirements.

2. Calculate the power requirements for a humanoid robot with 20 joints, each requiring 100W of peak power.

3. Research and compare different actuator technologies (servo, BLDC, SEA) for a specific humanoid application.

4. Design a safety system for a humanoid robot that includes emergency stops, torque limiting, and collision detection.

5. Create a bill of materials for a basic humanoid robot with 12 degrees of freedom, including approximate costs.