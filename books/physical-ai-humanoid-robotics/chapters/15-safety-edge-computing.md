---
title: "Chapter 15: Safety, Fail-safes, Edge Computing"
description: "Ensuring safe operation and efficient computation for humanoid robots"
---

# Chapter 15: Safety, Fail-safes, Edge Computing

## Overview

Safety is paramount in humanoid robotics, especially when these robots operate in human environments. This chapter explores comprehensive safety systems, fail-safe mechanisms, and edge computing strategies that ensure humanoid robots operate reliably and safely while performing complex tasks with real-time computational requirements.

## Safety Framework for Humanoid Robotics

### Safety by Design Principles

#### Inherent Safety
- Mechanical design that minimizes harm potential
- Low-power actuators and compliant control
- Rounded edges and soft materials where possible

#### Active Safety Systems
- Real-time monitoring and intervention
- Predictive safety using AI models
- Multi-layered safety architecture

#### Functional Safety Standards
- IEC 61508 for electrical/electronic systems
- ISO 13482 for service robots
- ISO 10218 for industrial robots

### Safety Architecture Layers

```cpp
class SafetySystem {
public:
    enum SafetyLevel {
        SAFETY_LEVEL_1 = 1,  // Basic safety functions
        SAFETY_LEVEL_2 = 2,  // Enhanced safety with redundancy
        SAFETY_LEVEL_3 = 3,  // High-integrity safety-critical systems
        SAFETY_LEVEL_4 = 4   // Mission-critical systems
    };

    SafetySystem(SafetyLevel level) : safety_level_(level) {
        initializeSafetyLayers();
    }

    bool checkSafety() {
        // Check all safety layers
        bool layer1_ok = checkBasicSafety();
        bool layer2_ok = checkEnhancedSafety();
        bool layer3_ok = checkCriticalSafety();

        if (safety_level_ >= SAFETY_LEVEL_2) {
            if (!layer1_ok) return triggerEmergencyStop();
        }

        if (safety_level_ >= SAFETY_LEVEL_3) {
            if (!layer2_ok) return triggerEmergencyStop();
        }

        return layer1_ok && layer2_ok && layer3_ok;
    }

private:
    SafetyLevel safety_level_;

    void initializeSafetyLayers() {
        // Initialize different safety layers based on safety level
        basic_layer_ = std::make_unique<BasicSafetyLayer>();
        enhanced_layer_ = std::make_unique<EnhancedSafetyLayer>();
        critical_layer_ = std::make_unique<CriticalSafetyLayer>();
    }

    bool checkBasicSafety() {
        // Check basic operational parameters
        return basic_layer_->checkOperationalLimits() &&
               basic_layer_->checkHardwareHealth();
    }

    bool checkEnhancedSafety() {
        // Check for potential hazards
        return enhanced_layer_->checkCollisionRisk() &&
               enhanced_layer_->checkStabilityMargins();
    }

    bool checkCriticalSafety() {
        // Check mission-critical safety parameters
        return critical_layer_->checkCriticalSystems() &&
               critical_layer_->checkEnvironmentalHazards();
    }

    bool triggerEmergencyStop() {
        // Execute emergency stop sequence
        return emergency_stop_handler_.executeStopSequence();
    }

    std::unique_ptr<BasicSafetyLayer> basic_layer_;
    std::unique_ptr<EnhancedSafetyLayer> enhanced_layer_;
    std::unique_ptr<CriticalSafetyLayer> critical_layer_;
    EmergencyStopHandler emergency_stop_handler_;
};
```

## Humanoid-Specific Safety Considerations

### Dynamic Balance Safety

#### Zero Moment Point (ZMP) Monitoring
```python
class BalanceSafetyMonitor:
    def __init__(self, com_height=0.8, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

        # Define safety margins
        self.zmp_margin = 0.05  # 5cm safety margin
        self.support_polygon = self.defineSupportPolygon()

    def calculate_zmp(self, com_pos, com_vel, com_acc):
        """
        Calculate Zero Moment Point for current state
        """
        # ZMP = CoM position - (CoM acceleration / omega^2)
        zmp_x = com_pos[0] - (com_acc[0] / (self.omega ** 2))
        zmp_y = com_pos[1] - (com_acc[1] / (self.omega ** 2))

        return np.array([zmp_x, zmp_y])

    def is_balance_safe(self, zmp, current_support_polygon=None):
        """
        Check if ZMP is within safe support polygon
        """
        if current_support_polygon is None:
            current_support_polygon = self.support_polygon

        # Calculate safe polygon with margins
        safe_polygon = self.expandPolygon(current_support_polygon, -self.zmp_margin)

        # Check if ZMP is within safe polygon
        return self.pointInPolygon(zmp, safe_polygon)

    def defineSupportPolygon(self):
        """
        Define support polygon based on foot positions
        """
        # Simplified: rectangular support polygon based on foot positions
        foot_width = 0.12  # 12cm foot width
        foot_length = 0.25 # 25cm foot length

        # For bipedal stance with feet shoulder-width apart
        half_stride = 0.15  # 15cm half stride
        half_width = 0.10   # 10cm half foot separation

        return np.array([
            [-half_stride, -half_width - foot_width/2],  # Left foot back-left
            [-half_stride, half_width + foot_width/2],   # Right foot back-right
            [half_stride + foot_length, half_width + foot_width/2],   # Right foot front-right
            [half_stride + foot_length, -half_width - foot_width/2]   # Left foot front-left
        ])

    def expandPolygon(self, polygon, margin):
        """
        Expand or contract polygon by margin
        Positive margin expands, negative contracts
        """
        # Simplified implementation - in practice would use more sophisticated algorithms
        center = np.mean(polygon, axis=0)
        expanded = []
        for vertex in polygon:
            direction = vertex - center
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0:
                expanded_vertex = vertex + (direction / direction_norm) * margin
            else:
                expanded_vertex = vertex
            expanded.append(expanded_vertex)
        return np.array(expanded)

    def pointInPolygon(self, point, polygon):
        """
        Check if point is inside polygon using ray casting algorithm
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
```

### Fall Prevention and Recovery

```python
class FallPreventionSystem:
    def __init__(self):
        self.recovery_thresholds = {
            'angular_velocity': np.radians(10),  # 10 deg/s
            'com_deviation': 0.15,               # 15cm from base
            'zmp_deviation': 0.10               # 10cm from support
        }

        self.recovery_strategies = [
            'ankle_strategy',
            'hip_strategy',
            'stepping_strategy',
            'grabbing_strategy'
        ]

    def assessFallRisk(self, robot_state):
        """
        Assess fall risk based on current robot state
        """
        risk_metrics = {
            'angular_velocity_risk': self.assessAngularVelocityRisk(robot_state),
            'com_stability_risk': self.assessCOMStabilityRisk(robot_state),
            'zmp_stability_risk': self.assessZMPStabilityRisk(robot_state)
        }

        overall_risk = max(risk_metrics.values())

        return {
            'risk_level': overall_risk,
            'risk_metrics': risk_metrics,
            'recommended_strategy': self.selectRecoveryStrategy(risk_metrics)
        }

    def assessAngularVelocityRisk(self, robot_state):
        """
        Assess risk based on angular velocity
        """
        angular_vel = robot_state['angular_velocity']
        magnitude = np.linalg.norm(angular_vel)

        # Risk increases as velocity approaches threshold
        risk = min(magnitude / self.recovery_thresholds['angular_velocity'], 1.0)
        return risk

    def assessCOMStabilityRisk(self, robot_state):
        """
        Assess risk based on center of mass position
        """
        com_pos = robot_state['com_position']
        base_pos = robot_state['base_position']

        # Calculate deviation from stable base
        deviation = np.linalg.norm(com_pos[:2] - base_pos[:2])

        risk = min(deviation / self.recovery_thresholds['com_deviation'], 1.0)
        return risk

    def assessZMPStabilityRisk(self, robot_state):
        """
        Assess risk based on ZMP position
        """
        zmp = robot_state['zmp']
        support_polygon = robot_state['support_polygon']

        # Calculate distance to nearest edge of support polygon
        distance_to_edge = self.distanceToPolygonEdge(zmp, support_polygon)

        if distance_to_edge < 0:  # ZMP outside polygon
            risk = 1.0
        else:
            # Risk decreases as distance to edge increases
            max_distance = 0.2  # Maximum safe distance
            risk = 1.0 - min(distance_to_edge / max_distance, 1.0)

        return risk

    def selectRecoveryStrategy(self, risk_metrics):
        """
        Select appropriate recovery strategy based on risk assessment
        """
        if risk_metrics['angular_velocity_risk'] > 0.7:
            # High angular velocity - use ankle strategy if possible
            return 'ankle_strategy'
        elif risk_metrics['com_stability_risk'] > 0.5:
            # COM deviation - use hip strategy
            return 'hip_strategy'
        elif risk_metrics['zmp_stability_risk'] > 0.8:
            # ZMP far from support - step or grab
            return 'stepping_strategy'
        else:
            # General instability - consider grabbing
            return 'grabbing_strategy'

    def executeRecovery(self, strategy, robot_state):
        """
        Execute selected recovery strategy
        """
        if strategy == 'ankle_strategy':
            return self.executeAnkleStrategy(robot_state)
        elif strategy == 'hip_strategy':
            return self.executeHipStrategy(robot_state)
        elif strategy == 'stepping_strategy':
            return self.executeSteppingStrategy(robot_state)
        elif strategy == 'grabbing_strategy':
            return self.executeGrabbingStrategy(robot_state)
        else:
            return False  # Unknown strategy

    def executeAnkleStrategy(self, robot_state):
        """
        Ankle strategy: adjust ankle torques to move COM back over support
        """
        # Calculate required ankle torques
        com_pos = robot_state['com_position']
        zmp = robot_state['zmp']

        # Simple proportional control
        ankle_torque = -100 * (zmp - com_pos[:2])  # 100 Nm/m gain

        # Apply torques
        return self.applyAnkleTorques(ankle_torque)

    def executeHipStrategy(self, robot_state):
        """
        Hip strategy: move hip to shift COM back over support
        """
        # Calculate hip movement needed
        com_pos = robot_state['com_position']
        support_center = robot_state['support_center']

        hip_move = 0.5 * (support_center - com_pos[:2])  # 50% correction

        # Move hip
        return self.moveHip(hip_move)

    def executeSteppingStrategy(self, robot_state):
        """
        Stepping strategy: take a step to expand support polygon
        """
        # Calculate where to step
        zmp = robot_state['zmp']
        support_center = robot_state['support_center']

        # Step in direction opposite to ZMP deviation
        step_direction = zmp - support_center
        step_distance = min(np.linalg.norm(step_direction), 0.3)  # Max 30cm step

        if step_distance > 0.05:  # Only step if significantly off balance
            step_direction = step_direction / step_distance  # Normalize
            target_step_pos = support_center + step_direction * step_distance

            return self.takeStep(target_step_pos)
        else:
            return False  # No need to step

    def executeGrabbingStrategy(self, robot_state):
        """
        Grabbing strategy: reach for support surface
        """
        # Look for nearby support surfaces
        nearby_surfaces = self.findNearbySurfaces(robot_state)

        if nearby_surfaces:
            # Reach for closest surface
            closest_surface = min(nearby_surfaces, key=lambda s: s['distance'])
            return self.grabSurface(closest_surface)
        else:
            return False  # No surfaces to grab

    def distanceToPolygonEdge(self, point, polygon):
        """
        Calculate minimum distance from point to polygon edges
        """
        min_distance = float('inf')

        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            distance = self.distancePointToLineSegment(point, p1, p2)
            min_distance = min(min_distance, distance)

        return min_distance

    def distancePointToLineSegment(self, point, line_start, line_end):
        """
        Calculate distance from point to line segment
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        point_vec = point - line_start

        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:
            return np.linalg.norm(point - line_start)

        # Project point_vec onto line_vec
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))

        # Closest point on line segment
        projection = line_start + t * line_vec

        return np.linalg.norm(point - projection)
```

## Collision Safety Systems

### Human-Robot Collision Safety

```cpp
class CollisionSafetySystem {
public:
    CollisionSafetySystem(double max_collision_force = 150.0)  // 150N max
        : max_force_(max_collision_force),
          collision_detected_(false),
          collision_location_({0, 0, 0}) {}

    struct CollisionData {
        std::vector<std::string> affected_links;
        std::vector<double> impact_forces;
        std::vector<std::array<double, 3>> impact_positions;
        double timestamp;
        bool is_human_contact;
    };

    bool checkCollisionSafety(const RobotState& state) {
        // Check for collisions using distance sensors and force/torque sensors
        std::vector<CollisionData> collisions = detectCollisions(state);

        for (const auto& collision : collisions) {
            if (isUnsafeCollision(collision)) {
                handleUnsafeCollision(collision);
                return false;  // Unsafe state
            }
        }

        return true;  // Safe state
    }

    void setSafetyThresholds(double force_threshold, double velocity_threshold) {
        force_threshold_ = force_threshold;
        velocity_threshold_ = velocity_threshold;
    }

private:
    double max_force_;
    double force_threshold_ = 150.0;  // Default 150N
    double velocity_threshold_ = 0.5; // Default 0.5 m/s
    bool collision_detected_;
    std::array<double, 3> collision_location_;

    std::vector<CollisionData> detectCollisions(const RobotState& state) {
        std::vector<CollisionData> collisions;

        // Check distance sensors
        for (const auto& sensor : distance_sensors_) {
            if (sensor.getDistance() < safety_margin_) {
                CollisionData collision;
                collision.affected_links = {sensor.getAssociatedLink()};
                collision.impact_forces = {estimateImpactForce(sensor.getDistance())};
                collision.impact_positions = {sensor.getPosition()};
                collision.timestamp = getCurrentTime();
                collision.is_human_contact = isHumanDetected(sensor.getPosition());

                collisions.push_back(collision);
            }
        }

        // Check force/torque sensors
        for (const auto& ft_sensor : force_torque_sensors_) {
            auto wrench = ft_sensor.getWrench();
            double force_magnitude = std::sqrt(
                wrench.fx * wrench.fx +
                wrench.fy * wrench.fy +
                wrench.fz * wrench.fz
            );

            if (force_magnitude > force_threshold_) {
                CollisionData collision;
                collision.affected_links = {ft_sensor.getLink()};
                collision.impact_forces = {force_magnitude};
                collision.impact_positions = {ft_sensor.getPosition()};
                collision.timestamp = getCurrentTime();
                collision.is_human_contact = true;  // Assume human contact for high forces

                collisions.push_back(collision);
            }
        }

        return collisions;
    }

    bool isUnsafeCollision(const CollisionData& collision) {
        // Check if collision exceeds safety thresholds
        for (double force : collision.impact_forces) {
            if (force > force_threshold_) {
                return true;
            }
        }

        // Check if collision involves sensitive areas (head, face, etc.)
        for (const auto& pos : collision.impact_positions) {
            if (isSensitiveArea(pos) && collision.is_human_contact) {
                return true;
            }
        }

        return false;
    }

    void handleUnsafeCollision(const CollisionData& collision) {
        // Execute collision response
        emergencyStop();

        // Log collision for analysis
        logCollision(collision);

        // Notify operators
        notifyOperators(collision);

        // Initiate recovery sequence if needed
        if (collision.is_human_contact) {
            initiateHumanSafetyProtocol();
        }
    }

    double estimateImpactForce(double distance) {
        // Simple model: force increases as distance decreases
        // In practice, this would be more sophisticated
        if (distance <= 0.01) {  // 1cm threshold
            return max_force_;
        }
        return max_force_ * (0.05 - distance) / 0.04;  // Linear approximation
    }

    bool isHumanDetected(const std::array<double, 3>& position) {
        // Use vision or other sensors to detect if contact is with human
        // This is a simplified model
        return human_detection_system_.isPersonNear(position);
    }

    bool isSensitiveArea(const std::array<double, 3>& position) {
        // Check if impact location is in sensitive area (head, face, etc.)
        double head_height = 1.5;  // Approximate head height
        double head_radius = 0.15; // Approximate head radius

        if (position[2] > head_height - head_radius) {
            // Check if within head region
            double lateral_distance = std::sqrt(position[0]*position[0] + position[1]*position[1]);
            if (lateral_distance < head_radius) {
                return true;
            }
        }

        return false;
    }

    void emergencyStop() {
        // Cut power to all actuators immediately
        for (auto& actuator : actuators_) {
            actuator.emergencyStop();
        }
    }

    void logCollision(const CollisionData& collision) {
        // Log collision data for safety analysis
        collision_logger_.logCollision(collision);
    }

    void notifyOperators(const CollisionData& collision) {
        // Send notification to operators
        safety_monitor_.sendCollisionAlert(collision);
    }

    void initiateHumanSafetyProtocol() {
        // Specific protocol for human-robot collisions
        human_safety_protocol_.execute();
    }

    std::vector<DistanceSensor> distance_sensors_;
    std::vector<ForceTorqueSensor> force_torque_sensors_;
    std::vector<Actuator> actuators_;
    HumanDetectionSystem human_detection_system_;
    CollisionLogger collision_logger_;
    SafetyMonitor safety_monitor_;
    HumanSafetyProtocol human_safety_protocol_;

    double safety_margin_ = 0.05;  // 5cm safety margin
};
```

## Fail-Safe Mechanisms

### Redundant Safety Systems

```python
class RedundantSafetySystem:
    def __init__(self):
        # Primary safety system
        self.primary_safety = SafetySystem(level=3)

        # Secondary safety system (independent hardware)
        self.secondary_safety = HardwareSafetySystem()

        # Tertiary safety system (manual override)
        self.tertiary_safety = ManualOverrideSystem()

        # Voting system for critical decisions
        self.voting_threshold = 2  # Need 2 out of 3 systems to agree

    def checkSafety(self):
        """
        Check safety using redundant systems
        """
        primary_ok = self.primary_safety.checkSafety()
        secondary_ok = self.secondary_safety.checkSafety()
        tertiary_ok = self.tertiary_safety.checkSafety()

        safety_votes = sum([primary_ok, secondary_ok, tertiary_ok])

        if safety_votes >= self.voting_threshold:
            return True, "Safe"
        else:
            # Trigger emergency stop
            self.triggerEmergencyStop()
            return False, "Unsafe - Emergency Stop Activated"

    def triggerEmergencyStop(self):
        """
        Trigger emergency stop across all systems
        """
        self.primary_safety.emergencyStop()
        self.secondary_safety.emergencyStop()
        self.tertiary_safety.emergencyStop()

class HardwareSafetySystem:
    def __init__(self):
        # Dedicated hardware safety circuits
        self.hardware_watchdog = HardwareWatchdog()
        self.emergency_stop_button = EmergencyStopButton()
        self.power_cut_off = PowerCutOffSwitch()
        self.backup_battery = BackupBattery()

    def checkSafety(self):
        """
        Check safety using dedicated hardware
        """
        # Check watchdog timer
        if not self.hardware_watchdog.isAlive():
            return False

        # Check for emergency stop activation
        if self.emergency_stop_button.isPressed():
            return False

        # Check power system integrity
        if not self.power_cut_off.isNormal():
            return False

        return True

    def emergencyStop(self):
        """
        Execute hardware-level emergency stop
        """
        # Cut main power
        self.power_cut_off.cutPower()

        # Activate backup systems if needed
        if not self.backup_battery.isActive():
            self.backup_battery.activate()

class ManualOverrideSystem:
    def __init__(self):
        self.remote_control = RemoteControlInterface()
        self.manual_emergency_stop = ManualEmergencyStop()
        self.operator_authentication = OperatorAuthentication()

    def checkSafety(self):
        """
        Check if manual override is available and safe
        """
        # Check if operator is authenticated
        if not self.operator_authentication.isAuthenticated():
            return False

        # Check if remote control is responsive
        if not self.remote_control.isConnected():
            return False

        return True

    def emergencyStop(self):
        """
        Execute manual emergency stop
        """
        self.manual_emergency_stop.activate()
```

### Graceful Degradation

```cpp
class GracefulDegradationManager {
public:
    enum SystemHealth {
        HEALTHY = 0,
        DEGRADED_PERFORMANCE = 1,
        LIMITED_FUNCTIONALITY = 2,
        EMERGENCY_ONLY = 3,
        SYSTEM_DOWN = 4
    };

    GracefulDegradationManager() {
        initializeSystemHierarchy();
    }

    SystemHealth assessSystemHealth() {
        // Check all subsystems
        auto subsystem_health = checkAllSubsystems();

        // Determine overall system health based on critical components
        SystemHealth overall_health = HEALTHY;

        for (const auto& [component, health] : subsystem_health) {
            if (health > overall_health) {
                overall_health = health;
            }
        }

        return overall_health;
    }

    void executeDegradationPlan(SystemHealth current_health) {
        switch (current_health) {
            case HEALTHY:
                // Normal operation
                restoreNormalOperation();
                break;

            case DEGRADED_PERFORMANCE:
                // Reduce speed and precision
                reducePerformanceRequirements();
                break;

            case LIMITED_FUNCTIONALITY:
                // Limit to essential functions only
                switchToEssentialFunctions();
                break;

            case EMERGENCY_ONLY:
                // Only safety and basic functions
                enterEmergencyMode();
                break;

            case SYSTEM_DOWN:
                // Complete shutdown
                shutdownSafely();
                break;
        }
    }

private:
    std::map<std::string, SystemHealth> subsystem_health_;
    std::vector<std::string> critical_components_;

    void initializeSystemHierarchy() {
        // Define critical components and their dependencies
        critical_components_ = {
            "power_system",
            "balance_control",
            "collision_avoidance",
            "emergency_stop"
        };
    }

    std::map<std::string, SystemHealth> checkAllSubsystems() {
        std::map<std::string, SystemHealth> health_status;

        // Check power system
        health_status["power_system"] = checkPowerSystem();

        // Check balance control
        health_status["balance_control"] = checkBalanceControl();

        // Check collision avoidance
        health_status["collision_avoidance"] = checkCollisionAvoidance();

        // Check emergency systems
        health_status["emergency_systems"] = checkEmergencySystems();

        return health_status;
    }

    SystemHealth checkPowerSystem() {
        double battery_level = power_monitor_.getBatteryLevel();
        double voltage_stability = power_monitor_.getVoltageStability();

        if (battery_level < 0.1) return SYSTEM_DOWN;
        if (battery_level < 0.2 || voltage_stability < 0.8) return EMERGENCY_ONLY;
        if (battery_level < 0.3) return LIMITED_FUNCTIONALITY;
        if (voltage_stability < 0.9) return DEGRADED_PERFORMANCE;

        return HEALTHY;
    }

    SystemHealth checkBalanceControl() {
        double control_stability = balance_controller_.getStabilityMetric();
        double sensor_health = balance_controller_.getSensorHealth();

        if (control_stability < 0.3 || sensor_health < 0.4) return SYSTEM_DOWN;
        if (control_stability < 0.5 || sensor_health < 0.6) return EMERGENCY_ONLY;
        if (control_stability < 0.7 || sensor_health < 0.8) return LIMITED_FUNCTIONALITY;
        if (control_stability < 0.9 || sensor_health < 0.95) return DEGRADED_PERFORMANCE;

        return HEALTHY;
    }

    SystemHealth checkCollisionAvoidance() {
        double sensor_coverage = collision_avoidance_.getSensorCoverage();
        double algorithm_confidence = collision_avoidance_.getAlgorithmConfidence();

        if (sensor_coverage < 0.3 || algorithm_confidence < 0.3) return EMERGENCY_ONLY;
        if (sensor_coverage < 0.6 || algorithm_confidence < 0.6) return LIMITED_FUNCTIONALITY;
        if (sensor_coverage < 0.8 || algorithm_confidence < 0.8) return DEGRADED_PERFORMANCE;

        return HEALTHY;
    }

    SystemHealth checkEmergencySystems() {
        double estop_health = emergency_systems_.getEStopHealth();
        double backup_power_health = emergency_systems_.getBackupPowerHealth();

        if (estop_health < 0.5 || backup_power_health < 0.5) return SYSTEM_DOWN;
        if (estop_health < 0.8 || backup_power_health < 0.8) return EMERGENCY_ONLY;

        return HEALTHY;
    }

    void restoreNormalOperation() {
        // Restore full functionality
        for (auto& component : operational_components_) {
            component->restoreFullFunctionality();
        }
    }

    void reducePerformanceRequirements() {
        // Reduce speed limits, increase safety margins
        motion_controller_.setMaxSpeed(0.5);  // Reduce to 50% speed
        safety_system_.increaseSafetyMargins(0.2);  // Increase margins by 20%
    }

    void switchToEssentialFunctions() {
        // Limit to essential safety and basic movement
        disableNonEssentialSystems();
        enableEssentialSystems();

        // Limit to safe homing position
        motion_controller_.goToSafeHomePosition();
    }

    void enterEmergencyMode() {
        // Only emergency stop and basic monitoring
        disableAllMovement();
        enableMonitoringOnly();
    }

    void shutdownSafely() {
        // Execute safe shutdown sequence
        executeSafeShutdownSequence();
    }

    PowerMonitor power_monitor_;
    BalanceController balance_controller_;
    CollisionAvoidanceSystem collision_avoidance_;
    EmergencySystems emergency_systems_;

    std::vector<SystemComponent*> operational_components_;
    MotionController motion_controller_;
    SafetySystem safety_system_;
};
```

## Edge Computing for Safety-Critical Systems

### Real-Time Processing Requirements

```python
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

class SafetyCriticalEdgeComputing:
    def __init__(self, cpu_affinity=None):
        self.cpu_affinity = cpu_affinity or [0]  # Use specific CPU cores for safety tasks
        self.real_time_priority = 99  # Highest priority for safety tasks

        # Initialize safety-critical processing queues
        self.safety_queue = asyncio.Queue()
        self.control_queue = asyncio.Queue()
        self.monitoring_queue = asyncio.Queue()

        # Processing times for different safety levels
        self.processing_deadlines = {
            'critical': 0.001,    # 1ms deadline
            'high': 0.010,       # 10ms deadline
            'medium': 0.050,     # 50ms deadline
            'low': 0.100         # 100ms deadline
        }

        # Initialize real-time processing tasks
        self.safety_processor = RealTimeSafetyProcessor()
        self.control_processor = RealTimeControlProcessor()
        self.monitoring_processor = RealTimeMonitoringProcessor()

    async def processSafetyCriticalTasks(self):
        """
        Process safety-critical tasks with real-time guarantees
        """
        while True:
            try:
                # Get safety task with highest priority
                task = await asyncio.wait_for(
                    self.safety_queue.get(),
                    timeout=self.processing_deadlines['critical']
                )

                # Process with real-time constraints
                start_time = time.time()

                result = await self.safety_processor.process(task)

                processing_time = time.time() - start_time

                # Check if processing met deadline
                if processing_time > self.processing_deadlines['critical']:
                    self.handleDeadlineMiss('safety', processing_time)

                # Put result in appropriate queue
                await self.handleSafetyResult(result)

            except asyncio.TimeoutError:
                # Continue processing even if no tasks
                continue
            except Exception as e:
                self.handleProcessingError('safety', e)

    async def processControlTasks(self):
        """
        Process control tasks with high priority
        """
        while True:
            try:
                task = await asyncio.wait_for(
                    self.control_queue.get(),
                    timeout=self.processing_deadlines['high']
                )

                start_time = time.time()
                result = await self.control_processor.process(task)
                processing_time = time.time() - start_time

                if processing_time > self.processing_deadlines['high']:
                    self.handleDeadlineMiss('control', processing_time)

                await self.handleControlResult(result)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.handleProcessingError('control', e)

    def setupRealTimeEnvironment(self):
        """
        Configure real-time environment for safety-critical processing
        """
        # Set CPU affinity for safety threads
        import os
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, self.cpu_affinity)

        # Use real-time scheduling if available
        try:
            import resource
            # Set to SCHED_FIFO (real-time first-in-first-out)
            # This requires root privileges on most systems
            pass
        except ImportError:
            pass

class RealTimeSafetyProcessor:
    def __init__(self):
        self.safety_algorithms = {
            'collision_detection': self.collisionDetection,
            'balance_monitoring': self.balanceMonitoring,
            'emergency_stop': self.emergencyStop,
            'hardware_watchdog': self.hardwareWatchdog
        }

    async def process(self, task):
        """
        Process safety task with real-time guarantees
        """
        task_type = task.get('type')

        if task_type in self.safety_algorithms:
            # Execute with real-time constraints
            return await self.safety_algorithms[task_type](task)
        else:
            raise ValueError(f"Unknown safety task type: {task_type}")

    async def collisionDetection(self, task):
        """
        Real-time collision detection
        """
        sensor_data = task['sensor_data']

        # Use optimized collision detection algorithm
        collision_result = self.optimizedCollisionCheck(sensor_data)

        return {
            'type': 'collision_result',
            'timestamp': time.time(),
            'collision_detected': collision_result['detected'],
            'collision_location': collision_result['location'],
            'severity': collision_result['severity']
        }

    async def balanceMonitoring(self, task):
        """
        Real-time balance monitoring
        """
        state_data = task['state_data']

        # Calculate ZMP and check stability
        zmp = self.calculateZMP(state_data)
        stability = self.checkStability(zmp, state_data['support_polygon'])

        return {
            'type': 'balance_result',
            'timestamp': time.time(),
            'zmp': zmp,
            'stability_margin': stability['margin'],
            'risk_level': stability['risk']
        }

    async def emergencyStop(self, task):
        """
        Emergency stop execution
        """
        reason = task.get('reason', 'unknown')

        # Execute immediate stop sequence
        stop_result = self.executeImmediateStop(reason)

        return {
            'type': 'emergency_stop_result',
            'timestamp': time.time(),
            'executed': stop_result['success'],
            'reason': reason
        }

    def optimizedCollisionCheck(self, sensor_data):
        """
        Optimized collision detection algorithm
        """
        # Simplified implementation - in practice would use:
        # - Bounding volume hierarchies
        # - Spatial hashing
        # - GPU acceleration
        # - Precomputed collision maps

        detected = False
        location = [0, 0, 0]
        severity = 0.0

        # Check distance sensors
        for sensor in sensor_data.get('distance_sensors', []):
            if sensor['distance'] < 0.05:  # 5cm threshold
                detected = True
                location = sensor['position']
                severity = max(severity, (0.05 - sensor['distance']) / 0.05)

        return {
            'detected': detected,
            'location': location,
            'severity': severity
        }

    def calculateZMP(self, state_data):
        """
        Calculate Zero Moment Point from state data
        """
        com_pos = state_data['com_position']
        com_acc = state_data['com_acceleration']

        gravity = 9.81
        zmp_x = com_pos[0] - (com_pos[2] * com_acc[0]) / gravity
        zmp_y = com_pos[1] - (com_pos[2] * com_acc[1]) / gravity

        return [zmp_x, zmp_y]

    def checkStability(self, zmp, support_polygon):
        """
        Check if ZMP is within support polygon
        """
        from shapely.geometry import Point, Polygon

        zmp_point = Point(zmp[0], zmp[1])
        support_poly = Polygon(support_polygon)

        # Calculate distance to polygon boundary
        if support_poly.contains(zmp_point):
            # Inside polygon - positive margin
            boundary = support_poly.boundary
            distance = boundary.distance(zmp_point)
            return {'margin': distance, 'risk': 0.0}
        else:
            # Outside polygon - negative margin
            boundary = support_poly.boundary
            distance = boundary.distance(zmp_point)
            return {'margin': -distance, 'risk': 1.0}

class EdgeSafetyOrchestrator:
    def __init__(self):
        self.processors = {}
        self.load_balancer = LoadBalancer()
        self.fault_detector = FaultDetector()
        self.resource_manager = ResourceManager()

    def distributeSafetyTasks(self, tasks):
        """
        Distribute safety tasks across available edge computing resources
        """
        # Categorize tasks by safety level
        critical_tasks = [t for t in tasks if t.get('safety_level') == 'critical']
        high_tasks = [t for t in tasks if t.get('safety_level') == 'high']
        medium_tasks = [t for t in tasks if t.get('safety_level') == 'medium']

        # Assign critical tasks to dedicated safety processors
        for task in critical_tasks:
            processor = self.getDedicatedSafetyProcessor()
            processor.submitTask(task)

        # Distribute other tasks based on resource availability
        for task in high_tasks + medium_tasks:
            available_processor = self.load_balancer.getAvailableProcessor(task)
            if available_processor:
                available_processor.submitTask(task)

    def getDedicatedSafetyProcessor(self):
        """
        Get dedicated processor for critical safety tasks
        """
        # Check if we have a dedicated safety processor
        if 'safety_dedicated' in self.processors:
            return self.processors['safety_dedicated']

        # Create new dedicated processor
        safety_proc = SafetyDedicatedProcessor()
        self.processors['safety_dedicated'] = safety_proc
        return safety_proc

    def monitorResourceUsage(self):
        """
        Monitor resource usage and redistribute tasks if needed
        """
        while True:
            # Check resource utilization
            resource_usage = self.resource_manager.getResourceUsage()

            # Detect overloaded processors
            overloaded = [proc for proc, usage in resource_usage.items()
                         if usage['cpu'] > 0.9 or usage['memory'] > 0.9]

            if overloaded:
                # Redistribute tasks from overloaded processors
                for proc in overloaded:
                    self.redistributeProcessorTasks(proc)

            # Check for processor failures
            failed_processors = self.fault_detector.detectFailures()
            if failed_processors:
                self.handleProcessorFailures(failed_processors)

            time.sleep(0.1)  # Check every 100ms

    def handleProcessorFailures(self, failed_processors):
        """
        Handle failures of safety processors
        """
        for proc in failed_processors:
            # Reassign tasks from failed processor
            tasks_to_reassign = proc.getPendingTasks()
            self.distributeSafetyTasks(tasks_to_reassign)

            # Remove failed processor
            del self.processors[proc.id]

            # Create replacement processor if needed
            if proc.type == 'safety_dedicated':
                self.processors['safety_dedicated'] = SafetyDedicatedProcessor()
```

### Hardware Acceleration for Safety

```cpp
#include <cuda_runtime.h>
#include <opencl/opencl.h>

class SafetyHardwareAccelerator {
public:
    SafetyHardwareAccelerator() {
        initializeAccelerators();
    }

    // GPU-accelerated collision detection
    bool detectCollisionsGPU(const float* points, int num_points,
                            float* distances, int max_distance) {
        if (!gpu_available_) return false;

        // Launch CUDA kernel for collision detection
        cudaError_t err = launchCollisionDetectionKernel(
            points, num_points, distances, max_distance
        );

        return err == cudaSuccess;
    }

    // FPGA-accelerated safety monitoring
    bool monitorSafetySignalsFPGA() {
        if (!fpga_available_) return false;

        // Configure FPGA for safety monitoring
        configureFPGASafetyLogic();

        // Monitor critical signals
        return readFPGASafetyStatus();
    }

    // Real-time signal processing
    bool processSafetySignals(const float* signals, int signal_count) {
        if (signal_count > MAX_SIGNALS) return false;

        // Use SIMD instructions for parallel processing
        return processSignalBatch(signals, signal_count);
    }

private:
    bool gpu_available_ = false;
    bool fpga_available_ = false;
    bool fpga_configured_ = false;

    static const int MAX_SIGNALS = 1024;

    void initializeAccelerators() {
        // Check for GPU availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        gpu_available_ = (err == cudaSuccess && device_count > 0);

        // Check for FPGA availability
        fpga_available_ = checkFPGAAvailability();
    }

    cudaError_t launchCollisionDetectionKernel(const float* points, int num_points,
                                              float* distances, int max_distance) {
        // Allocate GPU memory
        float* d_points;
        float* d_distances;

        cudaMalloc(&d_points, num_points * 3 * sizeof(float));
        cudaMalloc(&d_distances, num_points * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_points, points, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int grid_size = (num_points + block_size - 1) / block_size;

        collisionDetectionKernel<<<grid_size, block_size>>>(
            d_points, num_points, d_distances, max_distance
        );

        // Copy results back
        cudaMemcpy(distances, d_distances, num_points * sizeof(float), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_points);
        cudaFree(d_distances);

        return cudaGetLastError();
    }

    bool checkFPGAAvailability() {
        // Check for FPGA presence and configuration capability
        // Implementation depends on specific FPGA platform
        return false; // Placeholder
    }

    void configureFPGASafetyLogic() {
        // Configure FPGA with safety monitoring logic
        // This would involve bitstream loading and register configuration
    }

    bool readFPGASafetyStatus() {
        // Read safety status from FPGA registers
        return true; // Placeholder
    }

    bool processSignalBatch(const float* signals, int count) {
        // Use SIMD instructions for parallel signal processing
        #ifdef __AVX__
        // Use AVX instructions for 8-way parallel processing
        int simd_count = count / 8;
        int remainder = count % 8;

        for (int i = 0; i < simd_count; i++) {
            __m256 signal_vec = _mm256_load_ps(&signals[i * 8]);
            // Process signals in parallel
            __m256 processed = processSignalVector(signal_vec);
            _mm256_store_ps(&signals[i * 8], processed);
        }

        // Process remaining signals
        for (int i = simd_count * 8; i < count; i++) {
            signals[i] = processSingleSignal(signals[i]);
        }

        return true;
        #else
        // Fallback to scalar processing
        for (int i = 0; i < count; i++) {
            if (processSingleSignal(signals[i]) < 0) return false;
        }
        return true;
        #endif
    }

    __m256 processSignalVector(__m256 signal_vec) {
        // Process 8 signals in parallel using AVX
        // Example: bound checking
        __m256 lower_bound = _mm256_set1_ps(-10.0f);
        __m256 upper_bound = _mm256_set1_ps(10.0f);

        // Clamp values to bounds
        __m256 clamped = _mm256_max_ps(_mm256_min_ps(signal_vec, upper_bound), lower_bound);

        return clamped;
    }

    float processSingleSignal(float signal) {
        // Process single signal with safety checks
        if (signal > 10.0f) return 10.0f;  // Upper bound
        if (signal < -10.0f) return -10.0f;  // Lower bound
        return signal;
    }
};

// CUDA kernel for collision detection
__global__ void collisionDetectionKernel(
    const float* points, int num_points,
    float* distances, int max_distance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        // Calculate distance to nearest obstacle
        float min_dist = max_distance;

        // Simple implementation - in practice would use spatial data structures
        for (int i = 0; i < num_points; i++) {
            if (i != idx) {
                float dx = points[idx * 3] - points[i * 3];
                float dy = points[idx * 3 + 1] - points[i * 3 + 1];
                float dz = points[idx * 3 + 2] - points[i * 3 + 2];

                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }

        distances[idx] = min_dist;
    }
}
```

## Safety Certification and Standards

### Compliance Framework

```python
class SafetyCertificationFramework:
    def __init__(self):
        self.certification_standards = {
            'ISO_13482': self.checkISO13482Compliance,
            'IEC_61508': self.checkIEC61508Compliance,
            'ISO_26262': self.checkISO26262Compliance,  # For mobile robots
            'EN_12100': self.checkEN12100Compliance      # Machinery safety
        }

        self.safety_requirements = self.loadSafetyRequirements()
        self.test_procedures = self.loadTestProcedures()
        self.documentation_templates = self.loadDocumentationTemplates()

    def performSafetyAssessment(self, robot_system):
        """
        Perform comprehensive safety assessment
        """
        assessment_results = {}

        for standard, checker in self.certification_standards.items():
            try:
                compliance_result = checker(robot_system)
                assessment_results[standard] = compliance_result
            except Exception as e:
                assessment_results[standard] = {
                    'compliant': False,
                    'issues': [str(e)],
                    'recommendations': []
                }

        return self.generateAssessmentReport(assessment_results)

    def checkISO13482Compliance(self, robot_system):
        """
        Check compliance with ISO 13482 (Personal Care Robots)
        """
        issues = []
        recommendations = []

        # Check personal space invasion limits
        if robot_system.max_speed > 0.7:  # 0.7 m/s limit for personal space
            issues.append("Robot speed exceeds ISO 13482 personal space limit")
            recommendations.append("Implement speed limiting in personal space zones")

        # Check force limits for physical contact
        if robot_system.max_end_effector_force > 150:  # 150N limit
            issues.append("End effector force exceeds ISO 13482 limit")
            recommendations.append("Implement force limiting in end effector")

        # Check emergency stop accessibility
        if not robot_system.has_accessible_estop():
            issues.append("Emergency stop not easily accessible")
            recommendations.append("Ensure emergency stop is within reach of users")

        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations
        }

    def checkIEC61508Compliance(self, robot_system):
        """
        Check compliance with IEC 61508 (Functional Safety)
        """
        issues = []
        recommendations = []

        # Check safety integrity level (SIL) requirements
        required_sil = robot_system.getRequiredSIL()
        implemented_sil = self.calculateImplementedSIL(robot_system)

        if implemented_sil < required_sil:
            issues.append(f"Safety Integrity Level insufficient: required={required_sil}, implemented={implemented_sil}")
            recommendations.append(f"Implement safety measures to achieve SIL {required_sil}")

        # Check diagnostic coverage
        diag_coverage = self.calculateDiagnosticCoverage(robot_system)
        if diag_coverage < self.getMinDiagnosticCoverage(required_sil):
            issues.append(f"Diagnostic coverage insufficient: achieved={diag_coverage:.2f}, required={self.getMinDiagnosticCoverage(required_sil):.2f}")
            recommendations.append("Improve diagnostic coverage with additional monitoring")

        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations
        }

    def calculateImplementedSIL(self, robot_system):
        """
        Calculate the implemented Safety Integrity Level
        """
        # Simplified calculation - in practice would involve detailed FMEA
        hardware_safety = robot_system.getHardwareSafetyScore()
        software_safety = robot_system.getSoftwareSafetyScore()
        systematic_safety = robot_system.getSystematicSafetyScore()

        # Combine scores to determine SIL
        combined_score = (hardware_safety + software_safety + systematic_safety) / 3

        if combined_score > 0.9:
            return 4  # SIL 4
        elif combined_score > 0.7:
            return 3  # SIL 3
        elif combined_score > 0.5:
            return 2  # SIL 2
        else:
            return 1  # SIL 1

    def calculateDiagnosticCoverage(self, robot_system):
        """
        Calculate diagnostic coverage of safety systems
        """
        detected_faults = robot_system.getDetectedFaults()
        total_possible_faults = robot_system.getTotalPossibleFaults()

        if total_possible_faults == 0:
            return 0.0

        return detected_faults / total_possible_faults

    def getMinDiagnosticCoverage(self, sil_level):
        """
        Get minimum diagnostic coverage required for SIL level
        """
        coverage_requirements = {
            1: 0.60,  # 60% for SIL 1
            2: 0.65,  # 65% for SIL 2
            3: 0.90,  # 90% for SIL 3
            4: 0.99   # 99% for SIL 4
        }
        return coverage_requirements.get(sil_level, 0.60)

    def generateAssessmentReport(self, assessment_results):
        """
        Generate comprehensive safety assessment report
        """
        report = {
            'timestamp': time.time(),
            'robot_system': 'Humanoid Robot Platform',
            'standards_evaluated': list(assessment_results.keys()),
            'overall_compliance': all(result['compliant'] for result in assessment_results.values()),
            'detailed_results': assessment_results,
            'executive_summary': self.generateExecutiveSummary(assessment_results),
            'certification_recommendation': self.generateCertificationRecommendation(assessment_results)
        }

        return report

    def generateExecutiveSummary(self, assessment_results):
        """
        Generate executive summary of assessment
        """
        compliant_standards = [std for std, result in assessment_results.items() if result['compliant']]
        non_compliant_standards = [std for std, result in assessment_results.items() if not result['compliant']]

        total_issues = sum(len(result['issues']) for result in assessment_results.values())

        return {
            'standards_met': len(compliant_standards),
            'standards_not_met': len(non_compliant_standards),
            'total_issues_identified': total_issues,
            'critical_issues': self.countCriticalIssues(assessment_results)
        }

    def generateCertificationRecommendation(self, assessment_results):
        """
        Generate certification recommendation
        """
        all_compliant = all(result['compliant'] for result in assessment_results.values())

        if all_compliant:
            return {
                'recommendation': 'CERTIFY',
                'confidence': 'HIGH',
                'conditions': 'None'
            }
        else:
            # Count severity of issues
            critical_issues = self.countCriticalIssues(assessment_results)
            major_issues = self.countMajorIssues(assessment_results)

            if critical_issues > 0:
                return {
                    'recommendation': 'DO_NOT_CERTIFY',
                    'confidence': 'LOW',
                    'conditions': 'Critical safety issues must be resolved'
                }
            elif major_issues > 0:
                return {
                    'recommendation': 'CERTIFY_WITH_CONDITIONS',
                    'confidence': 'MEDIUM',
                    'conditions': 'Major issues must be addressed before deployment'
                }
            else:
                return {
                    'recommendation': 'CERTIFY',
                    'confidence': 'MEDIUM',
                    'conditions': 'Minor issues should be monitored'
                }

    def countCriticalIssues(self, assessment_results):
        """
        Count critical safety issues
        """
        # In practice, this would involve detailed risk assessment
        # For now, we'll count issues related to collision, fall, or fire hazards
        critical_keywords = ['collision', 'fall', 'fire', 'burn', 'crush', 'pinch']
        critical_count = 0

        for result in assessment_results.values():
            for issue in result['issues']:
                if any(keyword in issue.lower() for keyword in critical_keywords):
                    critical_count += 1

        return critical_count

    def countMajorIssues(self, assessment_results):
        """
        Count major safety issues
        """
        # Major issues are those that could lead to injury or significant damage
        major_keywords = ['injury', 'damage', 'hazard', 'unsafe', 'malfunction']
        major_count = 0

        for result in assessment_results.values():
            for issue in result['issues']:
                if any(keyword in issue.lower() for keyword in major_keywords):
                    major_count += 1

        return major_count
```

## Implementation in ROS2

### Safety Manager Node

```cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>

class SafetyManagerNode : public rclcpp::Node {
public:
    SafetyManagerNode() : Node("safety_manager") {
        // Publishers and subscribers
        safety_status_pub_ = this->create_publisher<std_msgs::msg::Bool>("safety_status", 10);
        emergency_stop_pub_ = this->create_publisher<std_msgs::msg::Bool>("emergency_stop", 10);
        cmd_vel_filtered_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel_filtered", 10);

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&SafetyManagerNode::jointStateCallback, this, std::placeholders::_1)
        );

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10,
            std::bind(&SafetyManagerNode::cmdVelCallback, this, std::placeholders::_1)
        );

        // Timer for periodic safety checks
        safety_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),  // Check every 10ms
            std::bind(&SafetyManagerNode::safetyCheckTimer, this)
        );

        // Initialize safety systems
        collision_detector_ = std::make_unique<CollisionDetector>();
        balance_monitor_ = std::make_unique<BalanceMonitor>();
        emergency_handler_ = std::make_unique<EmergencyHandler>();

        is_safe_ = true;
        emergency_active_ = false;
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        // Update joint state for safety monitoring
        latest_joint_state_ = *msg;

        // Check for joint limit violations
        if (checkJointLimits(*msg)) {
            triggerEmergencyStop("Joint limit violation");
        }
    }

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // Store latest command for safety filtering
        latest_cmd_vel_ = *msg;

        // Check if command is safe to execute
        if (is_safe_ && !emergency_active_) {
            // Filter command based on safety constraints
            auto filtered_cmd = filterCommand(*msg);
            cmd_vel_filtered_pub_->publish(filtered_cmd);
        } else {
            // Publish zero velocity when unsafe
            geometry_msgs::msg::Twist zero_cmd;
            zero_cmd.linear.x = 0.0;
            zero_cmd.linear.y = 0.0;
            zero_cmd.linear.z = 0.0;
            zero_cmd.angular.x = 0.0;
            zero_cmd.angular.y = 0.0;
            zero_cmd.angular.z = 0.0;

            cmd_vel_filtered_pub_->publish(zero_cmd);
        }
    }

    void safetyCheckTimer() {
        // Perform periodic safety checks
        bool collision_risk = collision_detector_->checkForCollisions();
        bool balance_threat = balance_monitor_->checkBalanceThreat();
        bool system_health_ok = checkSystemHealth();

        bool currently_safe = !collision_risk && !balance_threat && system_health_ok;

        if (!currently_safe && is_safe_) {
            // Safety transition: safe -> unsafe
            RCLCPP_WARN(this->get_logger(), "Safety system transitioned to unsafe state");
            is_safe_ = false;

            // Publish safety status
            std_msgs::msg::Bool safety_status_msg;
            safety_status_msg.data = false;
            safety_status_pub_->publish(safety_status_msg);
        } else if (currently_safe && !is_safe_) {
            // Safety transition: unsafe -> safe
            RCLCPP_INFO(this->get_logger(), "Safety system transitioned to safe state");
            is_safe_ = true;

            // Publish safety status
            std_msgs::msg::Bool safety_status_msg;
            safety_status_msg.data = true;
            safety_status_pub_->publish(safety_status_msg);
        }

        // Update safety status
        std_msgs::msg::Bool status_msg;
        status_msg.data = is_safe_ && !emergency_active_;
        safety_status_pub_->publish(status_msg);
    }

    bool checkJointLimits(const sensor_msgs::msg::JointState& joint_state) {
        // Check if any joint is outside safe limits
        for (size_t i = 0; i < joint_state.name.size(); ++i) {
            std::string joint_name = joint_state.name[i];
            double position = joint_state.position[i];

            auto limit_it = joint_limits_.find(joint_name);
            if (limit_it != joint_limits_.end()) {
                const auto& limits = limit_it->second;
                if (position < limits.min_position || position > limits.max_position) {
                    RCLCPP_WARN(this->get_logger(),
                               "Joint %s exceeded position limits: %f (min: %f, max: %f)",
                               joint_name.c_str(), position, limits.min_position, limits.max_position);
                    return true;  // Joint limit violation
                }
            }
        }
        return false;  // No violations
    }

    geometry_msgs::msg::Twist filterCommand(const geometry_msgs::msg::Twist& cmd) {
        // Filter command to ensure safety
        geometry_msgs::msg::Twist filtered_cmd = cmd;

        // Limit linear velocity based on safety considerations
        double max_linear_vel = (is_safe_) ? 0.5 : 0.1;  // Slower when safety degraded
        double cmd_linear_mag = sqrt(cmd.linear.x * cmd.linear.x + cmd.linear.y * cmd.linear.y);

        if (cmd_linear_mag > max_linear_vel) {
            double scale = max_linear_vel / cmd_linear_mag;
            filtered_cmd.linear.x *= scale;
            filtered_cmd.linear.y *= scale;
        }

        // Limit angular velocity
        double max_angular_vel = (is_safe_) ? 0.5 : 0.2;
        if (fabs(cmd.angular.z) > max_angular_vel) {
            filtered_cmd.angular.z = (cmd.angular.z > 0) ? max_angular_vel : -max_angular_vel;
        }

        return filtered_cmd;
    }

    void triggerEmergencyStop(const std::string& reason) {
        if (!emergency_active_) {
            RCLCPP_ERROR(this->get_logger(), "Emergency stop triggered: %s", reason.c_str());

            emergency_active_ = true;

            // Publish emergency stop command
            std_msgs::msg::Bool estop_msg;
            estop_msg.data = true;
            emergency_stop_pub_->publish(estop_msg);

            // Execute emergency stop sequence
            emergency_handler_->executeEmergencySequence();
        }
    }

    bool checkSystemHealth() {
        // Check various system health indicators
        bool power_ok = checkPowerSystem();
        bool communication_ok = checkCommunicationSystem();
        bool sensor_ok = checkSensorHealth();

        return power_ok && communication_ok && sensor_ok;
    }

    bool checkPowerSystem() {
        // Check power system health
        // This would interface with power monitoring system
        return true;  // Placeholder
    }

    bool checkCommunicationSystem() {
        // Check communication health
        // Verify all critical topics are being published
        return true;  // Placeholder
    }

    bool checkSensorHealth() {
        // Check if safety-critical sensors are functioning
        // Verify joint state publisher is alive
        return true;  // Placeholder
    }

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr safety_status_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr emergency_stop_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_filtered_pub_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;

    rclcpp::TimerBase::SharedPtr safety_timer_;

    std::unique_ptr<CollisionDetector> collision_detector_;
    std::unique_ptr<BalanceMonitor> balance_monitor_;
    std::unique_ptr<EmergencyHandler> emergency_handler_;

    sensor_msgs::msg::JointState latest_joint_state_;
    geometry_msgs::msg::Twist latest_cmd_vel_;

    struct JointLimits {
        double min_position;
        double max_position;
        double max_velocity;
        double max_effort;
    };

    std::map<std::string, JointLimits> joint_limits_;
    bool is_safe_;
    bool emergency_active_;

    // Initialize joint limits
    void initializeJointLimits() {
        // Example joint limits - would be loaded from URDF or config
        JointLimits default_limits;
        default_limits.min_position = -3.14;
        default_limits.max_position = 3.14;
        default_limits.max_velocity = 2.0;
        default_limits.max_effort = 100.0;

        // Add limits for each joint
        joint_limits_["joint1"] = default_limits;
        joint_limits_["joint2"] = default_limits;
        // ... add for all joints
    }
};
```

## Performance Evaluation and Testing

### Safety System Testing

```python
class SafetySystemTester:
    def __init__(self):
        self.test_scenarios = self.loadTestScenarios()
        self.performance_metrics = {
            'response_time': [],
            'false_positive_rate': [],
            'false_negative_rate': [],
            'system_availability': []
        }

    def runComprehensiveSafetyTest(self, safety_system):
        """
        Run comprehensive safety tests
        """
        test_results = {}

        for scenario in self.test_scenarios:
            result = self.runScenarioTest(safety_system, scenario)
            test_results[scenario['name']] = result

        return self.generateTestReport(test_results)

    def runScenarioTest(self, safety_system, scenario):
        """
        Run specific safety scenario test
        """
        scenario_start_time = time.time()

        # Setup scenario conditions
        self.setupScenario(scenario)

        # Inject safety events
        safety_events = scenario['safety_events']
        detected_events = []
        missed_events = []

        for event in safety_events:
            # Inject event at specified time
            if time.time() - scenario_start_time >= event['time_offset']:
                # Check if safety system detects the event
                if safety_system.detectsEvent(event):
                    detected_events.append(event)
                else:
                    missed_events.append(event)

        # Measure response time
        response_times = []
        for event in detected_events:
            detection_time = safety_system.getLastDetectionTime()
            response_time = detection_time - (scenario_start_time + event['time_offset'])
            response_times.append(response_time)

        # Calculate metrics
        total_events = len(safety_events)
        detected_count = len(detected_events)
        missed_count = len(missed_events)

        false_positives = self.countFalsePositives(safety_system, scenario)

        metrics = {
            'detection_rate': detected_count / total_events if total_events > 0 else 0,
            'miss_rate': missed_count / total_events if total_events > 0 else 0,
            'false_positive_rate': false_positives / total_events if total_events > 0 else 0,
            'average_response_time': sum(response_times) / len(response_times) if response_times else float('inf'),
            'max_response_time': max(response_times) if response_times else float('inf'),
            'min_response_time': min(response_times) if response_times else float('inf')
        }

        return metrics

    def setupScenario(self, scenario):
        """
        Setup test scenario conditions
        """
        # Configure robot environment
        self.configureEnvironment(scenario['environment'])

        # Set robot initial state
        self.setRobotInitialState(scenario['initial_state'])

        # Configure safety system parameters
        self.configureSafetySystem(scenario['safety_config'])

    def countFalsePositives(self, safety_system, scenario):
        """
        Count false positive detections during scenario
        """
        # During periods where no safety events are expected,
        # count any safety system activations as false positives
        false_positive_count = 0

        # This would involve monitoring the system during safe periods
        # and counting unwarranted safety activations

        return false_positive_count

    def loadTestScenarios(self):
        """
        Load predefined safety test scenarios
        """
        scenarios = [
            {
                'name': 'collision_with_human',
                'description': 'Robot approaches human unexpectedly',
                'environment': {'humans_present': True, 'obstacles': []},
                'initial_state': {'position': [0, 0, 0], 'velocity': [0.2, 0, 0]},
                'safety_events': [
                    {'type': 'proximity', 'time_offset': 2.0, 'parameters': {'distance': 0.3}},
                    {'type': 'contact', 'time_offset': 2.5, 'parameters': {'force': 50}}
                ],
                'safety_config': {'collision_threshold': 0.5, 'contact_force_limit': 100}
            },
            {
                'name': 'balance_loss_recovery',
                'description': 'Robot loses balance and attempts recovery',
                'environment': {'surface': 'slippery', 'slope': 0.1},
                'initial_state': {'position': [0, 0, 0], 'com_offset': [0.1, 0.05, 0]},
                'safety_events': [
                    {'type': 'imbalance', 'time_offset': 1.0, 'parameters': {'angle': 15}},
                    {'type': 'zmp_violation', 'time_offset': 1.2, 'parameters': {'deviation': 0.15}}
                ],
                'safety_config': {'balance_threshold': 10, 'zmp_margin': 0.05}
            },
            {
                'name': 'emergency_stop_response',
                'description': 'Test emergency stop button functionality',
                'environment': {'emergency_button': True},
                'initial_state': {'motion_state': 'moving', 'speed': 0.3},
                'safety_events': [
                    {'type': 'emergency_stop', 'time_offset': 1.5, 'parameters': {'button_pressed': True}}
                ],
                'safety_config': {'stop_distance': 0.1, 'stop_time': 0.5}
            }
        ]

        return scenarios

    def generateTestReport(self, test_results):
        """
        Generate comprehensive test report
        """
        report = {
            'timestamp': time.time(),
            'test_summary': self.calculateTestSummary(test_results),
            'detailed_results': test_results,
            'recommendations': self.generateRecommendations(test_results),
            'compliance_status': self.calculateCompliance(test_results)
        }

        return report

    def calculateTestSummary(self, test_results):
        """
        Calculate summary statistics for test results
        """
        total_scenarios = len(test_results)
        passed_scenarios = 0
        overall_detection_rate = 0
        overall_response_time = 0

        for scenario_name, results in test_results.items():
            # Scenario passes if detection rate > 95% and response time < 100ms
            if (results['detection_rate'] > 0.95 and
                results['average_response_time'] < 0.1):
                passed_scenarios += 1

            overall_detection_rate += results['detection_rate']
            overall_response_time += results['average_response_time']

        return {
            'total_scenarios': total_scenarios,
            'passed_scenarios': passed_scenarios,
            'pass_rate': passed_scenarios / total_scenarios if total_scenarios > 0 else 0,
            'average_detection_rate': overall_detection_rate / total_scenarios if total_scenarios > 0 else 0,
            'average_response_time': overall_response_time / total_scenarios if total_scenarios > 0 else float('inf')
        }

    def generateRecommendations(self, test_results):
        """
        Generate recommendations based on test results
        """
        recommendations = []

        for scenario_name, results in test_results.items():
            if results['detection_rate'] < 0.95:
                recommendations.append(f"Improve detection sensitivity for {scenario_name}")

            if results['average_response_time'] > 0.1:
                recommendations.append(f"Reduce response time for {scenario_name}")

            if results['false_positive_rate'] > 0.05:
                recommendations.append(f"Tune false positive rate for {scenario_name}")

        return recommendations

    def calculateCompliance(self, test_results):
        """
        Calculate compliance with safety standards
        """
        # Check compliance with various metrics
        metrics_compliance = {
            'detection_rate': all(results['detection_rate'] >= 0.95 for results in test_results.values()),
            'response_time': all(results['average_response_time'] <= 0.1 for results in test_results.values()),
            'false_positive_rate': all(results['false_positive_rate'] <= 0.05 for results in test_results.values())
        }

        overall_compliance = all(metrics_compliance.values())

        return {
            'overall_compliance': overall_compliance,
            'metric_compliance': metrics_compliance,
            'standards_met': ['ISO_13482', 'IEC_61508'] if overall_compliance else []
        }
```

## Troubleshooting and Maintenance

### Safety System Diagnostics

```cpp
class SafetyDiagnostics {
public:
    struct DiagnosticReport {
        std::string component_name;
        std::string status;  // "OK", "WARNING", "ERROR", "CRITICAL"
        std::vector<std::string> issues;
        std::vector<std::string> recommendations;
        double confidence;
        std::string timestamp;
    };

    std::vector<DiagnosticReport> runDiagnostics() {
        std::vector<DiagnosticReport> reports;

        reports.push_back(checkEmergencyStopSystem());
        reports.push_back(checkCollisionAvoidanceSystem());
        reports.push_back(checkBalanceControlSystem());
        reports.push_back(checkCommunicationSystem());
        reports.push_back(checkPowerSystem());

        return reports;
    }

    DiagnosticReport checkEmergencyStopSystem() {
        DiagnosticReport report;
        report.component_name = "Emergency Stop System";
        report.timestamp = getCurrentTimestamp();

        bool estop_functional = testEmergencyStopButton();
        bool estop_wired_correctly = verifyEmergencyStopWiring();
        bool estop_response_time_acceptable = measureEmergencyStopResponseTime();

        if (estop_functional && estop_wired_correctly && estop_response_time_acceptable) {
            report.status = "OK";
        } else {
            report.status = "CRITICAL";

            if (!estop_functional) {
                report.issues.push_back("Emergency stop button not responding");
                report.recommendations.push_back("Check button wiring and connections");
            }

            if (!estop_wired_correctly) {
                report.issues.push_back("Emergency stop wiring incorrect");
                report.recommendations.push_back("Verify wiring diagram and connections");
            }

            if (!estop_response_time_acceptable) {
                report.issues.push_back("Emergency stop response time too slow");
                report.recommendations.push_back("Check system latency and optimize");
            }
        }

        report.confidence = 0.95;  // High confidence in this test
        return report;
    }

    DiagnosticReport checkCollisionAvoidanceSystem() {
        DiagnosticReport report;
        report.component_name = "Collision Avoidance System";
        report.timestamp = getCurrentTimestamp();

        bool sensors_functional = testAllDistanceSensors();
        bool algorithm_working = verifyCollisionAlgorithm();
        bool response_appropriate = testCollisionResponse();

        if (sensors_functional && algorithm_working && response_appropriate) {
            report.status = "OK";
        } else {
            report.status = "ERROR";

            if (!sensors_functional) {
                report.issues.push_back("One or more distance sensors not functional");
                report.recommendations.push_back("Calibrate or replace malfunctioning sensors");
            }

            if (!algorithm_working) {
                report.issues.push_back("Collision detection algorithm not working correctly");
                report.recommendations.push_back("Verify algorithm parameters and update if needed");
            }

            if (!response_appropriate) {
                report.issues.push_back("Collision response inappropriate");
                report.recommendations.push_back("Adjust response parameters");
            }
        }

        report.confidence = 0.85;  // Moderate confidence
        return report;
    }

    DiagnosticReport checkBalanceControlSystem() {
        DiagnosticReport report;
        report.component_name = "Balance Control System";
        report.timestamp = getCurrentTimestamp();

        bool sensors_ok = checkBalanceSensors();
        bool controller_stable = verifyBalanceController();
        bool recovery_working = testBalanceRecovery();

        if (sensors_ok && controller_stable && recovery_working) {
            report.status = "OK";
        } else {
            report.status = "WARNING";

            if (!sensors_ok) {
                report.issues.push_back("Balance sensors not calibrated properly");
                report.recommendations.push_back("Recalibrate IMU and force sensors");
            }

            if (!controller_stable) {
                report.issues.push_back("Balance controller showing instability");
                report.recommendations.push_back("Tune controller parameters");
            }

            if (!recovery_working) {
                report.issues.push_back("Balance recovery not functioning");
                report.recommendations.push_back("Test and verify recovery algorithms");
            }
        }

        report.confidence = 0.80;  // Moderate confidence
        return report;
    }

    void generateMaintenanceSchedule(const std::vector<DiagnosticReport>& reports) {
        // Generate maintenance recommendations based on diagnostic results
        for (const auto& report : reports) {
            if (report.status == "CRITICAL") {
                scheduleImmediateMaintenance(report.component_name);
            } else if (report.status == "ERROR") {
                scheduleMaintenance(report.component_name, "within 24 hours");
            } else if (report.status == "WARNING") {
                scheduleMaintenance(report.component_name, "within 1 week");
            }
        }
    }

private:
    bool testEmergencyStopButton() {
        // Test that emergency stop button can be pressed and detected
        return true;  // Placeholder
    }

    bool verifyEmergencyStopWiring() {
        // Verify emergency stop circuit is properly wired
        return true;  // Placeholder
    }

    bool measureEmergencyStopResponseTime() {
        // Measure time from button press to system stop
        return true;  // Placeholder
    }

    bool testAllDistanceSensors() {
        // Test all distance sensors for proper operation
        return true;  // Placeholder
    }

    bool verifyCollisionAlgorithm() {
        // Verify collision detection algorithm works correctly
        return true;  // Placeholder
    }

    bool testCollisionResponse() {
        // Test that collision response is appropriate
        return true;  // Placeholder
    }

    bool checkBalanceSensors() {
        // Check balance-related sensors (IMU, force sensors, etc.)
        return true;  // Placeholder
    }

    bool verifyBalanceController() {
        // Verify balance controller stability
        return true;  // Placeholder
    }

    bool testBalanceRecovery() {
        // Test balance recovery algorithms
        return true;  // Placeholder
    }

    void scheduleImmediateMaintenance(const std::string& component) {
        // Schedule immediate maintenance for critical components
    }

    void scheduleMaintenance(const std::string& component, const std::string& timeframe) {
        // Schedule maintenance within specified timeframe
    }

    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        return std::ctime(&time_t);
    }
};
```

## Future Trends and Considerations

### AI-Enhanced Safety Systems

```python
class AIEnhancedSafetySystem:
    def __init__(self):
        self.anomaly_detector = AnomalyDetectionAI()
        self.predictive_safety_model = PredictiveSafetyModel()
        self.adaptive_safety_controller = AdaptiveSafetyController()

    def enhancedSafetyCheck(self, robot_state, environment_state):
        """
        Perform AI-enhanced safety check
        """
        # Traditional safety check
        traditional_safety_ok = self.traditionalSafetyCheck(robot_state)

        # AI-based anomaly detection
        anomalies = self.anomaly_detector.detectAnomalies(robot_state, environment_state)

        # Predictive safety assessment
        future_risks = self.predictive_safety_model.assessFutureRisks(robot_state, environment_state)

        # Adaptive safety response
        safety_response = self.adaptive_safety_controller.determineResponse(
            traditional_safety_ok, anomalies, future_risks
        )

        return {
            'safe': safety_response['safe'],
            'confidence': safety_response['confidence'],
            'detected_anomalies': anomalies,
            'predicted_risks': future_risks,
            'recommended_action': safety_response['action']
        }

    def traditionalSafetyCheck(self, robot_state):
        """
        Traditional safety check (collision, balance, etc.)
        """
        # Implementation of traditional safety checks
        return True  # Placeholder

class AnomalyDetectionAI:
    def __init__(self):
        # Load pre-trained anomaly detection model
        self.model = self.loadModel()

    def detectAnomalies(self, robot_state, environment_state):
        """
        Detect anomalous patterns that could indicate safety issues
        """
        # Preprocess state data
        features = self.preprocessState(robot_state, environment_state)

        # Run anomaly detection
        anomaly_scores = self.model.predict(features)

        # Identify anomalies above threshold
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score > self.anomaly_threshold:
                anomalies.append({
                    'type': self.getAnomalyType(i),
                    'severity': score,
                    'description': self.getAnomalyDescription(i)
                })

        return anomalies

    def loadModel(self):
        """
        Load pre-trained anomaly detection model
        """
        # This would load a trained model (e.g., Isolation Forest, Autoencoder)
        return None  # Placeholder

    def preprocessState(self, robot_state, environment_state):
        """
        Preprocess state data for anomaly detection
        """
        # Extract relevant features from state
        features = []

        # Robot state features
        features.extend(robot_state.get('joint_positions', []))
        features.extend(robot_state.get('joint_velocities', []))
        features.extend(robot_state.get('accelerations', []))

        # Environmental features
        features.extend(environment_state.get('obstacle_distances', []))
        features.extend(environment_state.get('surface_properties', []))

        return features

    def getAnomalyType(self, feature_index):
        """
        Get anomaly type based on feature index
        """
        # Map feature index to anomaly type
        return "unknown"  # Placeholder

    def getAnomalyDescription(self, feature_index):
        """
        Get human-readable description of anomaly
        """
        return "Anomalous pattern detected"  # Placeholder

    anomaly_threshold = 0.7
```

## Conclusion

Safety, fail-safes, and edge computing form the critical foundation for deploying humanoid robots in real-world environments. The systems described in this chapter provide multiple layers of protection, from low-level hardware safety circuits to high-level AI-enhanced monitoring systems.

The key to successful safety implementation lies in:
1. **Defense in depth**: Multiple independent safety layers
2. **Real-time performance**: Edge computing for immediate response
3. **Graceful degradation**: Safe operation even when components fail
4. **Continuous monitoring**: Proactive detection of potential issues
5. **Standards compliance**: Adherence to relevant safety standards

As humanoid robots become more prevalent, safety systems will need to evolve to handle increasingly complex scenarios while maintaining the highest levels of protection for both the robots and humans in their environment.

The next chapter will explore the capstone project guide, bringing together all the concepts covered in this book to create a comprehensive humanoid robotics project.

## Exercises

1. Design and implement a redundant emergency stop system with voting logic.

2. Create a balance monitoring system that predicts falls and initiates recovery actions.

3. Implement a collision detection system using distance sensors and force/torque measurements.

4. Design an edge computing architecture for real-time safety processing with guaranteed deadlines.

5. Develop a safety certification plan for a humanoid robot project, including test scenarios and documentation requirements.