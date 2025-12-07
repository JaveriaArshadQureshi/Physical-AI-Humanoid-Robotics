---
title: "Chapter 12: Imitation Learning + Teleoperation"
description: "Learning from expert demonstrations and human control"
---

# Chapter 12: Imitation Learning + Teleoperation

## Overview

Imitation learning and teleoperation provide powerful approaches for humanoid robots to acquire complex behaviors by learning from human demonstrations or by directly following human control. This chapter explores how robots can learn from expert demonstrations, the technologies enabling effective teleoperation, and the integration of these approaches with autonomous systems.

## Introduction to Imitation Learning

### What is Imitation Learning?

Imitation learning, also known as learning from demonstration, is a machine learning paradigm where an agent learns to perform tasks by observing and mimicking expert demonstrations. Unlike reinforcement learning, which learns through trial-and-error, imitation learning leverages existing knowledge from experts.

### Key Advantages

#### Sample Efficiency
- Direct learning from expert demonstrations
- No need for reward engineering
- Faster learning compared to RL in many cases

#### Safe Learning
- Expert demonstrations are typically safe
- No dangerous exploration required
- Learning from successful examples only

#### Natural Task Specification
- Tasks specified through demonstration rather than reward functions
- Complex behaviors can be taught intuitively
- Human knowledge transfer to robots

### Problem Formulation

Imitation learning addresses the problem of learning a policy `π(a|s)` from expert demonstrations `D = {(s_1, a_1), (s_2, a_2), ..., (s_n, a_n)}`.

The goal is to minimize the difference between the expert policy `π_e` and the learned policy `π`:
`min_π E[||π(a|s) - π_e(a|s)||]`

## Behavioral Cloning

### Basic Approach

Behavioral cloning treats imitation learning as a supervised learning problem:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class BehavioralCloning(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(BehavioralCloning, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class BehavioralCloningAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = BehavioralCloning(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, states, actions, epochs=100, batch_size=64):
        dataset = TensorDataset(
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()

                predicted_actions = self.network(batch_states)
                loss = self.criterion(predicted_actions, batch_actions)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def predict(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.network(state_tensor).squeeze(0)
        return action.cpu().numpy()
```

### Limitations of Behavioral Cloning

#### Covariate Shift
- Training on expert states, testing on agent states
- Agent drifts from expert distribution
- Performance degrades over time

#### Error Accumulation
- Small errors compound over time
- Agent moves to states not seen in demonstrations
- Catastrophic failure in long sequences

### Addressing Limitations

#### Data Aggregation (DAgger)
```python
class DAggerAgent:
    def __init__(self, state_dim, action_dim, expert_policy):
        self.bc_agent = BehavioralCloningAgent(state_dim, action_dim)
        self.expert_policy = expert_policy
        self.all_states = []
        self.all_actions = []

    def train_dagger(self, env, num_iterations=10, episodes_per_iter=10):
        for iteration in range(num_iterations):
            # Collect trajectories using current policy
            states, actions = self.collect_trajectories(env, episodes_per_iter)

            # Get expert actions for these states
            expert_actions = [self.expert_policy(state) for state in states]

            # Add to training dataset
            self.all_states.extend(states)
            self.all_actions.extend(expert_actions)

            # Retrain behavioral cloning agent
            self.bc_agent.train(self.all_states, self.all_actions, epochs=50)

            print(f"DAgger iteration {iteration+1} completed")

    def collect_trajectories(self, env, num_episodes):
        all_states = []
        all_actions = []

        for _ in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                # Use current learned policy to collect states
                action = self.bc_agent.predict(state)
                all_states.append(state)

                # Store the action taken (though we'll replace with expert action)
                obs, reward, done, info = env.step(action)
                state = obs

        return all_states, all_actions
```

## Advanced Imitation Learning Methods

### Generative Adversarial Imitation Learning (GAIL)

GAIL uses adversarial training to match the expert's state-action distribution:

```python
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class GAILAgent:
    def __init__(self, state_dim, action_dim, expert_dataset, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = BehavioralCloning(state_dim, action_dim).to(self.device)
        self.discriminator = Discriminator(state_dim, action_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

        self.expert_dataset = expert_dataset
        self.criterion = nn.BCELoss()

    def train_step(self, states, actions):
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)

        # Sample expert transitions
        expert_batch = self.sample_expert_batch(len(states))
        expert_states = torch.FloatTensor(expert_batch['states']).to(self.device)
        expert_actions = torch.FloatTensor(expert_batch['actions']).to(self.device)

        # Train discriminator
        self.disc_optimizer.zero_grad()

        # Discriminator loss: expert = 1, agent = 0
        expert_logits = self.discriminator(expert_states, expert_actions)
        agent_logits = self.discriminator(states, actions)

        disc_loss = (self.criterion(agent_logits, torch.zeros_like(agent_logits)) +
                    self.criterion(expert_logits, torch.ones_like(expert_logits))) / 2

        disc_loss.backward()
        self.disc_optimizer.step()

        # Train policy using discriminator reward
        self.policy_optimizer.zero_grad()

        # Compute discriminator-based reward for policy
        agent_logits = self.discriminator(states, actions)
        policy_loss = -torch.mean(torch.log(agent_logits + 1e-8))

        policy_loss.backward()
        self.policy_optimizer.step()

        return disc_loss.item(), policy_loss.item()

    def sample_expert_batch(self, batch_size):
        # Sample from expert dataset
        indices = torch.randperm(len(self.expert_dataset))[:batch_size]
        batch = self.expert_dataset[indices]
        return {'states': batch['states'], 'actions': batch['actions']}

    def get_reward(self, state, action):
        # Use discriminator to provide reward
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob_expert = self.discriminator(state_tensor, action_tensor)
            # Convert to reward (log probability)
            reward = torch.log(prob_expert + 1e-8).item()

        return reward
```

### Inverse Reinforcement Learning (IRL)

IRL learns the reward function from expert demonstrations:

```python
class MaxEntIRL(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MaxEntIRL, self).__init__()

        # Reward network
        self.reward_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.reward_network(x)

class MaxEntIRLAgent:
    def __init__(self, state_dim, action_dim, expert_dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reward_net = MaxEntIRL(state_dim, action_dim).to(self.device)
        self.policy = BehavioralCloning(state_dim, action_dim).to(self.device)

        self.reward_optimizer = optim.Adam(self.reward_net.parameters(), lr=1e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

        self.expert_dataset = expert_dataset

    def train(self, env, num_iterations=1000):
        for iteration in range(num_iterations):
            # Update reward function to make expert better than agent
            self.update_reward_function()

            # Update policy using current reward function
            self.update_policy_with_reward(env)

            if iteration % 100 == 0:
                print(f"IRL iteration {iteration}")

    def update_reward_function(self):
        # This is a simplified version - full implementation requires
        # solving the MDP for the current reward function
        pass

    def update_policy_with_reward(self, env):
        # Train policy using current reward function
        # This would typically involve running RL with the learned reward
        pass
```

## Teleoperation Systems

### Teleoperation Fundamentals

Teleoperation allows humans to control robots remotely, providing direct control while maintaining safety and precision:

#### System Components
- **Master Device**: Human input device (joystick, haptic device, exoskeleton)
- **Communication Channel**: Network connection between master and slave
- **Slave Robot**: The remote robot being controlled
- **Feedback System**: Visual, auditory, and haptic feedback to operator

### Haptic Feedback in Teleoperation

```python
import numpy as np

class HapticTeleoperation:
    def __init__(self, master_device, slave_robot):
        self.master = master_device
        self.slave = slave_robot
        self.communication_delay = 0.1  # 100ms delay
        self.stability_margin = 0.8

    def run_teleoperation_loop(self):
        while True:
            # Get master position and forces
            master_pos = self.master.get_position()
            master_forces = self.master.get_forces()

            # Send command to slave robot (with delay simulation)
            delayed_command = self.add_communication_delay(master_pos)

            # Execute on slave robot
            slave_response = self.slave.execute_command(delayed_command)

            # Get environmental forces on slave
            slave_forces = self.slave.get_environmental_forces()

            # Apply force feedback to master (with stability compensation)
            compensated_forces = self.stability_compensation(slave_forces)
            self.master.apply_force_feedback(compensated_forces)

    def add_communication_delay(self, position):
        # Simulate network delay
        # In practice, this would involve buffering and prediction
        return position

    def stability_compensation(self, forces):
        # Compensate for potential instability due to delays
        return forces * self.stability_margin
```

### Bilateral Control

Bilateral teleoperation provides force feedback to the operator:

```python
class BilateralTeleoperation:
    def __init__(self, position_gain=1.0, force_gain=0.5):
        self.Kp = position_gain  # Position scaling
        self.Kf = force_gain     # Force scaling
        self.prev_error = 0
        self.integral_error = 0

    def compute_control_signals(self, master_pos, slave_pos, desired_pos, contact_force):
        # Position error
        pos_error = desired_pos - slave_pos

        # PID controller for position tracking
        proportional = pos_error
        self.integral_error += pos_error
        derivative = pos_error - self.prev_error

        control_signal = (proportional * 1.0 +
                         self.integral_error * 0.1 +
                         derivative * 0.05)

        self.prev_error = pos_error

        # Force feedback to master
        force_feedback = self.Kf * contact_force

        return control_signal, force_feedback
```

## Human-Robot Interface Technologies

### Exoskeleton-Based Teleoperation

```python
class ExoskeletonTeleoperation:
    def __init__(self, exoskeleton_suit, humanoid_robot):
        self.exoskeleton = exoskeleton_suit
        self.robot = humanoid_robot
        self.mapping_matrix = self.initialize_mapping()

    def initialize_mapping(self):
        # Create mapping between exoskeleton joints and robot joints
        # This could be a learned mapping or predefined based on anatomy
        mapping = np.eye(self.exoskeleton.num_joints, self.robot.num_joints)
        # Adjust for different kinematic structures
        return mapping

    def run_mapping_loop(self):
        while True:
            # Read exoskeleton joint positions
            exo_positions = self.exoskeleton.get_joint_positions()
            exo_velocities = self.exoskeleton.get_joint_velocities()

            # Map to robot joint space
            robot_positions = self.map_positions(exo_positions)
            robot_velocities = self.map_velocities(exo_velocities)

            # Execute on robot
            self.robot.set_joint_positions(robot_positions)
            self.robot.set_joint_velocities(robot_velocities)

    def map_positions(self, exo_pos):
        # Apply learned or predefined mapping
        robot_pos = np.dot(self.mapping_matrix, exo_pos)
        return robot_pos

    def map_velocities(self, exo_vel):
        # Apply same mapping to velocities
        robot_vel = np.dot(self.mapping_matrix, exo_vel)
        return robot_vel
```

### Vision-Based Teleoperation

```python
import cv2
import numpy as np

class VisionBasedTeleoperation:
    def __init__(self, camera_system, robot_arm):
        self.camera = camera_system
        self.robot = robot_arm
        self.calibration_matrix = self.calibrate_camera_robot()

    def calibrate_camera_robot(self):
        # Perform hand-eye calibration
        # This establishes relationship between camera and robot coordinate frames
        return np.eye(4)  # Placeholder for actual calibration matrix

    def teleoperate_with_visual_feedback(self):
        while True:
            # Get camera image
            image = self.camera.capture()

            # Detect target object
            target_pos_2d = self.detect_target(image)

            # Convert 2D image coordinates to 3D world coordinates
            target_pos_3d = self.image_to_world(target_pos_2d)

            # Transform to robot base frame
            robot_target = self.transform_to_robot_frame(target_pos_3d)

            # Move robot to target
            self.robot.move_to(robot_target)

    def detect_target(self, image):
        # Use computer vision to detect target object
        # This could use feature detection, color-based detection, etc.
        # Return 2D coordinates of target in image
        return np.array([100, 100])  # Placeholder

    def image_to_world(self, pixel_coords):
        # Convert pixel coordinates to 3D world coordinates
        # Use camera intrinsic parameters and depth information
        # This might involve stereo vision or depth sensor
        return np.array([0.5, 0.2, 0.8])  # Placeholder

    def transform_to_robot_frame(self, world_coords):
        # Apply calibration matrix to transform to robot coordinate frame
        world_homogeneous = np.append(world_coords, 1)
        robot_homogeneous = np.dot(self.calibration_matrix, world_homogeneous)
        return robot_homogeneous[:3]
```

## Learning from Teleoperation Data

### Data Collection Pipeline

```python
class TeleoperationDataCollector:
    def __init__(self, robot, teleoperation_interface):
        self.robot = robot
        self.teleop = teleoperation_interface
        self.data_buffer = []
        self.max_buffer_size = 10000

    def collect_demonstration(self, task_name, max_steps=1000):
        print(f"Collecting demonstration for task: {task_name}")

        episode_data = []
        state = self.robot.get_state()

        for step in range(max_steps):
            # Get action from teleoperation interface
            action = self.teleop.get_action()

            # Store state-action pair
            transition = {
                'state': state.copy(),
                'action': action.copy(),
                'timestamp': self.get_timestamp(),
                'task': task_name
            }

            episode_data.append(transition)

            # Execute action
            self.robot.execute_action(action)
            next_state = self.robot.get_state()

            # Check termination condition
            if self.is_task_complete(next_state, task_name):
                break

            state = next_state

        # Add episode to dataset
        self.add_episode_to_buffer(episode_data)
        print(f"Collected {len(episode_data)} transitions")

    def add_episode_to_buffer(self, episode_data):
        self.data_buffer.extend(episode_data)

        # Maintain buffer size limit
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]

    def get_training_data(self):
        # Convert collected data to training format
        states = np.array([t['state'] for t in self.data_buffer])
        actions = np.array([t['action'] for t in self.data_buffer])

        return states, actions

    def save_demonstrations(self, filename):
        # Save collected demonstrations to file
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.data_buffer, f)
        print(f"Saved {len(self.data_buffer)} transitions to {filename}")
```

### Data Preprocessing for Imitation Learning

```python
class DemonstrationPreprocessor:
    def __init__(self):
        self.normalization_stats = {}

    def preprocess_demonstrations(self, states, actions):
        # Normalize states and actions
        normalized_states = self.normalize_states(states)
        normalized_actions = self.normalize_actions(actions)

        # Filter noisy data
        filtered_states, filtered_actions = self.filter_data(normalized_states, normalized_actions)

        # Add temporal features
        augmented_states = self.add_temporal_features(filtered_states)

        return augmented_states, filtered_actions

    def normalize_states(self, states):
        if 'state_mean' not in self.normalization_stats:
            self.normalization_stats['state_mean'] = np.mean(states, axis=0)
            self.normalization_stats['state_std'] = np.std(states, axis=0)

        mean = self.normalization_stats['state_mean']
        std = self.normalization_stats['state_std']

        return (states - mean) / (std + 1e-8)

    def normalize_actions(self, actions):
        if 'action_min' not in self.normalization_stats:
            self.normalization_stats['action_min'] = np.min(actions, axis=0)
            self.normalization_stats['action_max'] = np.max(actions, axis=0)

        min_val = self.normalization_stats['action_min']
        max_val = self.normalization_stats['action_max']

        # Normalize to [-1, 1] range
        return 2 * (actions - min_val) / (max_val - min_val + 1e-8) - 1

    def filter_data(self, states, actions):
        # Remove outliers and noisy data
        # Use velocity and acceleration constraints
        valid_indices = []

        for i in range(1, len(states) - 1):
            # Check action smoothness
            vel_change = np.abs(actions[i] - actions[i-1])
            acc_change = np.abs(actions[i+1] - 2*actions[i] + actions[i-1])

            if np.all(vel_change < 2.0) and np.all(acc_change < 5.0):  # Thresholds
                valid_indices.append(i)

        return states[valid_indices], actions[valid_indices]

    def add_temporal_features(self, states):
        # Add velocity and acceleration features
        velocities = np.diff(states, axis=0, append=states[-1:])
        accelerations = np.diff(velocities, axis=0, append=velocities[-1:])

        # Concatenate temporal features
        temporal_states = np.concatenate([states, velocities, accelerations], axis=1)
        return temporal_states
```

## Human-in-the-Loop Learning

### Interactive Imitation Learning

```python
class InteractiveImitationLearning:
    def __init__(self, agent, human_expert):
        self.agent = agent
        self.expert = human_expert
        self.correction_buffer = []
        self.uncertainty_threshold = 0.1

    def run_interactive_learning(self, env, max_episodes=100):
        for episode in range(max_episodes):
            state = env.reset()
            done = False
            step = 0

            while not done:
                # Agent acts
                agent_action = self.agent.act(state)

                # Check if agent is uncertain
                uncertainty = self.agent.get_uncertainty(state)

                if uncertainty > self.uncertainty_threshold:
                    # Request human correction
                    human_action = self.request_correction(state, agent_action)
                    self.store_correction(state, agent_action, human_action)

                    # Use human action for this step
                    action = human_action
                else:
                    # Use agent action
                    action = agent_action

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Update agent with this experience
                self.agent.update(state, action, reward, next_state, done)

                state = next_state
                step += 1

    def request_correction(self, state, agent_action):
        print("Agent uncertain, requesting human correction...")
        return self.expert.provide_correction(state, agent_action)

    def store_correction(self, state, agent_action, human_action):
        correction = {
            'state': state,
            'agent_action': agent_action,
            'human_action': human_action,
            'timestamp': self.get_timestamp()
        }
        self.correction_buffer.append(correction)

    def update_from_corrections(self):
        # Retrain agent using correction data
        if len(self.correction_buffer) > 100:  # Minimum batch size
            states = np.array([c['state'] for c in self.correction_buffer])
            actions = np.array([c['human_action'] for c in self.correction_buffer])

            self.agent.retrain_with_corrections(states, actions)
            self.correction_buffer = []  # Clear buffer after use
```

### Preference-Based Learning

```python
class PreferenceBasedLearning:
    def __init__(self, agent):
        self.agent = agent
        self.preference_buffer = []

    def collect_preference(self, state, action1, action2):
        # Present two actions to human and get preference
        preference = self.human_evaluate_actions(state, action1, action2)

        preference_data = {
            'state': state,
            'action1': action1,
            'action2': action2,
            'preferred': preference  # 0 for action1, 1 for action2
        }

        self.preference_buffer.append(preference_data)

        # Update reward model from preferences
        self.update_reward_model()

    def update_reward_model(self):
        # Learn reward function from preferences
        # This could use algorithms like DPO (Direct Preference Optimization)
        pass

    def human_evaluate_actions(self, state, action1, action2):
        # Simulate human evaluation (in practice, this would be actual human input)
        # For demonstration, we'll use a simple heuristic
        reward1 = self.estimate_action_reward(state, action1)
        reward2 = self.estimate_action_reward(state, action2)

        return 0 if reward1 > reward2 else 1

    def estimate_action_reward(self, state, action):
        # Simple reward estimation (would be learned in practice)
        return np.random.random()  # Placeholder
```

## Safety Considerations

### Safe Imitation Learning

```python
class SafeImitationLearning:
    def __init__(self, base_agent):
        self.agent = base_agent
        self.safety_checker = SafetyModule()
        self.fallback_controller = FallbackController()

    def safe_act(self, state):
        # Get action from learned policy
        learned_action = self.agent.act(state)

        # Check if action is safe
        if self.safety_checker.is_safe(state, learned_action):
            return learned_action
        else:
            # Use fallback controller
            safe_action = self.fallback_controller.get_safe_action(state)
            print("Safety override: using fallback controller")
            return safe_action

class SafetyModule:
    def __init__(self):
        self.joint_limits = {'min': -2.0, 'max': 2.0}
        self.velocity_limits = {'max': 1.0}
        self.collision_threshold = 0.1

    def is_safe(self, state, action):
        # Check joint limits
        joint_positions = state['joint_positions'] + action * 0.01  # Assuming small time step
        if np.any(joint_positions < self.joint_limits['min']) or \
           np.any(joint_positions > self.joint_limits['max']):
            return False

        # Check velocity limits
        joint_velocities = action  # Simplified
        if np.any(np.abs(joint_velocities) > self.velocity_limits['max']):
            return False

        # Check for potential collisions
        if self.would_collide(state, action):
            return False

        return True

    def would_collide(self, state, action):
        # Predict next state and check for collisions
        # This would involve forward kinematics and collision detection
        return False  # Placeholder
```

### Shared Autonomy

```python
class SharedAutonomy:
    def __init__(self, robot_agent, human_interface):
        self.robot_agent = robot_agent
        self.human_interface = human_interface
        self.autonomy_level = 0.5  # 0 = full human control, 1 = full autonomy

    def get_blended_action(self, state):
        # Get robot's suggested action
        robot_action = self.robot_agent.act(state)

        # Get human's intended action
        human_action = self.human_interface.get_intended_action(state)

        # Blend actions based on autonomy level
        blended_action = (self.autonomy_level * robot_action +
                         (1 - self.autonomy_level) * human_action)

        return blended_action

    def update_autonomy_level(self, task_context, human_skill_level):
        # Adjust autonomy level based on task difficulty and human capability
        if task_context == 'difficult' or human_skill_level == 'low':
            self.autonomy_level = 0.8  # Higher robot autonomy
        else:
            self.autonomy_level = 0.3  # Higher human control
```

## Integration with Autonomous Systems

### Hierarchical Control Architecture

```python
class HierarchicalImitationController:
    def __init__(self):
        self.high_level_planner = HighLevelPlanner()
        self.mid_level_controller = ImitationController()
        self.low_level_controller = LowLevelController()

    def execute_task(self, task_goal, current_state):
        # High-level: plan sequence of subtasks
        subtasks = self.high_level_planner.plan_subtasks(task_goal, current_state)

        for subtask in subtasks:
            # Mid-level: execute subtask using imitation learning
            success = self.execute_subtask_with_imitation(subtask, current_state)

            if not success:
                # Fallback to teleoperation or manual control
                success = self.request_human_assistance(subtask, current_state)

            if not success:
                # Emergency stop
                self.emergency_stop()
                break

    def execute_subtask_with_imitation(self, subtask, state):
        # Use learned policy to execute subtask
        return self.mid_level_controller.execute_with_imitation(subtask, state)

    def request_human_assistance(self, subtask, state):
        # Switch to teleoperation mode
        return self.human_interface.teleoperate_subtask(subtask, state)
```

## Evaluation and Metrics

### Imitation Learning Performance

```python
class ImitationLearningEvaluator:
    def __init__(self, expert_policy):
        self.expert_policy = expert_policy

    def evaluate_policy(self, learned_policy, env, num_episodes=10):
        metrics = {
            'average_return': [],
            'expert_action_accuracy': [],
            'state_distribution_similarity': [],
            'task_completion_rate': 0
        }

        total_completed = 0

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_return = 0
            action_matches = 0
            total_actions = 0

            while not done:
                # Get actions from both policies
                learned_action = learned_policy.act(state)
                expert_action = self.expert_policy.act(state)

                # Execute learned action
                next_state, reward, done, info = env.step(learned_action)

                # Compare actions
                if np.allclose(learned_action, expert_action, atol=0.1):
                    action_matches += 1
                total_actions += 1

                episode_return += reward
                state = next_state

            # Record metrics
            metrics['average_return'].append(episode_return)
            metrics['expert_action_accuracy'].append(action_matches / total_actions)

            if info.get('task_completed', False):
                total_completed += 1

        # Compute final metrics
        metrics['average_return'] = np.mean(metrics['average_return'])
        metrics['expert_action_accuracy'] = np.mean(metrics['expert_action_accuracy'])
        metrics['task_completion_rate'] = total_completed / num_episodes

        return metrics
```

## Real-World Applications

### Surgical Robotics

```python
class SurgicalImitationLearning:
    def __init__(self):
        self.safety_constraints = {
            'force_limit': 2.0,  # Newtons
            'velocity_limit': 0.05,  # m/s
            'collision_free_zone': True
        }

    def learn_surgical_procedure(self, expert_demonstrations):
        # Preprocess surgical demonstrations
        processed_data = self.preprocess_surgical_data(expert_demonstrations)

        # Train with strict safety constraints
        safe_policy = self.train_with_safety_constraints(processed_data)

        return safe_policy

    def preprocess_surgical_data(self, demonstrations):
        # Remove unsafe movements
        # Filter based on force/torque limits
        # Ensure sterile field maintenance
        return demonstrations  # Placeholder
```

### Industrial Manipulation

```python
class IndustrialImitationLearning:
    def __init__(self):
        self.cycle_time_requirements = 10.0  # seconds per task
        self.precision_requirements = 0.001  # 1mm precision

    def learn_automation_task(self, human_demonstration):
        # Optimize for speed and precision
        optimized_policy = self.optimize_for_industrial_requirements(
            human_demonstration
        )
        return optimized_policy

    def optimize_for_industrial_requirements(self, demonstration):
        # Minimize execution time while maintaining quality
        # Optimize trajectories for speed
        # Ensure repeatability
        return demonstration  # Placeholder
```

## Implementation in ROS2

### Imitation Learning Node

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <control_msgs/msg/joint_trajectory.hpp>
#include <std_msgs/msg/bool.hpp>

class ImitationLearningNode : public rclcpp::Node {
public:
    ImitationLearningNode() : Node("imitation_learning_node") {
        // Publishers and subscribers
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&ImitationLearningNode::jointStateCallback, this, std::placeholders::_1)
        );

        command_pub_ = this->create_publisher<control_msgs::msg::JointTrajectory>(
            "joint_trajectory_controller/joint_trajectory", 10
        );

        demonstration_mode_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "imitation_learning/demonstration_mode", 10,
            std::bind(&ImitationLearningNode::demonstrationModeCallback, this, std::placeholders::_1)
        );

        // Initialize imitation learning model
        // In practice, this would interface with Python via pybind11 or similar
        demonstration_mode_ = false;
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (demonstration_mode_) {
            // Collect demonstration data
            collectDemonstrationData(*msg);
        } else {
            // Execute learned policy
            executeLearnedPolicy(*msg);
        }
    }

    void demonstrationModeCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        demonstration_mode_ = msg->data;
        if (demonstration_mode_) {
            RCLCPP_INFO(this->get_logger(), "Entering demonstration mode");
            startDemonstration();
        } else {
            RCLCPP_INFO(this->get_logger(), "Exiting demonstration mode");
            endDemonstration();
        }
    }

    void collectDemonstrationData(const sensor_msgs::msg::JointState& joint_state) {
        // Store joint positions and velocities as demonstration data
        DemonstrationData data;
        data.timestamp = this->get_clock()->now();

        for (size_t i = 0; i < joint_state.position.size(); ++i) {
            data.joint_positions.push_back(joint_state.position[i]);
        }

        for (size_t i = 0; i < joint_state.velocity.size(); ++i) {
            data.joint_velocities.push_back(joint_state.velocity[i]);
        }

        demonstration_buffer_.push_back(data);
    }

    void executeLearnedPolicy(const sensor_msgs::msg::JointState& joint_state) {
        // In a real implementation, this would call the learned policy
        // For this example, we'll simulate the policy execution

        std::vector<double> current_state = extractState(joint_state);
        std::vector<double> action = simulateLearnedPolicy(current_state);

        publishAction(action);
    }

    std::vector<double> extractState(const sensor_msgs::msg::JointState& joint_state) {
        std::vector<double> state;
        for (size_t i = 0; i < joint_state.position.size(); ++i) {
            state.push_back(joint_state.position[i]);
        }
        return state;
    }

    std::vector<double> simulateLearnedPolicy(const std::vector<double>& state) {
        // Simulated policy execution
        // In practice, this would call the trained model
        std::vector<double> action(state.size(), 0.0);
        // Add some simulated control logic
        for (size_t i = 0; i < action.size(); ++i) {
            action[i] = state[i] * 0.1;  // Simple proportional control
        }
        return action;
    }

    void publishAction(const std::vector<double>& action) {
        control_msgs::msg::JointTrajectory msg;
        trajectory_msgs::msg::JointTrajectoryPoint point;

        // Set joint names (should match your robot)
        msg.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"};

        // Set joint positions
        point.positions = action;
        point.time_from_start.sec = 0;
        point.time_from_start.nanosec = 50000000;  // 50ms

        msg.points.push_back(point);

        command_pub_->publish(msg);
    }

    void startDemonstration() {
        demonstration_buffer_.clear();
        RCLCPP_INFO(this->get_logger(), "Demonstration started");
    }

    void endDemonstration() {
        RCLCPP_INFO(this->get_logger(),
                   "Collected %zu demonstration points",
                   demonstration_buffer_.size());

        // In practice, would train the model with collected data
        trainImitationModel();
    }

    void trainImitationModel() {
        // Train the imitation learning model with collected data
        // This would typically be done in Python
        RCLCPP_INFO(this->get_logger(), "Training imitation model...");
    }

    struct DemonstrationData {
        builtin_interfaces::msg::Time timestamp;
        std::vector<double> joint_positions;
        std::vector<double> joint_velocities;
    };

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr demonstration_mode_sub_;
    rclcpp::Publisher<control_msgs::msg::JointTrajectory>::SharedPtr command_pub_;

    bool demonstration_mode_;
    std::vector<DemonstrationData> demonstration_buffer_;
};
```

## Challenges and Future Directions

### Current Challenges

#### Generalization
- Adapting to new environments
- Handling object variations
- Robustness to disturbances

#### Safety and Reliability
- Ensuring safe operation during learning
- Handling edge cases
- Validation and verification

#### Scalability
- Learning complex multi-step tasks
- Handling high-dimensional state/action spaces
- Efficient data collection

### Emerging Trends

#### Foundation Models for Imitation Learning
- Large pre-trained vision models
- Language-conditioned imitation
- Cross-task generalization

#### Human-Robot Collaboration
- Seamless transitions between human and robot control
- Intuitive communication interfaces
- Trust and transparency in autonomous systems

#### Neuromorphic Imitation Learning
- Event-based sensory processing
- Ultra-low power consumption
- Bio-inspired learning algorithms

## Troubleshooting Common Issues

### Data Quality Problems
- Noisy demonstrations
- Inconsistent expert behavior
- Insufficient demonstration diversity

### Learning Algorithm Issues
- Covariate shift in behavioral cloning
- Mode collapse in GAN-based methods
- Instability in adversarial training

### System Integration Problems
- Latency in teleoperation
- Synchronization between master and slave
- Calibration errors

## Performance Evaluation

### Standard Benchmarks

#### Manipulation Tasks
- Block stacking and arrangement
- Object retrieval and placement
- Tool use and multi-step tasks

#### Locomotion Tasks
- Walking pattern imitation
- Balance recovery behaviors
- Terrain adaptation

### Metrics

#### Task Performance
- Success rate and completion time
- Trajectory accuracy
- Energy efficiency

#### Learning Efficiency
- Samples to convergence
- Transfer performance
- Generalization capability

## Conclusion

Imitation learning and teleoperation provide powerful approaches for humanoid robots to acquire complex behaviors by learning from human demonstrations or direct control. These methods offer advantages in sample efficiency, safety, and natural task specification compared to pure reinforcement learning approaches.

The integration of imitation learning with teleoperation creates hybrid systems where humans can demonstrate tasks, robots can learn from these demonstrations, and both can collaborate effectively. As the field advances, we can expect more sophisticated algorithms that enable robots to learn complex behaviors from limited demonstrations and generalize these behaviors to new situations.

The next chapter will explore the practical aspects of building a humanoid robot, including actuators, joints, and hardware choices that enable the implementation of the algorithms discussed in this book.

## Exercises

1. Implement a behavioral cloning agent for a simple robotic manipulation task.

2. Research and implement the DAgger algorithm to address the covariate shift problem in behavioral cloning.

3. Design a teleoperation interface for a humanoid robot that includes safety constraints and haptic feedback.

4. Implement a data collection pipeline for gathering human demonstrations of a specific task.

5. Investigate how language models can be integrated with imitation learning to enable learning from natural language instructions.