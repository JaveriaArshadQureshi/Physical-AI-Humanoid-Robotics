---
title: "Chapter 11: Reinforcement Learning for Robotics"
description: "Learning control policies through environmental interaction"
---

# Chapter 11: Reinforcement Learning for Robotics

## Overview

Reinforcement Learning (RL) offers a powerful paradigm for humanoid robots to learn complex behaviors through trial-and-error interaction with their environment. This chapter explores the application of RL techniques to robotics, covering fundamental algorithms, implementation considerations, and specialized approaches for physical robots operating in real-world environments.

## Introduction to Reinforcement Learning

### The RL Framework

Reinforcement Learning is based on the interaction between an agent and its environment:

```
Environment State → Agent Action → Environment Reward → Next State → ...
```

#### Key Components
- **Agent**: The learning entity (robot)
- **Environment**: The external world the agent interacts with
- **State (s)**: Complete description of the environment
- **Action (a)**: What the agent can do
- **Reward (r)**: Feedback signal for good/bad actions
- **Policy (π)**: Strategy for selecting actions

### Markov Decision Process (MDP)

The mathematical framework for RL problems:
```
M = <S, A, P, R, γ>
```
Where:
- S: State space
- A: Action space
- P: State transition probabilities
- R: Reward function
- γ: Discount factor

### The RL Objective

Maximize expected cumulative reward:
```
J(π) = E[Σ(γ^t * r_t) | π]
```

## RL Algorithms for Robotics

### Value-Based Methods

#### Q-Learning

Q-learning learns the optimal action-value function:
```
Q(s, a) = E[r + γ * max_a' Q(s', a') | s, a]
```

```python
import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))

    def discretize_state(self, continuous_state):
        # Convert continuous state to discrete representation
        # This is a simplified example - real applications need more sophisticated discretization
        return tuple(continuous_state.astype(int))

    def choose_action(self, state):
        state_key = self.discretize_state(state)

        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(self.action_space)
        else:
            # Exploit: best known action
            return self.action_space[np.argmax(self.q_table[state_key])]

    def update(self, state, action, reward, next_state):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])

        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
```

### Deep Q-Network (DQN)

For high-dimensional state spaces, DQN uses neural networks:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.action_dim = action_dim
        self.update_target_frequency = 1000
        self.step_count = 0

    def act(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()

    def update(self, state, action, reward, next_state, done, gamma=0.99):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)

        current_q_values = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1))
        next_q_values = self.target_network(next_state_tensor).max(1)[0].detach()
        target_q_values = reward_tensor + gamma * next_q_values * (1 - done_tensor.float())

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
```

### Policy Gradient Methods

Policy gradient methods directly optimize the policy:

#### REINFORCE Algorithm
```python
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def update(self, log_probs, rewards, gamma=0.99):
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute policy loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
```

### Actor-Critic Methods

Combining policy and value learning:

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_extractor(state)

        action_probs = torch.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)

        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.network(state_tensor)

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item()

    def update(self, state, action, reward, next_state, done, gamma=0.99):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)

        action_probs, state_value = self.network(state_tensor)
        _, next_state_value = self.network(next_state_tensor)

        # Compute advantage
        target_value = reward_tensor + gamma * next_state_value * (1 - done)
        advantage = target_value - state_value

        # Compute losses
        action_dist = torch.distributions.Categorical(action_probs)
        log_prob = action_dist.log_prob(action_tensor)

        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)

        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss

        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

## Deep RL for Continuous Control

### Deep Deterministic Policy Gradient (DDPG)

For continuous action spaces:

```python
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        return self.l3(q)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Noise for exploration
        self.noise = torch.FloatTensor(action_dim).to(self.device)

    def select_action(self, state, noise_scale=0.1):
        state_tensor = torch.FloatTensor(state).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy()

        # Add noise for exploration
        noise = noise_scale * np.random.randn(action.shape[0])
        action = (action + noise).clip(-1, 1)

        return action

    def update(self, replay_buffer, batch_size=100, gamma=0.99, tau=0.005):
        # Sample batch from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        # Compute target Q-value
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action).detach()
        target_Q = reward + (not_done * gamma * target_Q).detach()

        # Critic loss
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### Soft Actor-Critic (SAC)

State-of-the-art algorithm for continuous control:

```python
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor network (Gaussian policy)
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(self.device)

        # Twin critics
        self.critic_1 = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_2 = QNetwork(state_dim, action_dim).to(self.device)

        # Target networks
        self.critic_target_1 = QNetwork(state_dim, action_dim).to(self.device)
        self.critic_target_2 = QNetwork(state_dim, action_dim).to(self.device)

        # Copy weights to target networks
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.alpha = alpha  # Temperature parameter

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            _, _, action = self.actor.sample(state_tensor)
        else:
            action, _, _ = self.actor.sample(state_tensor)

        return action.cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005):
        # Sample batch
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        not_done = torch.FloatTensor(not_done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Compute next action and log probability
            next_action, next_log_prob, _ = self.actor.sample(next_state)

            # Compute target Q-value
            next_q1 = self.critic_target_1(next_state, next_action)
            next_q2 = self.critic_target_2(next_state, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward + not_done * gamma * next_q

        # Critic loss
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()

        # Actor loss
        pi, log_pi, _ = self.actor.sample(state)
        q1_pi = self.critic_1(state, pi)
        q2_pi = self.critic_2(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

## Robotics-Specific RL Challenges

### Safety and Stability

#### Safe RL Approaches
```python
class SafeRLAgent:
    def __init__(self, base_agent, safety_constraints):
        self.base_agent = base_agent
        self.safety_constraints = safety_constraints

    def safe_act(self, state):
        # Get initial action from base agent
        action = self.base_agent.act(state)

        # Check safety constraints
        if self.is_safe_action(state, action):
            return action
        else:
            # Project action to safe set
            safe_action = self.project_to_safe_set(state, action)
            return safe_action

    def is_safe_action(self, state, action):
        # Check various safety constraints
        joint_limits_safe = self.check_joint_limits(state, action)
        collision_safe = self.check_collision(state, action)
        stability_safe = self.check_balance(state, action)

        return joint_limits_safe and collision_safe and stability_safe

    def project_to_safe_set(self, state, action):
        # Project action to satisfy safety constraints
        # This is a simplified example
        projected_action = np.clip(action, -1, 1)  # Clip to safe range
        return projected_action
```

### Sample Efficiency

#### Hindsight Experience Replay (HER)
```python
class HERBuffer:
    def __init__(self, buffer_size, k=4):
        self.buffer_size = buffer_size
        self.k = k  # Number of additional goals to sample
        self.buffer = []
        self.episode_buffer = []

    def store_transition(self, state, action, reward, next_state, done, goal):
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal
        }
        self.episode_buffer.append(transition)

        if done:
            # Add original transition
            self.buffer.append(self.episode_buffer)

            # Add HER transitions
            self.add_her_transitions()
            self.episode_buffer = []

    def add_her_transitions(self):
        episode = self.episode_buffer
        T = len(episode)

        for t in range(T):
            # Sample k future states as goals
            future_states = []
            for _ in range(self.k):
                future_t = np.random.randint(t, T)
                future_states.append(episode[future_t]['state'])

            for future_state in future_states:
                # Create new transition with future state as goal
                her_episode = []
                for i in range(len(episode)):
                    original = episode[i]
                    new_reward = self.compute_her_reward(
                        original['next_state'], future_state
                    )
                    her_transition = {
                        'state': original['state'],
                        'action': original['action'],
                        'reward': new_reward,
                        'next_state': original['next_state'],
                        'done': original['done'],
                        'goal': future_state
                    }
                    her_episode.append(her_transition)

                self.buffer.append(her_episode)

    def sample_batch(self, batch_size):
        # Sample transitions from buffer
        batch = []
        for _ in range(batch_size):
            episode = random.choice(self.buffer)
            transition = random.choice(episode)
            batch.append(transition)

        return batch
```

### Exploration Strategies

#### Intrinsic Motivation
```python
class IntrinsicMotivationAgent:
    def __init__(self, base_agent, curiosity_module):
        self.base_agent = base_agent
        self.curiosity_module = curiosity_module

    def compute_reward(self, state, action, reward, next_state):
        # Compute extrinsic reward (from environment)
        extrinsic_reward = reward

        # Compute intrinsic reward (from curiosity)
        intrinsic_reward = self.curiosity_module.compute_intrinsic_reward(
            state, action, next_state
        )

        # Combine rewards
        total_reward = extrinsic_reward + self.intrinsic_weight * intrinsic_reward

        return total_reward

class ICM(nn.Module):  # Intrinsic Curiosity Module
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ICM, self).__init__()

        # Forward model (predict next state from current state and action)
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Inverse model (predict action from state transitions)
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)  # Encoded state representation
        )

    def forward(self, state, action, next_state):
        # Encode states
        state_enc = self.state_encoder(state)
        next_state_enc = self.state_encoder(next_state)

        # Inverse model: predict action from state transition
        inverse_input = torch.cat([state_enc, next_state_enc], dim=1)
        predicted_action = self.inverse_model(inverse_input)

        # Forward model: predict next state from current state and action
        forward_input = torch.cat([state_enc, action], dim=1)
        predicted_next_state = self.forward_model(forward_input)

        # Compute prediction errors
        inverse_loss = F.mse_loss(predicted_action, action)
        forward_loss = F.mse_loss(predicted_next_state, next_state_enc)

        # Intrinsic reward is forward prediction error
        intrinsic_reward = forward_loss.detach()

        return intrinsic_reward, inverse_loss, forward_loss
```

## Sim-to-Real Transfer

### Domain Randomization
```python
class DomainRandomizationEnv:
    def __init__(self, base_env):
        self.base_env = base_env
        self.randomization_params = {
            'mass_range': [0.8, 1.2],  # Object mass multiplier
            'friction_range': [0.5, 1.5],  # Friction coefficient range
            'visual_params': {
                'lighting': ['bright', 'dim', 'overcast'],
                'textures': ['smooth', 'rough', 'patterned']
            }
        }

    def randomize_environment(self):
        # Randomize physical parameters
        mass_multiplier = np.random.uniform(
            self.randomization_params['mass_range'][0],
            self.randomization_params['mass_range'][1]
        )
        self.base_env.set_object_mass(mass_multiplier)

        friction_coeff = np.random.uniform(
            self.randomization_params['friction_range'][0],
            self.randomization_params['friction_range'][1]
        )
        self.base_env.set_friction(friction_coeff)

        # Randomize visual parameters
        lighting = np.random.choice(self.randomization_params['visual_params']['lighting'])
        self.base_env.set_lighting(lighting)

        texture = np.random.choice(self.randomization_params['visual_params']['textures'])
        self.base_env.set_texture(texture)

        return self.base_env
```

### System Identification and Adaptation
```python
class AdaptiveRLAgent:
    def __init__(self, base_agent, system_id_module):
        self.base_agent = base_agent
        self.system_id = system_id_module
        self.adaptation_frequency = 1000
        self.step_count = 0

    def update_system_model(self, state, action, next_state):
        # Update system identification model
        self.system_id.update(state, action, next_state)

        # Adapt policy based on updated model
        if self.step_count % self.adaptation_frequency == 0:
            self.adapt_policy()

    def adapt_policy(self):
        # Adapt policy based on identified system parameters
        system_params = self.system_id.get_parameters()
        self.base_agent.update_system_parameters(system_params)
```

## Multi-Task and Transfer Learning

### Meta-Learning for Robotics
```python
class MetaRLAgent:
    def __init__(self, meta_learner, task_encoder):
        self.meta_learner = meta_learner
        self.task_encoder = task_encoder
        self.context_window = 10  # Number of recent transitions to condition on

    def adapt_to_new_task(self, task_description, initial_experience):
        # Encode task description
        task_embedding = self.task_encoder(task_description)

        # Initialize policy with task embedding
        self.meta_learner.initialize_policy(task_embedding)

        # Fast adaptation using initial experience
        for state, action, reward, next_state in initial_experience:
            self.meta_learner.update_from_experience(
                state, action, reward, next_state
            )

    def act_with_context(self, state, recent_transitions):
        # Use recent experience as context
        context = self.encode_context(recent_transitions)

        # Act using both task embedding and context
        return self.meta_learner.act_with_context(state, context)
```

## Implementation in ROS2

### RL Training Node
```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <control_msgs/msg/joint_trajectory.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

class RLTrainingNode : public rclcpp::Node {
public:
    RLTrainingNode() : Node("rl_training_node") {
        // Publishers and subscribers
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&RLTrainingNode::jointStateCallback, this, std::placeholders::_1)
        );

        action_pub_ = this->create_publisher<control_msgs::msg::JointTrajectory>(
            "joint_trajectory_controller/joint_trajectory", 10
        );

        reward_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            "rl/reward", 10
        );

        // Initialize RL agent (would connect to Python via pybind11 or similar)
        // For this example, we'll simulate the RL agent
        initializeRLAgent();
    }

private:
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        // Extract state from joint states
        std::vector<double> state = extractStateFromJointState(*msg);

        // Get action from RL agent
        std::vector<double> action = getRLAction(state);

        // Execute action
        publishAction(action);

        // Calculate and publish reward
        double reward = calculateReward(state, action);
        publishReward(reward);
    }

    std::vector<double> extractStateFromJointState(const sensor_msgs::msg::JointState& joint_state) {
        std::vector<double> state;

        // Extract joint positions and velocities
        for (size_t i = 0; i < joint_state.position.size(); ++i) {
            state.push_back(joint_state.position[i]);
        }

        for (size_t i = 0; i < joint_state.velocity.size(); ++i) {
            state.push_back(joint_state.velocity[i]);
        }

        return state;
    }

    std::vector<double> getRLAction(const std::vector<double>& state) {
        // In a real implementation, this would call the RL agent
        // For simulation, return random action
        std::vector<double> action(7, 0.0);  // 7-DOF arm
        for (auto& a : action) {
            a = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;  // -1 to 1
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
        point.time_from_start.nanosec = 100000000;  // 100ms

        msg.points.push_back(point);

        action_pub_->publish(msg);
    }

    double calculateReward(const std::vector<double>& state, const std::vector<double>& action) {
        // Calculate reward based on task
        // This is a simplified example for reaching task
        double target_x = 0.5, target_y = 0.0, target_z = 0.8;

        // Get end-effector position from state (simplified)
        double ee_x = state[0];  // This would require forward kinematics
        double ee_y = state[1];
        double ee_z = state[2];

        // Distance to target
        double dist = sqrt(pow(ee_x - target_x, 2) + pow(ee_y - target_y, 2) + pow(ee_z - target_z, 2));

        // Reward is negative distance (closer is better)
        double reward = -dist;

        // Add small penalty for large actions
        double action_penalty = 0.0;
        for (const auto& a : action) {
            action_penalty += 0.01 * a * a;
        }

        return reward - action_penalty;
    }

    void publishReward(double reward) {
        std_msgs::msg::Float64 msg;
        msg.data = reward;
        reward_pub_->publish(msg);
    }

    void initializeRLAgent() {
        // Initialize RL agent
        // In practice, this would connect to a Python RL framework
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<control_msgs::msg::JointTrajectory>::SharedPtr action_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr reward_pub_;
};
```

## Humanoid-Specific RL Applications

### Bipedal Locomotion
```python
class BipedalWalkerRL:
    def __init__(self):
        self.balance_reward_weight = 1.0
        self.velocity_reward_weight = 0.5
        self.energy_penalty_weight = 0.1

    def compute_locomotion_reward(self, robot_state, action, target_velocity=1.0):
        reward = 0.0

        # Balance reward (keep torso upright)
        torso_angle = robot_state['torso_angle']
        balance_reward = np.exp(-abs(torso_angle))  # Higher reward for upright position
        reward += self.balance_reward_weight * balance_reward

        # Forward velocity reward
        forward_velocity = robot_state['forward_velocity']
        velocity_reward = abs(forward_velocity - target_velocity)
        reward += self.velocity_reward_weight * (1.0 - min(velocity_reward, 1.0))

        # Energy efficiency penalty
        joint_torques = action  # Assuming action represents torques
        energy_penalty = np.sum(np.abs(joint_torques) * np.abs(robot_state['joint_velocities']))
        reward -= self.energy_penalty_weight * energy_penalty

        # Penalty for falling
        if robot_state['torso_angle'] > np.pi/3 or robot_state['torso_angle'] < -np.pi/3:
            reward -= 100  # Large penalty for falling

        return reward
```

### Whole-Body Control
```python
class WholeBodyRL:
    def __init__(self):
        self.task_completion_weight = 2.0
        self.balance_weight = 1.5
        self.smoothness_weight = 0.5

    def compute_whole_body_reward(self, state, action, task_goal):
        reward = 0.0

        # Task completion reward
        task_progress = self.compute_task_progress(state, task_goal)
        reward += self.task_completion_weight * task_progress

        # Balance maintenance
        zmp_deviation = self.compute_zmp_deviation(state)
        balance_reward = np.exp(-abs(zmp_deviation))  # Higher reward for stable ZMP
        reward += self.balance_weight * balance_reward

        # Smoothness penalty
        action_smoothness = self.compute_action_smoothness(action)
        reward -= self.smoothness_weight * action_smoothness

        return reward

    def compute_task_progress(self, state, goal):
        # Compute how close we are to task completion
        # This would depend on specific task (manipulation, navigation, etc.)
        return 0.0  # Placeholder

    def compute_zmp_deviation(self, state):
        # Compute Zero Moment Point deviation from stable region
        # This is a simplified calculation
        return 0.0  # Placeholder

    def compute_action_smoothness(self, action):
        # Penalize jerky or discontinuous actions
        return np.sum(np.abs(action))  # Simplified smoothness penalty
```

## Advanced RL Techniques

### Multi-Agent RL
```python
class MultiRobotRL:
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.robots = [RLAgent() for _ in range(num_robots)]

    def centralized_training(self, states, actions, rewards, next_states):
        # Train all robots together with centralized critic
        # Share information between robots during training
        for i in range(self.num_robots):
            # Use information from all robots to update each agent
            self.robots[i].update_centralized(
                states, actions, rewards[i], next_states
            )

    def decentralized_execution(self, local_states):
        # Each robot acts based on local observations
        actions = []
        for i, local_state in enumerate(local_states):
            action = self.robots[i].act_decentralized(local_state)
            actions.append(action)
        return actions
```

### Hierarchical RL
```python
class HierarchicalRL:
    def __init__(self):
        self.high_level_policy = HighLevelPolicy()
        self.low_level_policy = LowLevelPolicy()

    def act(self, state, high_level_frequency=10):
        # Execute high-level action for several steps
        if self.step_count % high_level_frequency == 0:
            self.high_level_action = self.high_level_policy.act(state)

        # Execute low-level action using high-level action as context
        low_level_action = self.low_level_policy.act(
            state, self.high_level_action
        )

        self.step_count += 1
        return low_level_action
```

## Evaluation and Benchmarking

### Standard Robotics RL Benchmarks

#### MuJoCo Environments
- Ant, HalfCheetah, Hopper, Walker2d for locomotion
- Reacher, Pusher, Striker for manipulation
- Humanoid for complex whole-body control

#### PyBullet Environments
- More realistic physics simulation
- Better for sim-to-real transfer
- Support for complex humanoid models

### Performance Metrics

#### Learning Efficiency
- Sample efficiency (samples needed to reach target performance)
- Convergence speed
- Asymptotic performance

#### Robustness
- Performance across different initial conditions
- Generalization to new environments
- Transfer to real robots

## Troubleshooting Common Issues

### Training Instability
- Use experience replay to break correlation
- Target networks for stable Q-learning
- Proper hyperparameter tuning

### Exploration vs. Exploitation
- Anneal exploration parameters over time
- Use intrinsic motivation for better exploration
- Curriculum learning for complex tasks

### Sim-to-Real Gap
- Domain randomization during training
- System identification and adaptation
- Robust control design

## Future Directions

### Foundation Models for RL
- Pre-trained vision and language models for better perception
- Transfer learning across tasks and robots
- Few-shot learning from demonstrations

### Neuromorphic RL
- Event-based sensors and learning
- Ultra-low power consumption
- Bio-inspired learning algorithms

### Human-Robot Interaction
- Learning from human feedback
- Socially-aware RL
- Collaborative task learning

## Conclusion

Reinforcement Learning provides powerful tools for humanoid robots to learn complex behaviors through environmental interaction. The combination of deep learning with RL algorithms enables robots to learn policies for high-dimensional state and action spaces that were previously intractable.

While challenges remain in terms of sample efficiency, safety, and sim-to-real transfer, the field continues to advance with new algorithms and techniques specifically designed for robotics applications. The integration of RL with other AI techniques, such as vision-language models, promises even more capable and adaptive robotic systems.

The next chapter will explore imitation learning and teleoperation, which provide complementary approaches to RL by learning from expert demonstrations.

## Exercises

1. Implement a simple DQN agent to solve a basic robotic control task (e.g., reaching).

2. Research and implement the Soft Actor-Critic (SAC) algorithm for continuous control.

3. Design a reward function for a bipedal walking task that balances forward progress, stability, and energy efficiency.

4. Implement Hindsight Experience Replay (HER) for a robotic manipulation task.

5. Investigate how domain randomization can be applied to make RL policies more robust to real-world variations.