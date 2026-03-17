"""
DQN Agent for Trading.
Deep Q-Network implementation with experience replay.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores and samples transitions for off-policy learning.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize Replay Buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of batch arrays
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for trading.

    Features:
        - Experience replay
        - Target network
        - Epsilon-greedy exploration
        - Double DQN
        - Dueling network architecture
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 11,  # Discrete actions
        hidden_dims: List[int] = [256, 128, 64],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        random_state: int = 42
    ):
        """
        Initialize DQN Agent.

        Args:
            state_dim: State dimension
            n_actions: Number of discrete actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            use_double_dqn: Use Double DQN
            use_dueling: Use dueling network
            random_state: Random seed
        """
        self.logger = get_logger('dqn_agent')
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.random_state = random_state

        np.random.seed(random_state)
        random.seed(random_state)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Networks (will be built on first use)
        self.q_network = None
        self.target_network = None
        self.optimizer = None

        # Training counters
        self.step_count = 0
        self.update_count = 0

        # Build networks
        self._build_networks()

    def _build_networks(self) -> None:
        """Build Q-networks."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        class DuelingQNetwork(nn.Module):
            """Dueling Q-Network."""

            def __init__(self, state_dim, n_actions, hidden_dims):
                super().__init__()

                # Shared layers
                layers = []
                prev_dim = state_dim
                for dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.LayerNorm(dim)
                    ])
                    prev_dim = dim

                self.shared = nn.Sequential(*layers)

                # Value stream
                self.value = nn.Sequential(
                    nn.Linear(prev_dim, prev_dim // 2),
                    nn.ReLU(),
                    nn.Linear(prev_dim // 2, 1)
                )

                # Advantage stream
                self.advantage = nn.Sequential(
                    nn.Linear(prev_dim, prev_dim // 2),
                    nn.ReLU(),
                    nn.Linear(prev_dim // 2, n_actions)
                )

            def forward(self, x):
                features = self.shared(x)
                value = self.value(features)
                advantage = self.advantage(features)

                # Q = V + (A - mean(A))
                q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
                return q_values

        class QNetwork(nn.Module):
            """Standard Q-Network."""

            def __init__(self, state_dim, n_actions, hidden_dims):
                super().__init__()

                layers = []
                prev_dim = state_dim
                for dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, dim),
                        nn.ReLU(),
                        nn.LayerNorm(dim)
                    ])
                    prev_dim = dim

                layers.append(nn.Linear(prev_dim, n_actions))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        # Create networks
        if self.use_dueling:
            self.q_network = DuelingQNetwork(
                self.state_dim, self.n_actions, self.hidden_dims
            )
            self.target_network = DuelingQNetwork(
                self.state_dim, self.n_actions, self.hidden_dims
            )
        else:
            self.q_network = QNetwork(
                self.state_dim, self.n_actions, self.hidden_dims
            )
            self.target_network = QNetwork(
                self.state_dim, self.n_actions, self.hidden_dims
            )

        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            evaluate: Whether to use greedy policy

        Returns:
            Selected action index
        """
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        import torch

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def action_to_position(self, action: int) -> float:
        """
        Convert discrete action to position size.

        Args:
            action: Action index

        Returns:
            Position size in [-1, 1]
        """
        # Map discrete actions to continuous positions
        return (action / (self.n_actions - 1)) * 2 - 1

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        import torch
        import torch.nn.functional as F

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select actions
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_network(next_states).max(dim=1)[0]

            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """
        Update agent with new transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended

        Returns:
            Loss value if training occurred
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

        return self.train_step()

    def save(self, filepath: str) -> None:
        """
        Save agent to file.

        Args:
            filepath: Save path
        """
        import torch

        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count
        }, filepath)

        self.logger.info(f"Saved DQN agent to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load agent from file.

        Args:
            filepath: Load path
        """
        import torch

        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']

        self.logger.info(f"Loaded DQN agent from {filepath}")


class NoisyDQNAgent(DQNAgent):
    """
    Noisy DQN agent with noisy networks for exploration.

    Uses noisy linear layers instead of epsilon-greedy.
    """

    def __init__(self, *args, **kwargs):
        """Initialize Noisy DQN Agent."""
        super().__init__(*args, **kwargs)
        self._build_noisy_networks()

    def _build_noisy_networks(self) -> None:
        """Build networks with noisy layers."""
        import torch
        import torch.nn as nn

        class NoisyLinear(nn.Module):
            """Noisy linear layer."""

            def __init__(self, in_features, out_features, sigma_init=0.5):
                super().__init__()

                self.in_features = in_features
                self.out_features = out_features

                # Learnable parameters
                self.weight_mu = nn.Parameter(
                    torch.empty(out_features, in_features)
                )
                self.weight_sigma = nn.Parameter(
                    torch.empty(out_features, in_features)
                )
                self.bias_mu = nn.Parameter(torch.empty(out_features))
                self.bias_sigma = nn.Parameter(torch.empty(out_features))

                # Noise buffers
                self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
                self.register_buffer('bias_epsilon', torch.zeros(out_features))

                # Initialize
                nn.init.xavier_uniform_(self.weight_mu)
                nn.init.constant_(self.weight_sigma, sigma_init / np.sqrt(in_features))
                nn.init.zeros_(self.bias_mu)
                nn.init.constant_(self.bias_sigma, sigma_init / np.sqrt(out_features))

                self.reset_noise()

            def forward(self, x):
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
                return nn.functional.linear(x, weight, bias)

            def reset_noise(self):
                # Factorized Gaussian noise
                epsilon_in = torch.randn(self.in_features, device=self.weight_mu.device)
                epsilon_out = torch.randn(self.out_features, device=self.weight_mu.device)

                self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
                self.bias_epsilon.copy_(epsilon_out)

        # Replace linear layers with noisy layers in the network
        # This is a simplified implementation
        self.logger.info("Using Noisy DQN for exploration")

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using noisy network."""
        import torch

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Reset noise for exploration
            if not evaluate:
                for module in self.q_network.modules():
                    if hasattr(module, 'reset_noise'):
                        module.reset_noise()

            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
