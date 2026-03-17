"""
PPO Agent for Trading.
Proximal Policy Optimization implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger


@dataclass
class RolloutBuffer:
    """
    Buffer for storing rollout data.

    Attributes:
        states: List of states
        actions: List of actions
        rewards: List of rewards
        values: List of value estimates
        log_probs: List of log probabilities
        dones: List of done flags
    """
    states: List[np.ndarray]
    actions: List[float]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self) -> None:
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization agent for trading.

    Features:
        - Continuous action space
        - Generalized Advantage Estimation (GAE)
        - Value function clipping
        - Entropy bonus
        - Learning rate annealing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        use_gae: bool = True,
        normalize_advantage: bool = True,
        random_state: int = 42
    ):
        """
        Initialize PPO Agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm
            n_epochs: Number of training epochs per update
            batch_size: Mini-batch size
            use_gae: Use Generalized Advantage Estimation
            normalize_advantage: Normalize advantages
            random_state: Random seed
        """
        self.logger = get_logger('ppo_agent')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.normalize_advantage = normalize_advantage
        self.random_state = random_state

        np.random.seed(random_state)

        # Networks
        self.policy = None
        self.value_network = None
        self.optimizer = None

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Build networks
        self._build_networks()

    def _build_networks(self) -> None:
        """Build policy and value networks."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)

        class ActorCriticNetwork(nn.Module):
            """Combined Actor-Critic Network."""

            def __init__(self, state_dim, action_dim, hidden_dims):
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

                # Policy head (actor)
                self.policy_mean = nn.Linear(prev_dim, action_dim)
                self.policy_log_std = nn.Parameter(torch.zeros(action_dim))

                # Value head (critic)
                self.value = nn.Linear(prev_dim, 1)

            def forward(self, x):
                features = self.shared(x)

                # Policy output
                mean = self.policy_mean(features)
                std = torch.exp(self.policy_log_std.clamp(-20, 2))

                # Value output
                value = self.value(features)

                return mean, std, value

        # Create network
        self.network = ActorCriticNetwork(
            self.state_dim, self.action_dim, self.hidden_dims
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False
    ) -> Tuple[float, float, float]:
        """
        Select action using current policy.

        Args:
            state: Current state
            evaluate: Whether to use deterministic policy

        Returns:
            Tuple of (action, value, log_prob)
        """
        import torch
        from torch.distributions import Normal

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, std, value = self.network(state_tensor)

            if evaluate:
                action = mean.squeeze().cpu().numpy()
                log_prob = 0.0
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1).item()
                action = action.squeeze().cpu().numpy()

            value = value.squeeze().cpu().numpy()

        # Clip action to [-1, 1]
        action = np.clip(action, -1, 1)

        return float(action), float(value), float(log_prob)

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            next_value: Value estimate for next state

        Returns:
            Tuple of (advantages, returns)
        """
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        last_gae = 0
        last_return = next_value

        for i in reversed(range(n)):
            if dones[i]:
                last_gae = 0
                last_return = values[i]

            delta = rewards[i] + self.gamma * last_return * (1 - dones[i]) - values[i]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * last_gae
            advantages[i] = last_gae
            last_return = values[i] + advantages[i]
            returns[i] = last_return

        return advantages, returns

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Update policy and value networks.

        Args:
            next_value: Value of the final state

        Returns:
            Dictionary of training metrics
        """
        import torch
        import torch.nn.functional as F
        from torch.distributions import Normal

        if len(self.buffer) < self.batch_size:
            return {}

        # Get data from buffer
        states = np.array(self.buffer.states)
        actions = np.array(self.buffer.actions)
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        log_probs = np.array(self.buffer.log_probs)
        dones = np.array(self.buffer.dones)

        # Compute advantages and returns
        if self.use_gae:
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        else:
            returns = rewards + self.gamma * np.append(values[1:], next_value) * (1 - dones)
            advantages = returns - values

        # Normalize advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # Training epochs
        n_samples = len(states)
        indices = np.arange(n_samples)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Forward pass
                mean, std, values_pred = self.network(states[batch_indices])

                # Create distribution
                dist = Normal(mean, std)

                # New log probabilities
                new_log_probs = dist.log_prob(actions[batch_indices].unsqueeze(-1)).sum(dim=-1)

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values_pred.squeeze(), returns[batch_indices])

                # Entropy bonus
                entropy = dist.entropy().sum(dim=-1).mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # Clear buffer
        self.buffer.clear()

        n_updates = self.n_epochs * (n_samples // self.batch_size)

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }

    def save(self, filepath: str) -> None:
        """
        Save agent to file.

        Args:
            filepath: Save path
        """
        import torch

        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)

        self.logger.info(f"Saved PPO agent to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load agent from file.

        Args:
            filepath: Load path
        """
        import torch

        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Loaded PPO agent from {filepath}")


class A2CAgent(PPOAgent):
    """
    Advantage Actor-Critic agent.

    Simpler version of PPO without clipping and multiple epochs.
    """

    def __init__(
        self,
        *args,
        n_epochs: int = 1,  # A2C only needs one epoch
        **kwargs
    ):
        """Initialize A2C Agent."""
        super().__init__(*args, n_epochs=n_epochs, **kwargs)
        self.logger = get_logger('a2c_agent')

    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Update policy and value networks (A2C version).

        Args:
            next_value: Value of the final state

        Returns:
            Dictionary of training metrics
        """
        import torch
        import torch.nn.functional as F
        from torch.distributions import Normal

        if len(self.buffer) < 1:
            return {}

        # Get data from buffer
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones)

        # Compute returns and advantages
        if self.use_gae:
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        else:
            returns = rewards + self.gamma * np.append(values[1:], next_value) * (1 - dones)
            advantages = returns - values

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Forward pass
        mean, std, values_pred = self.network(states)
        dist = Normal(mean, std)

        # Log probabilities
        log_probs = dist.log_prob(actions.unsqueeze(-1)).sum(dim=-1)

        # Policy loss (no clipping for A2C)
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(values_pred.squeeze(), returns)

        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()

        # Total loss
        loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
