"""
Training Utilities for RL Agents.
Training loops, callbacks, and evaluation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from rl_agents.trading_env import TradingEnvironment
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ppo_agent import PPOAgent, A2CAgent


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        save_freq: Model save frequency
        log_freq: Logging frequency
        learning_rate: Learning rate (or schedule)
        batch_size: Batch size
        n_envs: Number of parallel environments
    """
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    n_eval_episodes: int = 5
    save_freq: int = 50000
    log_freq: int = 1000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_envs: int = 1


@dataclass
class TrainingMetrics:
    """
    Training metrics.

    Attributes:
        timestep: Current timestep
        episode: Current episode
        episode_reward: Episode reward
        episode_length: Episode length
        loss: Training loss
        learning_rate: Current learning rate
        epsilon: Current epsilon (for DQN)
    """
    timestep: int = 0
    episode: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    epsilon: float = 0.0


class Trainer:
    """
    Training manager for RL agents.

    Provides:
        - Training loops
        - Evaluation
        - Checkpointing
        - Logging
        - Early stopping
    """

    def __init__(
        self,
        agent: Any,
        env: TradingEnvironment,
        config: Optional[TrainingConfig] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize Trainer.

        Args:
            agent: RL agent to train
            env: Trading environment
            config: Training configuration
            log_dir: Directory for logs and checkpoints
        """
        self.logger = get_logger('trainer')
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        self.log_dir = log_dir or './logs'

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Training state
        self.timestep = 0
        self.episode = 0
        self.best_reward = -np.inf
        self.training_history: List[Dict] = []

    def train(self) -> Dict[str, Any]:
        """
        Train the agent.

        Returns:
            Training results dictionary
        """
        self.logger.info(f"Starting training for {self.config.total_timesteps} timesteps")

        start_time = datetime.now()
        episode_rewards = []
        episode_lengths = []

        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        losses = []

        while self.timestep < self.config.total_timesteps:
            # Select action
            if isinstance(self.agent, DQNAgent):
                action = self.agent.select_action(state)
                # Convert discrete action to continuous
                position = self.agent.action_to_position(action)
            else:
                action, value, log_prob = self.agent.select_action(state)
                position = action

            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(position)

            # Update agent
            if isinstance(self.agent, DQNAgent):
                loss = self.agent.update(state, action, reward, next_state, terminated or truncated)
            else:
                self.agent.buffer.add(state, action, reward, value, log_prob, terminated or truncated)
                loss = None

                # Update PPO at end of episode
                if terminated or truncated:
                    next_value = 0.0
                    loss_dict = self.agent.update(next_value)
                    loss = loss_dict.get('policy_loss', 0) if loss_dict else None

            if loss is not None:
                losses.append(loss)

            # Update counters
            self.timestep += 1
            episode_reward += reward
            episode_length += 1
            state = next_state

            # End of episode
            if terminated or truncated:
                self.episode += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Log episode
                if self.episode % self.config.log_freq == 0:
                    avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                    avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                    avg_loss = np.mean(losses[-100:]) if losses else 0

                    self.logger.info(
                        f"Timestep {self.timestep}/{self.config.total_timesteps} | "
                        f"Episode {self.episode} | "
                        f"Avg Reward: {avg_reward:.4f} | "
                        f"Avg Length: {avg_length:.1f} | "
                        f"Avg Loss: {avg_loss:.6f}"
                    )

                # Save training history
                self.training_history.append({
                    'timestep': self.timestep,
                    'episode': self.episode,
                    'reward': episode_reward,
                    'length': episode_length,
                    'loss': losses[-1] if losses else 0
                })

                # Evaluate
                if self.episode % self.config.eval_freq == 0:
                    eval_result = self.evaluate()
                    if eval_result['mean_reward'] > self.best_reward:
                        self.best_reward = eval_result['mean_reward']
                        self._save_checkpoint('best_model')

                # Save checkpoint
                if self.timestep % self.config.save_freq == 0:
                    self._save_checkpoint(f'model_{self.timestep}')

                # Reset environment
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0

        # Final save
        self._save_checkpoint('final_model')

        training_time = (datetime.now() - start_time).total_seconds()

        results = {
            'total_timesteps': self.timestep,
            'total_episodes': self.episode,
            'training_time': training_time,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'best_reward': self.best_reward
        }

        self.logger.info(f"Training completed: {results}")

        return results

    def evaluate(
        self,
        n_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the agent.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Evaluation results
        """
        n_episodes = n_episodes or self.config.n_eval_episodes

        episode_rewards = []
        episode_lengths = []
        portfolio_stats = []

        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # Select action (greedy for DQN, deterministic for PPO)
                if isinstance(self.agent, DQNAgent):
                    action = self.agent.select_action(state, evaluate=True)
                    position = self.agent.action_to_position(action)
                else:
                    action, _, _ = self.agent.select_action(state, evaluate=True)
                    position = action

                state, reward, terminated, truncated, _ = self.env.step(position)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            portfolio_stats.append(self.env.get_portfolio_stats())

        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'portfolio_stats': {
                key: np.mean([s.get(key, 0) for s in portfolio_stats])
                for key in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            }
        }

        self.logger.info(
            f"Evaluation: Mean Reward = {results['mean_reward']:.4f} "
            f"+/- {results['std_reward']:.4f}"
        )

        return results

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        filepath = os.path.join(self.log_dir, f'{name}.pt')
        self.agent.save(filepath)

        # Save training history
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        self.agent.load(filepath)
        self.logger.info(f"Loaded checkpoint from {filepath}")


class Callback:
    """Base callback class."""

    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer: Trainer) -> None:
        """Set trainer reference."""
        self.trainer = trainer

    def on_step(self, metrics: TrainingMetrics) -> bool:
        """
        Called at each training step.

        Args:
            metrics: Current training metrics

        Returns:
            True to continue training, False to stop
        """
        return True

    def on_episode_end(self, metrics: TrainingMetrics) -> bool:
        """
        Called at end of each episode.

        Args:
            metrics: Current training metrics

        Returns:
            True to continue training, False to stop
        """
        return True

    def on_training_end(self, metrics: TrainingMetrics) -> None:
        """Called at end of training."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        metric: str = 'reward'
    ):
        """
        Initialize Early Stopping Callback.

        Args:
            patience: Number of evaluations to wait
            min_delta: Minimum improvement
            metric: Metric to monitor
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.wait = 0
        self.best_value = -np.inf

    def on_step(self, metrics: TrainingMetrics) -> bool:
        """Check early stopping condition."""
        value = getattr(metrics, self.metric, 0)

        if value - self.best_value > self.min_delta:
            self.best_value = value
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.trainer.logger.info(f"Early stopping triggered after {self.wait} evaluations")
            return False

        return True


class ProgressBarCallback(Callback):
    """Progress bar callback."""

    def __init__(self):
        """Initialize Progress Bar Callback."""
        super().__init__()
        self.pbar = None

    def on_step(self, metrics: TrainingMetrics) -> bool:
        """Update progress bar."""
        if self.pbar is None:
            from tqdm import tqdm
            self.pbar = tqdm(
                total=self.trainer.config.total_timesteps,
                desc="Training"
            )

        self.pbar.update(1)
        self.pbar.set_postfix({
            'reward': f"{metrics.episode_reward:.4f}",
            'loss': f"{metrics.loss:.6f}"
        })

        return True

    def on_training_end(self, metrics: TrainingMetrics) -> None:
        """Close progress bar."""
        if self.pbar is not None:
            self.pbar.close()


def train_dqn(
    data: pd.DataFrame,
    total_timesteps: int = 500000,
    learning_rate: float = 1e-4,
    **kwargs
) -> Tuple[DQNAgent, Dict]:
    """
    Convenience function to train a DQN agent.

    Args:
        data: Training data
        total_timesteps: Total timesteps
        learning_rate: Learning rate
        **kwargs: Additional arguments

    Returns:
        Tuple of (trained agent, training results)
    """
    # Create environment
    env = TradingEnvironment(data, **kwargs)

    # Create agent
    state_dim = env.observation_dim
    agent = DQNAgent(
        state_dim=state_dim,
        learning_rate=learning_rate
    )

    # Train
    config = TrainingConfig(total_timesteps=total_timesteps)
    trainer = Trainer(agent, env, config)
    results = trainer.train()

    return agent, results


def train_ppo(
    data: pd.DataFrame,
    total_timesteps: int = 500000,
    learning_rate: float = 3e-4,
    **kwargs
) -> Tuple[PPOAgent, Dict]:
    """
    Convenience function to train a PPO agent.

    Args:
        data: Training data
        total_timesteps: Total timesteps
        learning_rate: Learning rate
        **kwargs: Additional arguments

    Returns:
        Tuple of (trained agent, training results)
    """
    # Create environment
    env = TradingEnvironment(data, **kwargs)

    # Create agent
    state_dim = env.observation_dim
    agent = PPOAgent(
        state_dim=state_dim,
        learning_rate=learning_rate
    )

    # Train
    config = TrainingConfig(total_timesteps=total_timesteps)
    trainer = Trainer(agent, env, config)
    results = trainer.train()

    return agent, results
