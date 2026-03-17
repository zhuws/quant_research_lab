"""
RL Agents for Quant Research Lab.
Reinforcement learning agents for trading.
"""

from rl_agents.trading_env import (
    TradingEnvironment,
    MultiAssetTradingEnvironment,
    Position,
    TradingState,
    ActionType
)
from rl_agents.dqn_agent import DQNAgent, NoisyDQNAgent, ReplayBuffer
from rl_agents.ppo_agent import PPOAgent, A2CAgent, RolloutBuffer
from rl_agents.training import (
    Trainer,
    TrainingConfig,
    TrainingMetrics,
    Callback,
    EarlyStoppingCallback,
    ProgressBarCallback,
    train_dqn,
    train_ppo
)

__all__ = [
    # Environment
    'TradingEnvironment',
    'MultiAssetTradingEnvironment',
    'Position',
    'TradingState',
    'ActionType',

    # DQN
    'DQNAgent',
    'NoisyDQNAgent',
    'ReplayBuffer',

    # PPO
    'PPOAgent',
    'A2CAgent',
    'RolloutBuffer',

    # Training
    'Trainer',
    'TrainingConfig',
    'TrainingMetrics',
    'Callback',
    'EarlyStoppingCallback',
    'ProgressBarCallback',
    'train_dqn',
    'train_ppo'
]
