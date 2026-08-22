import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Type, Union, Generator

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces

class DualHeadActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        
        # Override the value net to output 2 values (short-term and long-term)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 2)
        
        # Re-initialize the optimizer to include the new value_net parameters
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class MultiDiscountRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,  # This will act as gamma_short
        n_envs: int = 1,
        gamma_long: float = 0.999,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs
        )
        self.gamma_short = gamma
        self.gamma_long = gamma_long

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        
        # Two reward streams
        self.rewards_short = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rewards_long = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Two return streams stored in a single (buffer_size, n_envs, 2) tensor
        self.returns = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        
        # Two value streams (from the dual head)
        self.values = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        # Total advantage (scalar per env)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Per-component advantages for logging
        self.advantages_short = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages_long = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        
        # Decompose the reward
        r = np.array(reward).copy()
        # In our environment, terminal goal/fail rewards are exactly 15.0 or -10.0
        is_long = np.isclose(r, 15.0) | np.isclose(r, -10.0)
        self.rewards_long[self.pos] = np.where(is_long, r, 0.0)
        self.rewards_short[self.pos] = np.where(is_long, 0.0, r)
        
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        
        # Value is now [n_envs, 2]
        self.values[self.pos] = value.clone().cpu().numpy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().flatten()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        # last_values is [n_envs, 2]
        last_values_cpu = last_values.clone().cpu().numpy()
        last_values_short = last_values_cpu[:, 0]
        last_values_long = last_values_cpu[:, 1]
        
        last_gae_lam_short = 0
        last_gae_lam_long = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values_short = last_values_short
                next_values_long = last_values_long
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values_short = self.values[step + 1, :, 0]
                next_values_long = self.values[step + 1, :, 1]
            
            # Short-term GAE
            delta_short = self.rewards_short[step] + self.gamma_short * next_values_short * next_non_terminal - self.values[step, :, 0]
            last_gae_lam_short = delta_short + self.gamma_short * self.gae_lambda * next_non_terminal * last_gae_lam_short
            
            # Long-term GAE
            delta_long = self.rewards_long[step] + self.gamma_long * next_values_long * next_non_terminal - self.values[step, :, 1]
            last_gae_lam_long = delta_long + self.gamma_long * self.gae_lambda * next_non_terminal * last_gae_lam_long
            
            # Store returns and total advantages
            self.returns[step, :, 0] = last_gae_lam_short + self.values[step, :, 0]
            self.returns[step, :, 1] = last_gae_lam_long + self.values[step, :, 1]

            self.advantages_short[step] = last_gae_lam_short
            self.advantages_long[step] = last_gae_lam_long
            self.advantages[step] = last_gae_lam_short + last_gae_lam_long

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[Any] = None) -> RolloutBufferSamples:
        # returns and values are stacked so their last dimension is 2
        # when we flatten, (batch_size, 2) becomes (batch_size * 2)
        # This will perfectly align for F.mse_loss in PPO.train!
        
        # Combined returns and values are shaped (buffer_size * n_envs, 2) due to SB3 get()
        # We flatten them to (batch_size * 2) so PPO MSE Loss works automatically
        
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),   # (batch_size * 2,)
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds].flatten(),                  # (batch_size * 2,)
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class DualHeadPPO(PPO):
    def __init__(self, *args, gamma_long=0.999, **kwargs):
        self.gamma_long = gamma_long
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        
        # Override the buffer with our custom one
        self.rollout_buffer = MultiDiscountRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            gamma_long=self.gamma_long,
        )
