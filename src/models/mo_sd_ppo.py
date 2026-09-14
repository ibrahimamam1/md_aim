"""
mo_sd_ppo.py
────────────
Multi-Objective, State-Dependent PPO (MOSDPPO) and MultiObjectiveRolloutBuffer.

Supports:
  - Baseline: Fixed single discount γ_0.
  - Experiment A: State-dependent single discount γ(s).
  - Experiment B: Multi-objective state-dependent discount [γ_l, γ_s(s)] (Core).
  - Experiment C: Learnable discount factors γ_φ(s) with anti-cheating regularized loss.
  - Ablation: State-dependent reward weighting λ(s) with fixed discount.
"""

import os
import sys
import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mo_sd_models import MultiObjectiveActorCriticPolicy, LearnableDiscountNet


class MultiObjectiveRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer supporting decoupled vector rewards, dual value heads,
    per-step state-dependent discount factors [γ_l(t), γ_s(t)], and scalarized GAE.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
        critic_dim: int = 2,
        weight_l: float = 0.5,
        weight_s: float = 0.5,
    ):
        self.critic_dim = critic_dim
        self.weight_l = weight_l
        self.weight_s = weight_s

        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

        # Decomposed vector rewards [buffer_size, n_envs, critic_dim]
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.critic_dim), dtype=np.float32)

        # Dual value heads and returns
        self.values = np.zeros((self.buffer_size, self.n_envs, self.critic_dim), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs, self.critic_dim), dtype=np.float32)

        # Per-step, per-head discount factors [buffer_size, n_envs, critic_dim]
        self.gammas = np.zeros((self.buffer_size, self.n_envs, self.critic_dim), dtype=np.float32)

        # Target risk for auxiliary discount net supervision
        self.risks = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Scalar total advantage for policy gradient
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Component advantages for logging & analysis
        self.advantages_per_head = np.zeros((self.buffer_size, self.n_envs, self.critic_dim), dtype=np.float32)

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
        gammas: np.ndarray,
        risk: Optional[np.ndarray] = None,
    ) -> None:
        """
        Adds a transition to the buffer.
        """
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()

        # Handle vector reward
        r_arr = np.array(reward, dtype=np.float32)
        if self.critic_dim == 2:
            if r_arr.ndim == 1:
                # If scalar passed, duplicate or reshape
                r_arr = np.column_stack([r_arr, np.zeros_like(r_arr)])
            self.rewards[self.pos] = r_arr.reshape((self.n_envs, 2))
        else:
            self.rewards[self.pos] = r_arr.reshape((self.n_envs, 1))

        # Handle value
        val_np = value.clone().cpu().numpy()
        self.values[self.pos] = val_np.reshape((self.n_envs, self.critic_dim))

        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().flatten()

        # Handle per-step discount factors
        g_arr = np.array(gammas, dtype=np.float32).reshape((self.n_envs, self.critic_dim))
        self.gammas[self.pos] = g_arr

        # Handle risk
        if risk is not None:
            self.risks[self.pos] = np.array(risk, dtype=np.float32).flatten()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Computes GAE per objective stream using per-step state-dependent discount factors.
        """
        last_values_cpu = last_values.clone().cpu().numpy().reshape((self.n_envs, self.critic_dim))

        for head_idx in range(self.critic_dim):
            last_gae_lam = 0.0
            last_val_head = last_values_cpu[:, head_idx]

            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_values = last_val_head
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_values = self.values[step + 1, :, head_idx]

                # Per-step discount factor for this head
                gamma_step = self.gammas[step, :, head_idx]

                # TD error δ_t = r_t + γ_t * V(s_{t+1}) * (1 - d_{t+1}) - V(s_t)
                delta = (
                    self.rewards[step, :, head_idx]
                    + gamma_step * next_values * next_non_terminal
                    - self.values[step, :, head_idx]
                )

                # GAE recurrence: A_t = δ_t + γ_t * λ * (1 - d_{t+1}) * A_{t+1}
                last_gae_lam = delta + gamma_step * self.gae_lambda * next_non_terminal * last_gae_lam

                self.returns[step, :, head_idx] = last_gae_lam + self.values[step, :, head_idx]
                self.advantages_per_head[step, :, head_idx] = last_gae_lam

        # Compute scalarized total advantage for policy gradient
        if self.critic_dim == 2:
            norm_sum = self.weight_l + self.weight_s
            w_l = self.weight_l / norm_sum if norm_sum > 0 else 0.5
            w_s = self.weight_s / norm_sum if norm_sum > 0 else 0.5
            self.advantages = (
                w_l * self.advantages_per_head[:, :, 0] + w_s * self.advantages_per_head[:, :, 1]
            )
        else:
            self.advantages = self.advantages_per_head[:, :, 0]

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[Any] = None) -> RolloutBufferSamples:
        """
        Samples a batch. Values and returns retain shape (batch_size, critic_dim) or (batch_size*critic_dim,)
        """
        flat_inds = batch_inds
        data = (
            self.observations[flat_inds],
            self.actions[flat_inds],
            self.values[flat_inds].flatten() if self.critic_dim > 1 else self.values[flat_inds].squeeze(-1),
            self.log_probs[flat_inds],
            self.advantages[flat_inds],
            self.returns[flat_inds].flatten() if self.critic_dim > 1 else self.returns[flat_inds].squeeze(-1),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def get_extra_samples(self, batch_inds: np.ndarray) -> Dict[str, th.Tensor]:
        """
        Returns additional tensors (risks, gammas, unflattened returns) needed for Exp C.
        """
        return {
            "risks": self.to_torch(self.risks[batch_inds]),
            "gammas": self.to_torch(self.gammas[batch_inds]),
            "returns_unflat": self.to_torch(self.returns[batch_inds]),
            "values_unflat": self.to_torch(self.values[batch_inds]),
            "adv_l": self.to_torch(self.advantages_per_head[batch_inds, :, 0]) if self.critic_dim > 1 else self.to_torch(self.advantages[batch_inds]),
            "adv_s": self.to_torch(self.advantages_per_head[batch_inds, :, 1]) if self.critic_dim > 1 else th.zeros_like(self.to_torch(self.advantages[batch_inds])),
        }


class MOSDPPO(PPO):
    """
    Multi-Objective, State-Dependent Proximal Policy Optimization.
    """

    def __init__(
        self,
        policy: Union[str, Type[MultiObjectiveActorCriticPolicy]],
        env: Union[VecEnv, str],
        mode: str = "exp_b",
        gamma_0: float = 0.99,
        gamma_l: float = 0.99,
        gamma_s_normal: float = 0.95,
        gamma_s_danger: float = 0.0,
        weight_l: float = 0.5,
        weight_s: float = 0.5,
        learnable_discount_net: Optional[LearnableDiscountNet] = None,
        lambda_reg_discount: float = 0.01,
        critic_dim: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.mode = mode.lower()
        self.gamma_0 = gamma_0
        self.gamma_l = gamma_l
        self.gamma_s_normal = gamma_s_normal
        self.gamma_s_danger = gamma_s_danger
        self.weight_l = weight_l
        self.weight_s = weight_s
        self.learnable_discount_net = learnable_discount_net
        self.lambda_reg_discount = lambda_reg_discount

        # Infer critic_dim
        if critic_dim is None:
            self.critic_dim = 2 if self.mode in ("exp_b", "exp_c") else 1
        else:
            self.critic_dim = critic_dim

        # Ensure policy_kwargs passes critic_dim
        if "policy_kwargs" not in kwargs or kwargs["policy_kwargs"] is None:
            kwargs["policy_kwargs"] = {}
        kwargs["policy_kwargs"]["critic_dim"] = self.critic_dim

        super().__init__(policy, env, *args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()

        # Instantiate our custom multi-objective buffer
        self.rollout_buffer = MultiObjectiveRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma_0,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            critic_dim=self.critic_dim,
            weight_l=self.weight_l,
            weight_s=self.weight_s,
        )

        if self.mode == "exp_c" and self.learnable_discount_net is None:
            obs_dim = int(np.prod(self.observation_space.shape))
            self.learnable_discount_net = LearnableDiscountNet(
                input_dim=obs_dim,
                gamma_0_l=self.gamma_l,
                gamma_0_s=self.gamma_s_normal,
            ).to(self.device)

    def compute_step_discounts(
        self,
        obs: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the discount factor(s) for each environment worker at the current step.
        Returns: (gammas: [n_envs, critic_dim], risks: [n_envs])
        """
        n = self.n_envs
        gammas = np.zeros((n, self.critic_dim), dtype=np.float32)
        risks = np.zeros(n, dtype=np.float32)

        for i in range(n):
            info_i = infos[i] if i < len(infos) else {}
            conflict_info = info_i.get("conflict_info", {})
            is_conflict = conflict_info.get("is_conflict", False)
            risk_val = float(conflict_info.get("conflict_risk", 1.0 if is_conflict else 0.0))
            risks[i] = risk_val

            if self.mode == "baseline":
                # Single fixed discount
                gammas[i, 0] = self.gamma_0

            elif self.mode == "exp_a":
                # State-dependent single discount: short horizon when in conflict
                gammas[i, 0] = self.gamma_s_danger if is_conflict else self.gamma_0

            elif self.mode == "exp_b":
                # Multi-objective state-dependent discounts: [gamma_l, gamma_s(s)]
                gammas[i, 0] = self.gamma_l
                gammas[i, 1] = self.gamma_s_danger if is_conflict else self.gamma_s_normal

            elif self.mode == "exp_c":
                # Handled via batch neural network forward pass below
                pass

            elif self.mode == "ablation":
                # State-dependent reward weighting: fixed discount
                gammas[i, 0] = self.gamma_0

        if self.mode == "exp_c" and self.learnable_discount_net is not None:
            with th.no_grad():
                obs_t = th.as_tensor(obs, device=self.device, dtype=th.float32)
                g_l_t, g_s_t, _ = self.learnable_discount_net(obs_t)
                gammas[:, 0] = g_l_t.cpu().numpy().flatten()
                gammas[:, 1] = g_s_t.cpu().numpy().flatten()

        return gammas, risks

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MultiObjectiveRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collects experiences, extracts vector rewards and conflict metrics,
        evaluates state-dependent discounts, and populates the rollout buffer.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            # Compute state-dependent discounts based on the step transition
            gammas, risks = self.compute_step_discounts(new_obs, infos)

            # Extract decomposed vector rewards if present
            if self.critic_dim == 2:
                vec_rewards = np.zeros((env.num_envs, 2), dtype=np.float32)
                for idx, info in enumerate(infos):
                    if "vector_reward" in info:
                        vec_rewards[idx] = info["vector_reward"]
                    else:
                        vec_rewards[idx] = [rewards[idx], 0.0]
                step_rewards = vec_rewards
            else:
                step_rewards = np.array(rewards, dtype=np.float32).reshape((env.num_envs, 1))

            # Suppress SB3's corrupted scalar addition on truncation
            for info in infos:
                if "TimeLimit.truncated" in info:
                    info["TimeLimit.truncated"] = False

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,
                actions,
                step_rewards,
                self._last_episode_starts,
                values,
                log_probs,
                gammas=gammas,
                risk=risks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Updates policy and value parameters using PPO surrogate loss.
        Also trains LearnableDiscountNet in Exp C.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                # Reshape for multi-head values
                if self.critic_dim > 1:
                    values = values.flatten()
                else:
                    values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                target_returns = rollout_data.returns
                value_loss = F.mse_loss(target_returns, values)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approx_kl
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        # Exp C: Train LearnableDiscountNet
        if self.mode == "exp_c" and self.learnable_discount_net is not None:
            all_obs = th.as_tensor(self.rollout_buffer.observations.reshape(-1, *self.rollout_buffer.obs_shape), device=self.device, dtype=th.float32)
            all_risks = th.as_tensor(self.rollout_buffer.risks.flatten(), device=self.device, dtype=th.float32)

            discount_loss, discount_metrics = self.learnable_discount_net.compute_loss(
                all_obs, target_risk=all_risks, lambda_reg=self.lambda_reg_discount
            )

            self.learnable_discount_net.optimizer.zero_grad()
            discount_loss.backward()
            self.learnable_discount_net.optimizer.step()

            for k, v in discount_metrics.items():
                self.logger.record(k, v)

        self._n_updates += self.n_epochs
        explained_var = self.rollout_buffer.values.flatten()
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())

        # Log component advantages and discounts
        self.logger.record("discount/mean_adv", float(self.rollout_buffer.advantages.mean()))
        if self.critic_dim > 1:
            self.logger.record("discount/mean_adv_l", float(self.rollout_buffer.advantages_per_head[:, :, 0].mean()))
            self.logger.record("discount/mean_adv_s", float(self.rollout_buffer.advantages_per_head[:, :, 1].mean()))
            self.logger.record("discount/mean_gamma_l", float(self.rollout_buffer.gammas[:, :, 0].mean()))
            self.logger.record("discount/mean_gamma_s", float(self.rollout_buffer.gammas[:, :, 1].mean()))
        else:
            self.logger.record("discount/mean_gamma", float(self.rollout_buffer.gammas[:, :, 0].mean()))

