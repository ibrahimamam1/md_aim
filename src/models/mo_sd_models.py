"""
mo_sd_models.py
───────────────
Neural network architectures for Multi-Objective, State-Dependent Discounting.

1. MultiObjectiveActorCriticPolicy:
   - Configurable for dual value heads [V_l(s), V_s(s)] (Exp B, Exp C)
     or single value head V(s) (Baseline, Exp A, Ablation).
   - Compatible with AttentionFeatureExtractor and MLP trunks.

2. LearnableDiscountNet (Experiment C):
   - Implements Section 10 & 11 learnable discount architecture f_φ(s) = [γ_l(s), γ_s(s)].
   - Enforces bounding: γ_min <= γ_i(s) <= γ_max.
   - Provides structural risk prior: γ_s(s) = γ_max - (γ_max - γ_min) * ρ_φ(s)
     where ρ_φ(s) ∈ [0, 1] represents learned conflict risk.
   - Anti-cheating regularized loss: L_reg = λ_γ * ||γ(s) - γ_0||^2 to prevent
     the agent from artificially shrinking horizons to ignore catastrophic events.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


class MultiObjectiveActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy supporting both single (dim=1) and multi-objective (dim=2) value heads.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        critic_dim: int = 2,
        *args,
        **kwargs,
    ):
        self.critic_dim = critic_dim
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        # Replace default 1-output value head with critic_dim outputs
        if self.critic_dim != 1:
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.critic_dim)

            # Re-instantiate optimizer so the new value_net weights are tracked
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict value(s) for a given observation tensor.
        Returns: [batch_size, critic_dim] (or [batch_size, 1]).
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_vf = self.mlp_extractor.forward_critic(features)
        else:
            pi_features, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self.value_net(latent_vf)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass for policy, value, and log_prob.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for given latent features
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        returning values [batch_size, critic_dim], log_prob, and entropy.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


class LearnableDiscountNet(nn.Module):
    """
    Learnable discount network γ_φ(s) = [γ_l(s), γ_s(s)] for Experiment C.

    Features:
      1. Bounded range: γ_min <= γ_i(s) <= γ_max via sigmoid scaling.
      2. Structural risk prior option:
         - Output predicted risk ρ_φ(s) ∈ [0, 1].
         - γ_s(s) = γ_max - (γ_max - γ_min) * ρ_φ(s).
         - High risk naturally contracts the safety horizon without reward cheating.
      3. Anti-cheating L2 regularization to anchor discounts against baseline γ_0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        gamma_min: float = 0.0,
        gamma_max: float = 0.99,
        gamma_0_l: float = 0.99,
        gamma_0_s: float = 0.95,
        use_structural_prior: bool = True,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_0_l = gamma_0_l
        self.gamma_0_s = gamma_0_s
        self.use_structural_prior = use_structural_prior

        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if use_structural_prior:
            # Efficiency head (outputs raw logit for γ_l)
            self.efficiency_head = nn.Linear(hidden_dim, 1)
            # Risk head (outputs raw logit for conflict risk ρ ∈ [0, 1])
            self.risk_head = nn.Linear(hidden_dim, 1)
        else:
            # Direct 2-output head for [γ_l, γ_s]
            self.discount_head = nn.Linear(hidden_dim, 2)

        self.optimizer = th.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass.
        Returns:
            gamma_l: [batch_size, 1] bounded discount for efficiency
            gamma_s: [batch_size, 1] bounded discount for safety
            risk_pred: [batch_size, 1] predicted conflict risk
        """
        feat = self.trunk(obs)

        if self.use_structural_prior:
            raw_eff = self.efficiency_head(feat)
            raw_risk = self.risk_head(feat)

            # Efficiency discount bounded in [gamma_min, gamma_max]
            # Initialize close to gamma_0_l
            gamma_l = self.gamma_min + (self.gamma_max - self.gamma_min) * th.sigmoid(raw_eff)

            # Conflict risk ρ ∈ [0, 1]
            risk_pred = th.sigmoid(raw_risk)

            # Structural prior: higher risk -> lower gamma_s
            gamma_s = self.gamma_max - (self.gamma_max - self.gamma_min) * risk_pred
        else:
            raw_discounts = self.discount_head(feat)
            scaled = self.gamma_min + (self.gamma_max - self.gamma_min) * th.sigmoid(raw_discounts)
            gamma_l = scaled[:, 0:1]
            gamma_s = scaled[:, 1:2]
            risk_pred = 1.0 - (gamma_s - self.gamma_min) / max(self.gamma_max - self.gamma_min, 1e-6)

        return gamma_l, gamma_s, risk_pred

    def compute_loss(
        self,
        obs: th.Tensor,
        target_risk: Optional[th.Tensor] = None,
        lambda_reg: float = 0.01,
        lambda_risk: float = 1.0,
    ) -> Tuple[th.Tensor, Dict[str, float]]:
        """
        Computes regularization loss and optional auxiliary risk prediction loss.
        """
        gamma_l, gamma_s, risk_pred = self.forward(obs)

        # L2 Regularization towards anchor priors
        reg_l = th.mean((gamma_l - self.gamma_0_l) ** 2)
        reg_s = th.mean((gamma_s - self.gamma_0_s) ** 2)
        reg_loss = lambda_reg * (reg_l + reg_s)

        # Auxiliary risk supervision loss if environment provided conflict signal
        risk_loss = th.tensor(0.0, device=obs.device)
        if target_risk is not None and self.use_structural_prior:
            target_risk_t = target_risk.view_as(risk_pred).float()
            risk_loss = lambda_risk * F.binary_cross_entropy(risk_pred, target_risk_t)

        total_loss = reg_loss + risk_loss

        metrics = {
            "discount_net/loss": float(total_loss.item()),
            "discount_net/reg_loss": float(reg_loss.item()),
            "discount_net/risk_loss": float(risk_loss.item()) if target_risk is not None else 0.0,
            "discount_net/mean_gamma_l": float(gamma_l.mean().item()),
            "discount_net/mean_gamma_s": float(gamma_s.mean().item()),
            "discount_net/mean_risk": float(risk_pred.mean().item()),
        }
        return total_loss, metrics

