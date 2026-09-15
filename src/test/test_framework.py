"""
test_framework.py
─────────────────
Comprehensive automated test suite verifying all research components:
  1. MultiObjectiveRolloutBuffer with state-dependent GAE recursions.
  2. MultiObjectiveActorCriticPolicy (dual & single heads).
  3. LearnableDiscountNet (structural risk prior and bounded output).
  4. S1..S7 scenario definitions and parameters.
  5. MOSDPPO algorithm initialization and rollout step discount logic.
"""

import os
import sys
import unittest
import numpy as np
import torch as th
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.mo_sd_models import MultiObjectiveActorCriticPolicy, LearnableDiscountNet
from src.models.mo_sd_ppo import MultiObjectiveRolloutBuffer, MOSDPPO
from src.scenarios.traffic_scenarios import get_scenario_definition


class TestMultiObjectiveStateDependentFramework(unittest.TestCase):

    def setUp(self):
        self.obs_dim = 34  # Attention env dimension
        self.action_dim = 1
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def test_learnable_discount_net(self):
        """Verify LearnableDiscountNet bounds, structural prior, and loss computation."""
        net = LearnableDiscountNet(
            input_dim=self.obs_dim,
            hidden_dim=64,
            gamma_min=0.0,
            gamma_max=0.99,
            gamma_0_l=0.99,
            gamma_0_s=0.95,
            use_structural_prior=True,
        )

        batch_size = 8
        dummy_obs = th.randn(batch_size, self.obs_dim)
        gamma_l, gamma_s, risk = net(dummy_obs)

        # Verify shapes
        self.assertEqual(gamma_l.shape, (batch_size, 1))
        self.assertEqual(gamma_s.shape, (batch_size, 1))
        self.assertEqual(risk.shape, (batch_size, 1))

        # Verify bounds: 0.0 <= gamma <= 0.99
        self.assertTrue(th.all(gamma_l >= 0.0) and th.all(gamma_l <= 0.99))
        self.assertTrue(th.all(gamma_s >= 0.0) and th.all(gamma_s <= 0.99))
        self.assertTrue(th.all(risk >= 0.0) and th.all(risk <= 1.0))

        # Test loss computation
        dummy_target_risk = th.tensor([0.0, 1.0, 0.5, 0.0, 1.0, 0.2, 0.8, 0.0]).unsqueeze(1)
        loss, metrics = net.compute_loss(dummy_obs, target_risk=dummy_target_risk)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("discount_net/loss", metrics)
        self.assertIn("discount_net/reg_loss", metrics)

    def test_multi_objective_policy(self):
        """Verify dual-head critic and policy outputs."""
        def dummy_lr(p):
            return 3e-4

        # Dual-head policy
        policy = MultiObjectiveActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=dummy_lr,
            critic_dim=2,
        )

        dummy_obs = th.randn(4, self.obs_dim)
        actions, values, log_probs = policy(dummy_obs)

        self.assertEqual(actions.shape, (4, 1))
        self.assertEqual(values.shape, (4, 2))  # [batch, 2] for dual critic
        self.assertEqual(log_probs.shape, (4,))

        # Predict values method
        val_pred = policy.predict_values(dummy_obs)
        self.assertEqual(val_pred.shape, (4, 2))

    def test_state_dependent_gae_buffer(self):
        """Verify MultiObjectiveRolloutBuffer GAE recursion with state-dependent discounts."""
        buffer_size = 10
        n_envs = 2
        buf = MultiObjectiveRolloutBuffer(
            buffer_size=buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_envs=n_envs,
            critic_dim=2,
            weight_l=0.6,
            weight_s=0.4,
            gae_lambda=0.95,
        )
        buf.reset()

        # Add 10 steps of transitions with varying discounts
        for t in range(buffer_size):
            obs = np.random.randn(n_envs, self.obs_dim).astype(np.float32)
            act = np.random.uniform(-1.0, 1.0, size=(n_envs, 1)).astype(np.float32)
            # Vector reward: r_l and r_s
            reward = np.array([[0.1, -0.05], [0.2, -0.1]], dtype=np.float32)
            starts = np.zeros(n_envs, dtype=np.float32)
            vals = th.tensor([[1.0, -0.5], [1.2, -0.4]], dtype=th.float32)
            log_prob = th.tensor([0.1, 0.2], dtype=th.float32)

            # State-dependent discounts: gamma_l=0.99, gamma_s=0.0 during danger at step 5
            if t == 5:
                gammas = np.array([[0.99, 0.0], [0.99, 0.0]], dtype=np.float32)
            else:
                gammas = np.array([[0.99, 0.95], [0.99, 0.95]], dtype=np.float32)

            buf.add(obs, act, reward, starts, vals, log_prob, gammas=gammas)

        self.assertTrue(buf.full)

        last_vals = th.tensor([[1.5, -0.2], [1.4, -0.3]], dtype=th.float32)
        dones = np.zeros(n_envs, dtype=np.float32)

        # Compute returns and advantage
        buf.compute_returns_and_advantage(last_vals, dones)

        # Verify shapes
        self.assertEqual(buf.returns.shape, (buffer_size, n_envs, 2))
        self.assertEqual(buf.advantages.shape, (buffer_size, n_envs))
        self.assertEqual(buf.advantages_per_head.shape, (buffer_size, n_envs, 2))

        # Check scalarized advantage formula: A = 0.6 * A_l + 0.4 * A_s
        expected_adv = 0.6 * buf.advantages_per_head[:, :, 0] + 0.4 * buf.advantages_per_head[:, :, 1]
        np.testing.assert_allclose(buf.advantages, expected_adv, rtol=1e-5)

        # Verify that at step 5 (where gamma_s = 0.0), A_s is purely immediate TD error:
        # delta = r_s + 0 - V_s = -0.05 - (-0.5) = 0.45
        delta_expected_0 = -0.05 - (-0.5)
        self.assertAlmostEqual(buf.advantages_per_head[5, 0, 1], delta_expected_0, places=4)

    def test_scenarios_definitions(self):
        """Verify that all 7 scenarios S1..S7 instantiate valid configs."""
        for s_id in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
            scen_def = get_scenario_definition(s_id, self.root_dir)
            self.assertEqual(scen_def["scenario_id"], s_id)
            self.assertIn("description", scen_def)
            self.assertIsNotNone(scen_def["net_params"])
            self.assertIsNotNone(scen_def["vehicles"])
            self.assertTrue(os.path.exists(scen_def["net_file"]), f"Net file missing: {scen_def['net_file']}")

    def test_progress_tracking_and_terminal_goal(self):
        """Verify monotonic progress accumulation, terminal departure handling, and lack of spikes."""
        from src.scenarios.traffic_scenarios import create_scenario_env

        env = create_scenario_env("S1", root_dir=self.root_dir)
        obs, info = env.reset()
        self.assertGreater(env.total_route_length, 80.0)
        self.assertEqual(env.prev_progress, 0.0)

        reached_goal = False
        deltas = []

        for step in range(80):
            conflict = env.last_conflict_info
            if conflict["min_ttc"] < 3.0 or conflict["min_gap"] < 10.0:
                action = [-0.8]
            else:
                action = [0.8]

            obs, r, term, trunc, info = env.step(action)
            r_dict = info.get("reward_dict", {})
            p_delta = r_dict.get("progress_delta", 0.0)
            deltas.append(p_delta)

            # Monotonicity check
            self.assertGreaterEqual(p_delta, 0.0)

            if term or trunc:
                tele = info.get("mo_telemetry", {})
                if tele.get("success") == 1:
                    reached_goal = True
                    # Terminal delta must be a smooth continuation, not a giant 1.0 leap
                    self.assertLessEqual(p_delta, 0.20)
                    # Cumulative progress must sum to exactly 1.0
                    self.assertAlmostEqual(sum(deltas), 1.0, places=4)
                    self.assertAlmostEqual(tele.get("cumulative_progress", 0.0), 1.0, places=4)
                    # Terminal r_l must include goal reward (15.0)
                    self.assertGreater(r_dict.get("r_l", 0.0), 14.0)

                    # Verify reward decomposition components and additive identities
                    r_prog = tele.get("progress_reward", 0.0)
                    r_goal = tele.get("goal_reward", 0.0)
                    r_time = tele.get("time_penalty", 0.0)
                    r_gap = tele.get("gap_penalty", 0.0)
                    r_col = tele.get("collision_penalty", 0.0)
                    r_l_tot = tele.get("total_long_term_reward", 0.0)
                    r_s_tot = tele.get("total_safety_reward", 0.0)
                    r_tot = tele.get("total_reward", 0.0)

                    # Check R_l = R_progress + R_goal + R_time
                    self.assertAlmostEqual(r_l_tot, r_prog + r_goal + r_time, places=4)
                    # Check R_s = R_gap + R_collision
                    self.assertAlmostEqual(r_s_tot, r_gap + r_col, places=4)
                    # Check scalar total reward = 0.5 * R_l + 0.5 * R_s
                    self.assertAlmostEqual(r_tot, 0.5 * r_l_tot + 0.5 * r_s_tot, places=4)
                break

        env.close()
        self.assertTrue(reached_goal, "Test agent should safely reach the goal in scenario S1")

    def test_baseline_uses_standard_sb3_buffer(self):
        """Verify that in baseline mode, MOSDPPO uses native SB3 RolloutBuffer with gamma_0."""
        from stable_baselines3.common.buffers import RolloutBuffer
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym

        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(-1.0, 1.0, shape=(34,), dtype=np.float32)
                self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
            def reset(self, **kwargs):
                return np.zeros(34, dtype=np.float32), {}
            def step(self, a):
                return np.zeros(34, dtype=np.float32), 1.0, False, False, {}

        vec_env = DummyVecEnv([MockEnv])
        gamma_test = 0.97

        model = MOSDPPO(
            policy=MultiObjectiveActorCriticPolicy,
            env=vec_env,
            mode="baseline",
            gamma_0=gamma_test,
            n_steps=64,
            batch_size=32,
        )

        # Ensure buffer is native SB3 RolloutBuffer, not MultiObjectiveRolloutBuffer
        self.assertIs(type(model.rollout_buffer), RolloutBuffer)
        self.assertIsNot(type(model.rollout_buffer), MultiObjectiveRolloutBuffer)
        self.assertEqual(model.critic_dim, 1)
        self.assertAlmostEqual(model.gamma, gamma_test)
        self.assertAlmostEqual(model.rollout_buffer.gamma, gamma_test)

    def test_multiobjective_rollout_buffer_samples_are_strictly_1d(self):
        """Verify that MultiObjectiveRolloutBuffer._get_samples produces strictly 1D tensors, preventing (B, B) broadcasting."""
        buffer_size = 16
        batch_size = 8
        n_envs = 1

        buf = MultiObjectiveRolloutBuffer(
            buffer_size=buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_envs=n_envs,
            critic_dim=2,
        )
        buf.reset()

        for _ in range(buffer_size):
            obs = np.random.randn(n_envs, self.obs_dim).astype(np.float32)
            act = np.random.uniform(-1.0, 1.0, size=(n_envs, 1)).astype(np.float32)
            rew = np.array([[0.5, -0.2]], dtype=np.float32)
            starts = np.zeros(n_envs, dtype=np.float32)
            vals = th.tensor([[0.2, -0.1]], dtype=th.float32)
            lp = th.tensor([0.1], dtype=th.float32)
            buf.add(obs, act, rew, starts, vals, lp, gammas=np.array([[0.99, 0.95]]))

        buf.compute_returns_and_advantage(th.tensor([[0.2, -0.1]]), np.zeros(n_envs))

        for sample in buf.get(batch_size=batch_size):
            # Crucial SB3 1D contracts
            self.assertEqual(sample.old_log_prob.shape, (batch_size,))
            self.assertEqual(sample.advantages.shape, (batch_size,))
            self.assertEqual(sample.returns.shape, (batch_size * 2,))
            self.assertEqual(sample.old_values.shape, (batch_size * 2,))

            # Verify no broadcasting occurs with policy log_prob
            mock_policy_log_prob = th.randn(batch_size)
            diff = mock_policy_log_prob - sample.old_log_prob
            self.assertEqual(diff.shape, (batch_size,), "Diff must NOT broadcast to (B, B)!")
            ratio = th.exp(diff)
            self.assertEqual(ratio.shape, (batch_size,))
            loss_term = sample.advantages * ratio
            self.assertEqual(loss_term.shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()

