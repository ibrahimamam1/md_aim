#!/usr/bin/env python3
"""
Unit tests for Frenet distance computation, Delta ETA normalization,
car-following (leader-follower) behavior, and static conflict map (specifically West edge).
"""

import unittest
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString, GeometryCollection
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.alpha_env_v01 import AlphaEnv_v01
from src.envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention


class DummyKernelVehicle:
    def __init__(self, lengths=None):
        self.lengths = lengths or {}

    def get_length(self, veh_id):
        return self.lengths.get(veh_id, 5.0)


class TestFrenetAndConflicts(unittest.TestCase):
    def setUp(self):
        # Create a dummy instance of AlphaEnv_v01 without initializing SUMO kernel
        self.env = AlphaEnv_v01.__new__(AlphaEnv_v01)
        self.env.perception_radius = 50.0
        self.env.conflict_map = self.env._build_conflict_map()
        self.env.k = type('DummyK', (), {'vehicle': DummyKernelVehicle({'ego': 5.0, 'leader': 5.0, 'other': 5.0})})()

    def test_conflict_map_west_edge(self):
        """1. Check static conflict map and ensure all conflicts with West edge are correctly set."""
        cmap = self.env.conflict_map
        # Check WE straight from West ('E#L-X', 'E#X-R')
        we_route = ('E#L-X', 'E#X-R')
        self.assertIn(we_route, cmap)
        we_conflicts = cmap[we_route]
        # Must conflict with crossing straights (NS, SN), left turns (ES, NE, SW),
        # merging right turn SE ('E#D-X', 'E#X-R'), and itself WE for car-following across intersection!
        self.assertIn(we_route, we_conflicts) # WE (same-route leader tracking)
        self.assertIn(('E#D-X', 'E#X-R'), we_conflicts) # SE (merging right turn)
        self.assertIn(('E#T-X', 'E#X-D'), we_conflicts) # NS
        self.assertIn(('E#D-X', 'E#X-T'), we_conflicts) # SN
        self.assertIn(('E#R-X', 'E#X-D'), we_conflicts) # ES
        self.assertIn(('E#T-X', 'E#X-R'), we_conflicts) # NE
        self.assertIn(('E#D-X', 'E#X-L'), we_conflicts) # SW

        # Check WN left turn from West ('E#L-X', 'E#X-T')
        wn_route = ('E#L-X', 'E#X-T')
        self.assertIn(wn_route, cmap)
        wn_conflicts = cmap[wn_route]
        self.assertIn(wn_route, wn_conflicts)  # WN (same-route leader tracking)

        # Check WS right turn from West ('E#L-X', 'E#X-D')
        ws_route = ('E#L-X', 'E#X-D')
        self.assertIn(ws_route, cmap)
        ws_conflicts = cmap[ws_route]
        self.assertIn(ws_route, ws_conflicts)  # WS (same-route leader tracking)

    def test_frenet_car_following_leader_case(self):
        """
        4. When ego has a leader, desired behavior is other_dist_to_cp = 0 and
        ego_dist_to_cp = frenet distance to leader rear bumper (conflict point).
        And delta eta = ego_eta.
        """
        ego_line = LineString([(0, 0), (100, 0)])
        other_line = LineString([(0, 0), (100, 0)])
        ego_id = "ego"
        other_id = "leader"

        # Ego is at 20.0m, Leader is at 35.0m, leader length is 5.0m
        ego_pos_on_edge = 20.0
        other_pos_on_edge = 35.0
        ego_point = Point(20.0, 0.0)
        other_point = Point(35.0, 0.0)

        # Replicate car-following Case B logic from _get_local_observation
        other_proj = ego_line.project(other_point)
        self.assertGreaterEqual(other_proj, ego_pos_on_edge)

        leader_len = getattr(self.env.k.vehicle, 'get_length', lambda _id: 5.0)(other_id)
        ego_dist_to_cp = max(0.0, other_proj - ego_pos_on_edge - leader_len)
        other_dist_to_cp = 0.0

        self.assertEqual(other_dist_to_cp, 0.0)
        self.assertEqual(ego_dist_to_cp, 10.0)  # (35.0 - 20.0) - 5.0 = 10.0m to rear bumper!

        # Check delta eta computation
        ego_speed = 10.0
        other_speed = 8.0
        ego_eta = ego_dist_to_cp / max(ego_speed, 0.5)
        other_eta = other_dist_to_cp / max(other_speed, 0.5)
        delta_eta = ego_eta - other_eta
        self.assertEqual(other_eta, 0.0)
        self.assertEqual(delta_eta, ego_eta)  # delta_eta == ego_eta!

    def test_frenet_car_following_ego_is_leader(self):
        """Test the reverse case where Ego is the leader and Other is following."""
        ego_line = LineString([(0, 0), (100, 0)])
        other_line = LineString([(0, 0), (100, 0)])
        ego_id = "ego"
        other_id = "follower"

        # Ego is at 50.0m, Follower is at 30.0m
        ego_pos_on_edge = 50.0
        other_pos_on_edge = 30.0
        other_point = Point(30.0, 0.0)

        other_proj = ego_line.project(other_point)
        self.assertLess(other_proj, ego_pos_on_edge)

        ego_len = getattr(self.env.k.vehicle, 'get_length', lambda _id: 5.0)(ego_id)
        ego_dist_to_cp = 0.0
        other_dist_to_cp = max(0.0, ego_pos_on_edge - other_proj - ego_len)

        self.assertEqual(ego_dist_to_cp, 0.0)
        self.assertEqual(other_dist_to_cp, 15.0)  # (50.0 - 30.0) - 5.0 = 15.0m

    def test_frenet_crossing_intersection(self):
        """2. Check Frenet distances for crossing paths (Point intersection)."""
        ego_line = LineString([(0, 10), (100, 10)])
        other_line = LineString([(50, 0), (50, 100)])
        intersection = ego_line.intersection(other_line)
        self.assertEqual(intersection.geom_type, 'Point')

        ego_pos_on_edge = 20.0
        other_pos_on_edge = 2.0

        ego_proj = ego_line.project(intersection)
        other_proj = other_line.project(intersection)

        ego_dist_to_cp = max(0.0, ego_proj - ego_pos_on_edge)
        other_dist_to_cp = max(0.0, other_proj - other_pos_on_edge)

        self.assertEqual(ego_dist_to_cp, 30.0)  # 50.0 - 20.0
        self.assertEqual(other_dist_to_cp, 8.0) # 10.0 - 2.0 = 8.0

    def test_frenet_merging(self):
        """2. Check Frenet distance for merging paths (overlap_start)."""
        ego_line = LineString([(0, 0), (50, 0), (100, 0)])
        other_line = LineString([(50, -50), (50, 0), (100, 0)])
        intersection = ego_line.intersection(other_line)
        self.assertIn(intersection.geom_type, ['LineString', 'MultiLineString', 'GeometryCollection'])

        if hasattr(intersection, 'geoms'):
            candidates = [Point(g.coords[0]) for g in intersection.geoms if hasattr(g, 'coords') and len(g.coords) > 0]
            overlap_start = min(candidates, key=lambda p: ego_line.project(p))
        else:
            overlap_start = Point(intersection.coords[0])

        ego_pos_on_edge = 10.0
        other_pos_on_edge = 10.0

        ego_dist_to_cp = max(0.0, ego_line.project(overlap_start) - ego_pos_on_edge)
        other_dist_to_cp = max(0.0, other_line.project(overlap_start) - other_pos_on_edge)

        self.assertEqual(ego_dist_to_cp, 40.0)  # 50.0 - 10.0
        self.assertEqual(other_dist_to_cp, 40.0)  # 50.0 - 10.0

    def test_delta_eta_normalization_resolution(self):
        """
        3. Check delta eta computation and verify tanh normalisation maintains good resolution.
        """
        # Testing delta_eta / 5.0 scaling vs unscaled tanh
        time_diffs = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0])
        norm_scaled = np.tanh(time_diffs / 5.0)
        norm_unscaled = np.tanh(time_diffs)

        # Unscaled tanh saturates very early: tanh(2.0) = 0.964, tanh(3.0) = 0.995
        self.assertGreater(norm_unscaled[2], 0.95)
        self.assertGreater(norm_unscaled[3], 0.99)

        # Scaled tanh(/ 5.0) maintains smooth, distinguishable resolution:
        # ~0.0, ~0.20, ~0.38, ~0.54, ~0.76, ~0.96
        self.assertAlmostEqual(norm_scaled[0], 0.0, places=2)
        self.assertAlmostEqual(norm_scaled[1], 0.197, places=2)
        self.assertAlmostEqual(norm_scaled[2], 0.380, places=2)
        self.assertAlmostEqual(norm_scaled[3], 0.537, places=2)
        self.assertAlmostEqual(norm_scaled[4], 0.762, places=2)
        self.assertAlmostEqual(norm_scaled[5], 0.964, places=2)

        # Ensure that consecutive values are well-separated (>0.12 difference between 1s, 2s, 3s)
        self.assertGreater(norm_scaled[2] - norm_scaled[1], 0.15)
        self.assertGreater(norm_scaled[3] - norm_scaled[2], 0.15)
        self.assertGreater(norm_scaled[4] - norm_scaled[3], 0.20)

    def test_attention_env_uses_conflict_heuristic(self):
        """Verify that AlphaEnv_v01_Attention filters neighbors using _is_conflicting heuristic."""
        self.assertTrue(hasattr(AlphaEnv_v01_Attention, '_is_conflicting'), "AlphaEnv_v01_Attention must have _is_conflicting attribute")
        self.assertTrue(hasattr(AlphaEnv_v01_Attention, '_build_conflict_map'), "AlphaEnv_v01_Attention must have _build_conflict_map attribute")
        import inspect
        source_att = inspect.getsource(AlphaEnv_v01_Attention._get_local_observation)
        source_base = inspect.getsource(AlphaEnv_v01._get_local_observation)
        self.assertTrue(
            "_is_conflicting(ego_id, other_id)" in source_att or "super()._get_local_observation" in source_att,
            "AlphaEnv_v01_Attention._get_local_observation must check _is_conflicting or call super()._get_local_observation"
        )
        self.assertIn("_is_conflicting(ego_id, other_id)", source_base, "AlphaEnv_v01._get_local_observation must check self._is_conflicting(ego_id, other_id)")

    def test_attention_env_observation_matches_standard_plus_mask(self):
        """Verify that AlphaEnv_v01_Attention._get_local_observation produces the exact same features as AlphaEnv_v01 plus the neighbor mask."""
        env_std = AlphaEnv_v01.__new__(AlphaEnv_v01)
        env_att = AlphaEnv_v01_Attention.__new__(AlphaEnv_v01_Attention)

        for e in [env_std, env_att]:
            e.perception_radius = 100.0
            e.max_neighbours = 5
            e.conflict_map = e._build_conflict_map()
            e.routes = {"ego": ("E#T-X", "E#X-D"), "other1": ("E#T-X", "E#X-D")}
            e.last_obs = np.zeros(29 if isinstance(e, AlphaEnv_v01) and not isinstance(e, AlphaEnv_v01_Attention) else 34, dtype=np.float32)
            e._get_vehicle_polyline = lambda veh_id: (LineString([(0, 0), (0, 100)]), 20.0 if veh_id == "ego" else 30.0)

            class DummyVehicle:
                def get_route(self, _id): return ("E#T-X", "E#X-D")
                def get_distance(self, _id): return 20.0 if _id == "ego" else 30.0
                def get_speed(self, _id): return 10.0
                def get_heading(self, _id): return 0.0
                def get_ids(self): return ["ego", "other1"]
                def get_2d_position(self, _id): return (0.0, 20.0) if _id == "ego" else (0.0, 30.0)
                def get_edge(self, _id): return "E#T-X"
                def get_position(self, _id): return 20.0 if _id == "ego" else 30.0
                def get_length(self, _id): return 5.0

            class DummyNetwork:
                def edge_length(self, _edge): return 100.0
                def max_speed(self): return 30.0

            e.k = type('DummyK', (), {'vehicle': DummyVehicle(), 'network': DummyNetwork()})()

        obs_std, info_std = env_std._get_local_observation("ego")
        obs_att, info_att = env_att._get_local_observation("ego")

        # 1. Neighbor info must match
        self.assertEqual(len(info_std), len(info_att))
        for k_key in info_std[0].keys():
            self.assertAlmostEqual(info_std[0][k_key], info_att[0][k_key], places=6)

        # 2. Attention observation must be standard observation + neighbor mask (1 actual + 4 padded = [1.0, 0.0, 0.0, 0.0, 0.0])
        expected_mask = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        expected_att_obs = np.concatenate([obs_std, expected_mask])
        np.testing.assert_allclose(obs_att, expected_att_obs, rtol=1e-5, atol=1e-5)

    def test_attention_env_inherits_compute_reward(self):
        """Verify that AlphaEnv_v01_Attention inherits compute_reward from AlphaEnv_v01 without overriding it."""
        self.assertNotIn(
            "compute_reward",
            AlphaEnv_v01_Attention.__dict__,
            "AlphaEnv_v01_Attention must inherit compute_reward from AlphaEnv_v01 without overriding it"
        )

    def test_version_keys_parity_between_train_and_eval(self):
        """Verify that all version keys work cleanly in v0_1_evaluate._get_env_class and support continuous/continous spellings."""
        from src.test.v0_1_evaluate import _get_env_class
        expected_versions = [
            "heuristic_continuous", "heuristic_continous",
            "heuristic_discrete", "heuristic_discrete_3", "heuristic_discrete_5", "heuristic_discrete_10",
            "attention_continuous", "attention_continous",
            "attention_discrete", "attention_discrete_3", "attention_discrete_5", "attention_discrete_10"
        ]
        for ver in expected_versions:
            cls = _get_env_class(ver)
            self.assertTrue(hasattr(cls, "step"), f"Version key '{ver}' did not return a valid environment class")

        self.assertIs(_get_env_class("heuristic_continuous"), _get_env_class("heuristic_continous"))
        self.assertIs(_get_env_class("attention_continuous"), _get_env_class("attention_continous"))

        with self.assertRaises(ValueError):
            _get_env_class("invalid_version")


if __name__ == '__main__':
    unittest.main()

