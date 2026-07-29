#============ATTENTION-BASED VARIANT OF V0.1 ENV (NO CONFLICT HEURISTIC)============
from gymnasium.spaces import Box
import numpy as np
import sys 
import os 
from shapely.geometry import LineString, Point
sys.path.append(os.path.dirname(__file__))

from alpha_env_v01 import AlphaEnv_v01

class AlphaEnv_v01_Attention(AlphaEnv_v01):
    """
    Multi-Agent Alpha environment with stability fixes.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 100
       
        # Ego-centric observation: S_ego = [d_norm, v_norm, cos θ, sin θ]
        self.ego_obs_features = 4
        # Per-neighbor (ego-relative): S_i = [ego_d_to_cp, other_dist_to_cp, v, cos Δθ, sin Δθ]
        self.neighbour_obs_features = 5
        
        super().__init__(env_params, sim_params, network, simulator)
        
        # Defining action space - KEEP NORMALIZED
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(1, ), 
            dtype=np.float32)

        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours) + self.max_neighbours
        self.observation_space = Box(
            low=-1.0,  
            high=1.0,   
            shape=(total_obs_len, ),
            dtype=np.float32)

        self.last_action = 0.0
        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def get_state(self):
        self._update_routes()
        rl_ids = self.k.vehicle.get_rl_ids()
        if self.agent_id not in rl_ids:
            return self.last_obs
     
        obs, neighbors_info = self._get_local_observation(self.agent_id)
        self.last_obs = obs
        self.last_neighbors_info = neighbors_info  # cache for terminal step
        return obs

    def _get_local_observation(self, ego_id):
        obs_array, neighbors_info = super()._get_local_observation(ego_id)
        if isinstance(neighbors_info, dict):
            return obs_array, neighbors_info

        num_actual = len(neighbors_info)
        neighbor_mask = [1.0] * num_actual + [0.0] * (self.max_neighbours - num_actual)
        extended_obs = np.concatenate([obs_array, neighbor_mask], dtype=np.float32)
        return extended_obs, neighbors_info

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        # 1. Safely extract and sanitize the action
        try:
            action_val = float(rl_action[0]) if isinstance(rl_action, (list, np.ndarray)) else float(rl_action)
        except (TypeError, ValueError):
            action_val = 0.0
            
        if np.isnan(action_val) or np.isinf(action_val):
            action_val = 0.0  # Fallback to zero acceleration if NaN

        # Denormalize from [-1, 1] to [-max_decel, max_accel]
        if action_val >= 0:
            real_action = action_val * max_accel
        else:
            real_action = action_val * max_decel

        rl_ids = [self.agent_id]
        if self.agent_id in self.k.vehicle.get_ids():
            self.k.vehicle.apply_acceleration(rl_ids, [real_action])


