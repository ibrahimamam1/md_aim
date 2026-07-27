#============ATTENTION + DISCRETE ACTION SPACE VARIANT OF V0.1 ENV============

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import sys 
import os 

sys.path.append(os.path.dirname(__file__))

from alpha_env_v01_attention_continous import AlphaEnv_v01_Attention

class AlphaEnv_v01_AttentionDiscrete(AlphaEnv_v01_Attention):
    """
    Same as AlphaEnv_v01_Attention but with a discrete action space.
    5 bins: [-1.0, -0.5, 0.0, 0.5, 1.0]
    Mapped to: [-max_decel, -max_decel/2, 0, max_accel/2, max_accel]
    """
    
    ACCEL_BINS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        
        # Override action space to discrete
        self.action_space = Discrete(len(self.ACCEL_BINS))

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        # Map discrete action index to normalized value
        action_idx = int(rl_action)
        action_val = self.ACCEL_BINS[action_idx]
        
        # Denormalize from [-1, 1] to [-max_decel, max_accel]
        if action_val >= 0:
            real_action = action_val * max_accel
        else:
            real_action = action_val * max_decel

        rl_ids = []
        rl_ids.append(self.agent_id)
        if not rl_ids:
            return
        self.k.vehicle.apply_acceleration(rl_ids, [real_action])


    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        if agent_id not in self.k.vehicle.get_ids():
            return 0.0
        
        # 1. Sparse Terminal Rewards
        if fail:           return -10.0
        if goal_reached:   return 15.0
        
        obs_info = getattr(self, 'last_neighbors_info', []) 
        
        # 2. Progress Reward
        ego_dis = self.k.vehicle.get_distance(agent_id)
        if ego_dis == -1001: ego_dis = 0.0
        route = self.k.vehicle.get_route(agent_id)
        total_route_length = max(sum([self.k.network.edge_length(e) for e in route]), 1e-4)
        
        progress_norm = np.clip(ego_dis / total_route_length, 0.0, 1.0)
        
        if not hasattr(self, 'last_progress'):
            self.last_progress = progress_norm
            
        progress_delta = progress_norm - self.last_progress 
        self.last_progress = progress_norm
        
        # 3. Safety Penalty
        safety_penalty = 0.0
        for n in obs_info: 
            abs_d_eta = abs(n['d_eta'])
            # Only penalize if they are projected to arrive within a tight window of each other
            if abs_d_eta < 0.2: 
                # Exponential penalty: spikes hard as d_eta approaches 0
                safety_penalty += -np.exp(-abs_d_eta * 10.0) 

        # 4. Dense Reward Assembly
        return (
            10.0 * progress_delta     # reward for moving towards goal 
            + 1.0 * safety_penalty     # Penalty for crossing intersection unsafely
            - 0.01                     # Time penalty
        )
