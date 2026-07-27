import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import sys 
import os 
from shapely.geometry import LineString, Point
sys.path.append(os.path.dirname(__file__))

from base_env_single import Env_N

class AlphaEnv_v01(Env_N):
    """
    Multi-Agent Alpha environment with stability fixes.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = 50
       
        # Ego-centric observation: S_ego = [d_norm, v_norm, cos θ, sin θ]
        self.ego_obs_features = 4
        # Per-neighbor (ego-relative): S_i = [ego_d_to_cp, other_dist_to_cp, v, cos Δθ, sin Δθ]
        self.neighbour_obs_features = 5
        self.routes = dict()
        self.last_progress = 0.0
        super().__init__(env_params, sim_params, network, simulator)
        
        # Initialize the static conflict map
        self.conflict_map = self._build_conflict_map()

        # Defining action space - KEEP NORMALIZED
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(1, ), 
            dtype=np.float32)

        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours)
        self.observation_space = Box(
            low=-1.0,  
            high=1.0,   
            shape=(total_obs_len, ),
            dtype=np.float32)

        self.last_action = 0.0
        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self.last_neighbors_info = []
    
    def _update_routes(self):
        """
        Updates the stored routes for all active vehicles.
        Removes departed vehicles to prevent memory leaks.
        """
        current_ids = self.k.vehicle.get_ids()
            
        # 1. Add routes for new vehicles
        for veh_id in current_ids:
            if veh_id not in self.routes:
                self.routes[veh_id] = self.k.vehicle.get_route(veh_id)
                
        # 2. Cleanup departed vehicles
        active_ids_set = set(current_ids)
        for veh_id in list(self.routes.keys()):
            if veh_id not in active_ids_set:
                del self.routes[veh_id]

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

        # --- 1. Ego State ---
        route = self.k.vehicle.get_route(ego_id)
        total_route_length = sum([self.k.network.edge_length(edge) for edge in route])
        total_route_length = max(total_route_length, 1e-4)
    
        ego_dis = self.k.vehicle.get_distance(ego_id)
        if ego_dis == -1001:
            ego_dis = 0.0
    
        dis_to_goal = total_route_length - ego_dis
        dis_to_goal_norm = np.clip(dis_to_goal / total_route_length, -1.0, 1.0)
    
        ego_speed = self.k.vehicle.get_speed(ego_id)
        if ego_speed is None or ego_speed < 0:
            ego_speed = 0.0
        max_speed = self.k.network.max_speed()
        ego_speed_norm = np.clip(ego_speed / max_speed, -1.0, 1.0)
    
        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)
        ego_cos = np.cos(ego_angle_rad)
        ego_sin = np.sin(ego_angle_rad)
    
        obs_vector = [dis_to_goal_norm, ego_speed_norm, ego_sin, ego_cos]
    
        # --- 2. Neighbor States (Frenet-based) ---
        neighbors_info = []
        all_ids = self.k.vehicle.get_ids()
    
        pos_ret = self.k.vehicle.get_2d_position(ego_id)
        if pos_ret is None or pos_ret == -1001 or pos_ret == (-1001.0, -1001.0):
            return self.last_obs, {}
    
        ego_x, ego_y = pos_ret
    
        for other_id in all_ids:
            if other_id == ego_id:
                continue
    
            other_pos = self.k.vehicle.get_2d_position(other_id)
            if other_pos is None or other_pos == -1001:
                continue
    
            other_x, other_y = other_pos
            distance = np.sqrt((other_x - ego_x)**2 + (other_y - ego_y)**2)
    
            if not (self._is_conflicting(ego_id, other_id) and distance <= self.perception_radius):
                continue
    
            # Neighbor speed
            other_speed = self.k.vehicle.get_speed(other_id)
            if other_speed is None or other_speed < 0:
                other_speed = 0.0
            other_speed_norm = np.clip(other_speed / max_speed, 0.0, 1.0)
    
            # Neighbor heading
            other_heading = self.k.vehicle.get_heading(other_id)
            other_angle_rad = np.radians((-other_heading) + 90)
            other_sin = np.sin(other_angle_rad)
            other_cos = np.cos(other_angle_rad)
    
            # --- Compute conflict point ---
           # vx_ego, vy_ego = ego_cos, ego_sin
           # vx_other, vy_other = other_cos, other_sin
#
           # det = (-ego_cos * other_sin) + (ego_sin * other_cos)
#
           # if abs(det) < 0.05:
           #     # If parallel, ego is following other on same lane
           #     dx = other_x - ego_x
           #     dy = other_y - ego_y
           #     
           #     # Dot product gives the projection (longitudinal distance)
           #     # t1 is how far ego must travel to reach 'other'
           #     t1 = dx * vx_ego + dy * vy_ego
           #     
           #     # In a following scenario, the lead vehicle is already "at" the conflict
           #     # relative to its own path start, so we set its distance to 0.
           #     other_dist_to_cp = 0.0
           #     
           #     # Apply a 5.0m buffer for the lead vehicle's physical length
           #     ego_dist_to_cp = max(0, t1)
           # else:  # Intersecting Case
           #     dx = other_x - ego_x
           #     dy = other_y - ego_y
           #     
           #     t1 = (dx * (-vy_other) - dy * (-vx_other)) / det
           #     t2 = (dx * vy_ego - dy * vx_ego) / det
           # 
           #     # Lane width buffer (1.5m offset from center of 3m lane)
           #     ego_dist_to_cp = max(0, t1)
           #     other_dist_to_cp = max(0, t2)
            
             # Normalise dist_to_cp
            edge = self.k.vehicle.get_edge(other_id)
            
            #rel_speed = ego_speed - other_speed 

            # 2. TTC (Time to Collision) 
            #ttc = ego_dist_to_cp / max(rel_speed, 1e-3) if rel_speed > 0 else np.inf
            #ttc_norm = ttc_norm = 1.0 - np.exp(-ttc / 3.0)  # Normalize to a 10s horizon
            
            ego_line, ego_pos_on_edge = self._get_vehicle_polyline(ego_id)
            other_line, other_pos_on_edge = self._get_vehicle_polyline(other_id)
            
            # Find where the two geometric paths intersect
            intersection = ego_line.intersection(other_line)
            
            if intersection.is_empty:
                # Paths never cross (e.g., parallel lanes, turning away from each other)
                continue

            # Initialize distances
            ego_dist_to_cp = 0.0
            other_dist_to_cp = 0.0
            
            ego_point = Point(ego_x, ego_y)
            other_point = Point(other_x, other_y)
            geom_type = intersection.geom_type

            if geom_type in ['Point', 'MultiPoint']:
                if geom_type == 'MultiPoint':
                    # Find the first point of contact along each vehicle's respective path
                    ego_proj = min([ego_line.project(p) for p in intersection.geoms])
                    other_proj = min([other_line.project(p) for p in intersection.geoms])
                else:
                    # Standard single point
                    ego_proj = ego_line.project(intersection)
                    other_proj = other_line.project(intersection)

                ego_dist_to_cp = max(0.0, ego_proj - ego_pos_on_edge)
                other_dist_to_cp = max(0.0, other_proj - other_pos_on_edge)

                
            # Case B: Paths overlap (Car-Following or Merging)
            elif geom_type in ['LineString', 'MultiLineString', 'GeometryCollection']:
                is_car_following = False
                SAME_PATH_TOLERANCE = 2.0  # meters tolerance to snap to shared path
                
                # 1. Check if Other is physically on Ego's path (Other is in front/behind Ego)
                if ego_line.distance(other_point) < SAME_PATH_TOLERANCE:
                    other_proj = ego_line.project(other_point)
                    if other_proj >= ego_pos_on_edge:
                        ego_dist_to_cp = max(0.0, other_proj - ego_pos_on_edge)
                    else:
                        other_dist_to_cp = max(0.0, ego_pos_on_edge - other_proj)
                    is_car_following = True
                    
                # 2. Check if Ego is physically on Other's path (Ego is in front/behind Other)
                elif other_line.distance(ego_point) < SAME_PATH_TOLERANCE:
                    ego_proj = other_line.project(ego_point)
                    if ego_proj >= other_pos_on_edge:
                        other_dist_to_cp = max(0.0, ego_proj - other_pos_on_edge)
                    else:
                        ego_dist_to_cp = max(0.0, other_pos_on_edge - ego_proj)
                    is_car_following = True
                    
                # 3. Merging (Vehicles are on different unshared branches approaching the overlap)
                if not is_car_following:
                    # The conflict point is the very beginning of the overlapping segment
                    if hasattr(intersection, 'geoms'): 
                        # Handles MultiLineString and GeometryCollection
                        first_geom = intersection.geoms[0]
                        # If the first item in the collection is a Line/Point, grab its first coord
                        overlap_start = Point(first_geom.coords[0]) if hasattr(first_geom, 'coords') else Point(first_geom.geoms[0].coords[0])
                    else:
                        # Handles standard LineString
                        overlap_start = Point(intersection.coords[0])
                        
                    # Both vehicles must travel to the shared merge point
                    ego_dist_to_cp = max(0.0, ego_line.project(overlap_start) - ego_pos_on_edge)
                    other_dist_to_cp = max(0.0, other_line.project(overlap_start) - other_pos_on_edge)

            # 3. Delta ETA (Difference in arrival times at Conflict Point)
            ego_eta = ego_dist_to_cp / max(ego_speed, 0.5)
            other_eta = other_dist_to_cp / max(other_speed, 0.5)
            delta_eta = ego_eta - other_eta
            delta_eta_norm =  np.tanh(delta_eta / 2.0)

            ego_dist_to_cp = np.clip(ego_dist_to_cp / self.perception_radius, 0, 1)
            neighbors_info.append({
                'ego_dist_to_cp':        ego_dist_to_cp,
                'v':        other_speed_norm,
                'd_eta':        delta_eta_norm,
                'sin':     other_sin,
                'cos':     other_cos,
                'edge':     edge,
                'distance': distance,
            })
    
        # Sort by physical distance, take top k
        neighbors_info.sort(key=lambda n: n['distance'])
        neighbors_info = neighbors_info[:self.max_neighbours]
    
        for neighbor in neighbors_info:
            obs_vector.extend([
                neighbor['ego_dist_to_cp'],
                neighbor['v'],
                neighbor['d_eta'],
                neighbor['sin'],
                neighbor['cos'],
            ])
    
        # Pad missing neighbors: [ego_dist_to_cp=1(safe), other_dist_to_cp=1(safe), sin=0, cos=0]
        num_actual = len(neighbors_info)
        if num_actual < self.max_neighbours:
            for _ in range(self.max_neighbours - num_actual):
                obs_vector.extend([1.0, 0.0, 1.0, 0.0, 0.0])
    
        obs_array = np.array(obs_vector, dtype=np.float32)
        assert np.all(np.isfinite(obs_array)), f"Non-finite obs: {obs_array}"
    
        return obs_array, neighbors_info

    def _is_conflicting(self, veh1, veh2):
        """
        Determines if two vehicles have a conflicting path.
        Returns True if:
        1. They are currently on the same edge AND the other vehicle is AHEAD of ego.
        2. Their routes (source to destination) share any common edges (e.g. merging or shared goal).
        Vehicles on the same edge but BEHIND the ego are NOT considered conflicting.
        """
        edge1 = self.k.vehicle.get_edge(veh1)
        edge2 = self.k.vehicle.get_edge(veh2)
    
        # --- Condition A: Currently on the same edge ---
        if edge1 == edge2:
            # Get lane positions (distance from start of edge)
            pos1 = self.k.vehicle.get_position(veh1)
            pos2 = self.k.vehicle.get_position(veh2)
    
            # If either position is invalid (-1001), treat as conflict to be safe
            if pos1 == -1001 or pos2 == -1001:
                return True
    
            # Conflict only if other vehicle is AHEAD (pos2 > pos1)
            # If behind (pos2 < pos1), no conflict
            return pos2 > pos1
        
        # After crossing only same edge vehicles are conflicting
        if edge1.startswith("E#X"):
            return False 
        
        # --- Condition C: Route-based conflict (merging/shared destination) ---
        route1 = self.routes[veh1]
        route2 = self.routes[veh2]
    
        if not route1 or not route2:
            return False
    
        if edge2.startswith("E#X") and edge2 not in route1:
            return False 


        pattern_1 = (route1[0], route1[-1])
        pattern_2 = (route2[0], route2[-1])
    
        conflicting_patterns = self.conflict_map.get(pattern_1, [])
        return pattern_2 in conflicting_patterns

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params['max_accel']
        max_decel = self.env_params.additional_params['max_decel']

        action_val = float(rl_action)
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
    
    def _build_conflict_map(self):
        """
        Statically maps a (Source, Destination) pair to a list of conflicting 
        (Source, Destination) pairs for a full 4-way intersection.
        """
        # Edge IDs
        N_in, N_out = 'E#T-X', 'E#X-T'
        S_in, S_out = 'E#D-X', 'E#X-D'
        E_in, E_out = 'E#R-X',  'E#X-R'
        W_in, W_out = 'E#L-X',  'E#X-L'

        # 1. Define All Flows (Source, Destination)
        # Straight
        NS = (N_in, S_out)
        SN = (S_in, N_out)
        EW = (E_in, W_out)
        WE = (W_in, E_out)
        
        # Left Turns
        NE = (N_in, E_out)
        SW = (S_in, W_out)
        WN = (W_in, N_out)
        ES = (E_in, S_out)
        
        # Right Turns (Added for completeness)
        NW = (N_in, W_out)
        SE = (S_in, E_out)
        EN = (E_in, N_out)
        WS = (W_in, S_out)

        mapping = {}

        # 2. Straight Conflicts
        # A straight flow conflicts with: 
        # - Both crossing straights
        # - The oncoming left turn (crossing its path)
        # - Both crossing left turns
        mapping[NS] = [WE, EW, SW, WN, ES] 
        mapping[SN] = [WE, EW, NE, WN, ES]
        mapping[EW] = [NS, SN, WN, NE, SW]
        mapping[WE] = [NS, NE, ES, SN, SW, SE, WE]

        # 3. Left Turn Conflicts
        # A left turn conflicts with:
        # - The oncoming straight
        # - Both crossing straights
        # - Adjacent left turns (the ones to their immediate left and right)
        # - The oncoming right turn (merging into the same destination edge)
        # Note: Opposing lefts (e.g., NE and SW) usually pass each other safely.
        mapping[NE] = [SN, WE, EW, WN, ES, SE] 
        mapping[SW] = [NS, WE, EW, WN, ES, NW]
        mapping[WN] = [EW, EN, SN, SW, NS, NE, WN]
        mapping[ES] = [WE, NS, SN, NE, SW, EN]

        # 4. Right Turn Conflicts
        # A right turn conflicts with:
        # - Straight cross traffic approaching from the left
        # - Oncoming left turns (merging into the same destination edge)
        mapping[NW] = [EW, SW]
        mapping[SE] = [WE, NE]
        mapping[EN] = [SN, WN]
        mapping[WS] = [NS, ES, WS]

        return mapping

    def additional_command(self):
        """
        Update the sorting of vehicles using the self.sorted_ids variable.
        """
        for veh_id in self.k.vehicle.get_human_ids():
            self.k.vehicle.set_observed(veh_id)

        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    def _get_abs_position(self, veh_id):
        return self.absolute_position.get(veh_id, -1001)
