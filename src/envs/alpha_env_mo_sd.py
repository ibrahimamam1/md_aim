"""
alpha_env_mo_sd.py
──────────────────
Multi-Objective, State-Dependent Environment for Autonomous Intersection Management.
Directly subclasses Env_N (no legacy v01 dependencies).

Key features:
1. Decomposed Vector Rewards:
   - Long-term Efficiency: r_l = w_p * Δp_t - w_t + w_g * I_goal
   - Short-term Safety:     r_s = -λ_gap * penalty_gap - λ_ttc * φ(TTC_t) - R_c * I_collision
   Cleanly passed in info["vector_reward"] = [r_l, r_s] and info["reward_dict"].
2. Conflict & Risk State:
   - Computes minimum Time-to-Collision (TTC), normalized arrival time gap |d_eta|,
     and physical separation distance.
   - Conflict indicator C(s) ∈ {0, 1} and continuous risk score ρ(s) ∈ [0, 1].
3. Telemetry:
   - Comprehensive safety metrics: collision rate, near-collision rate, min TTC,
     min gap, emergency braking count, unsafe interaction count.
   - Efficiency metrics: traversal time, delay/waiting time, average speed, stops count.
   - Comfort metrics: acceleration variance, max deceleration, jerk, jerk variance.
4. Reward-Weighting Ablation Support:
   - r = r_l + λ(s) * r_s for isolating temporal horizon vs reward importance.
"""

import os
import sys
import numpy as np
from gymnasium.spaces import Box
from shapely.geometry import Point

# Ensure local imports resolve
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_env_single import Env_N


class AlphaEnv_MO_SD(Env_N):
    """
    Multi-Objective, State-Dependent Intersection Environment.
    Full implementation containing Frenet/polyline geometry, attention masks,
    vector rewards, conflict diagnostics, and Section 15 telemetry.
    """

    def __init__(
        self,
        env_params,
        sim_params,
        network,
        simulator="traci",
        # Efficiency reward parameters
        progress_weight=10.0,
        time_cost=0.01,
        goal_reward=15.0,
        # Safety reward parameters (calibrated to prevent gap penalties exceeding collision)
        gap_penalty_weight=0.25,
        ttc_penalty_weight=0.5,
        collision_penalty=20.0,
        # Conflict thresholds
        ttc_threshold=3.0,          # seconds
        d_eta_threshold=0.2,        # normalized time gap threshold
        danger_distance=20.0,       # meters
        # Scalarization weights
        weight_l=0.5,
        weight_s=0.5,
        # Reward adaptation ablation
        mode="multi_objective",     # "multi_objective", "ablation_reward_adaptation", "baseline"
        lambda_danger=5.0,
        lambda_normal=1.0,
        # Late-observed conflict scenario override (perception radius)
        perception_radius_override=None,
    ):
        self.prev_pos = dict()
        self.absolute_position = dict()
        self.max_neighbours = 5
        self.perception_radius = float(perception_radius_override) if perception_radius_override is not None else 100.0

        # Ego-centric observation: S_ego = [d_norm, v_norm, sin θ, cos θ] (4 features)
        self.ego_obs_features = 4
        # Per-neighbor: [ego_d_to_cp, other_dist_to_cp, v, other_sin, other_cos] (5 features)
        self.neighbour_obs_features = 5
        self.routes = dict()
        self.last_progress = 0.0

        super().__init__(env_params, sim_params, network, simulator=simulator)

        # Static conflict map
        self.conflict_map = self._build_conflict_map()

        # Action space: normalized acceleration in [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: 4 ego + (5 neighbor_features * 5 max_neighbors) + 5 mask = 34 dims
        total_obs_len = self.ego_obs_features + (self.neighbour_obs_features * self.max_neighbours) + self.max_neighbours
        self.observation_space = Box(low=-1.0, high=1.0, shape=(total_obs_len,), dtype=np.float32)

        self.last_action = 0.0
        self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self.last_neighbors_info = []

        self.progress_weight = float(progress_weight)
        self.time_cost = float(time_cost)
        self.goal_reward = float(goal_reward)

        self.gap_penalty_weight = float(gap_penalty_weight)
        self.ttc_penalty_weight = float(ttc_penalty_weight)
        self.collision_penalty = float(collision_penalty)

        self.ttc_threshold = float(ttc_threshold)
        self.d_eta_threshold = float(d_eta_threshold)
        self.danger_distance = float(danger_distance)

        self.weight_l = float(weight_l)
        self.weight_s = float(weight_s)

        self.mode = mode
        self.lambda_danger = float(lambda_danger)
        self.lambda_normal = float(lambda_normal)

        # Cache for step-level metrics
        self.last_vector_reward = np.zeros(2, dtype=np.float32)
        self.last_conflict_info = {
            "is_conflict": False,
            "min_ttc": float("inf"),
            "min_d_eta": 1.0,
            "min_gap": float("inf"),
            "conflict_risk": 0.0,
            "conflicting_neighbors_count": 0,
        }

        # Initialize telemetry
        self._init_mo_telemetry()

    def _init_mo_telemetry(self):
        """Initializes or resets comprehensive Section 15 telemetry."""
        self.mo_telemetry = {
            # Safety metrics
            "collision": False,
            "near_collision_count": 0,
            "min_ttc": float("inf"),
            "min_safe_gap": 1.0,
            "min_distance_gap": float("inf"),
            "emergency_braking_count": 0,
            "unsafe_interactions_count": 0,
            "conflict_steps_count": 0,
            "total_steps": 0,
            # Efficiency metrics
            "traversal_time": 0.0,
            "waiting_time": 0.0,
            "average_speed": 0.0,
            "stops_count": 0,
            "goal_reached": False,
            "cumulative_progress": 0.0,
            # Comfort metrics
            "accelerations": [],
            "jerks": [],
            "speeds": [],
            "max_deceleration": 0.0,
            "accel_variance": 0.0,
            "mean_abs_jerk": 0.0,
            "jerk_variance": 0.0,
            # Detailed reward component accumulators
            "reward_progress": 0.0,
            "reward_goal": 0.0,
            "reward_time": 0.0,
            "reward_gap": 0.0,
            "reward_collision": 0.0,
            # Reward totals
            "reward_l_total": 0.0,
            "reward_s_total": 0.0,
            "reward_total": 0.0,
        }
        self._was_stopped = False

    def reset(self, *, seed=None, options=None):
        self._init_mo_telemetry()
        self.last_vector_reward = np.zeros(2, dtype=np.float32)
        self.last_conflict_info = {
            "is_conflict": False,
            "min_ttc": float("inf"),
            "min_d_eta": 1.0,
            "min_gap": float("inf"),
            "conflict_risk": 0.0,
            "conflicting_neighbors_count": 0,
        }
        obs, info = super().reset(seed=seed, options=options)
        self.total_route_length = self._compute_route_length(self.agent_id)
        self.prev_progress = 0.0
        self.last_valid_distance = 0.0
        self._last_step_cache = None
        info["vector_reward"] = self.last_vector_reward.copy()
        info["conflict_info"] = dict(self.last_conflict_info)
        return obs, info

    def _compute_route_length(self, agent_id):
        """
        Computes accurate total route length including normal edges and
        connecting junction internal lanes via TraCI lane connections.
        """
        if agent_id is None:
            return 100.0
        try:
            traci = self.k.kernel_api
            route = traci.vehicle.getRoute(agent_id)
            if not route:
                return 100.0
            total_len = 0.0
            for i, edge in enumerate(route):
                lanes = [l for l in traci.lane.getIDList() if l.startswith(edge + '_')]
                edge_len = traci.lane.getLength(lanes[0]) if lanes else self.k.network.edge_length(edge)
                total_len += edge_len
                if i < len(route) - 1:
                    next_edge = route[i + 1]
                    links = traci.lane.getLinks(lanes[0]) if lanes else []
                    for link in links:
                        if link[0].startswith(next_edge + '_'):
                            internal_lane = link[4]
                            if internal_lane and internal_lane in traci.lane.getIDList():
                                total_len += traci.lane.getLength(internal_lane)
                            break
            return max(float(total_len), 1.0)
        except Exception:
            try:
                route = self.k.vehicle.get_route(agent_id)
                return max(sum([self.k.network.edge_length(e) for e in route]), 100.0)
            except Exception:
                return 100.0

    def _update_routes(self):
        """Updates stored routes for active vehicles and removes departed."""
        current_ids = self.k.vehicle.get_ids()
        for veh_id in current_ids:
            if veh_id not in self.routes:
                self.routes[veh_id] = self.k.vehicle.get_route(veh_id)
        active_set = set(current_ids)
        for veh_id in list(self.routes.keys()):
            if veh_id not in active_set:
                del self.routes[veh_id]

    def get_state(self):
        """Constructs the 34-dim attention observation vector."""
        self._update_routes()
        rl_ids = self.k.vehicle.get_rl_ids()
        if self.agent_id not in rl_ids:
            return self.last_obs

        obs, neighbors_info = self._get_local_observation(self.agent_id)
        self.last_obs = obs
        self.last_neighbors_info = neighbors_info
        return obs

    def _get_local_observation(self, ego_id):
        # 1. Ego State
        total_route_length = getattr(self, "total_route_length", None)
        if total_route_length is None or total_route_length <= 0:
            total_route_length = self._compute_route_length(ego_id)
            self.total_route_length = total_route_length

        ego_dis = self.k.vehicle.get_distance(ego_id)
        if ego_dis == -1001 or ego_dis is None:
            ego_dis = getattr(self, "last_valid_distance", 0.0)
        else:
            self.last_valid_distance = ego_dis

        dis_to_goal = max(0.0, total_route_length - ego_dis)
        dis_to_goal_norm = np.clip(dis_to_goal / total_route_length, -1.0, 1.0)

        ego_speed = max(self.k.vehicle.get_speed(ego_id) or 0.0, 0.0)
        max_speed = self.k.network.max_speed()
        ego_speed_norm = np.clip(ego_speed / max_speed, -1.0, 1.0)

        ego_heading = self.k.vehicle.get_heading(ego_id)
        ego_angle_rad = np.radians((-ego_heading) + 90)
        ego_cos = np.cos(ego_angle_rad)
        ego_sin = np.sin(ego_angle_rad)

        obs_vector = [dis_to_goal_norm, ego_speed_norm, ego_sin, ego_cos]

        # 2. Neighbor States (Frenet-based)
        neighbors_info = []
        all_ids = self.k.vehicle.get_ids()

        pos_ret = self.k.vehicle.get_2d_position(ego_id)
        if pos_ret is None or pos_ret == -1001 or pos_ret == (-1001.0, -1001.0):
            return self.last_obs, []

        ego_x, ego_y = pos_ret

        for other_id in all_ids:
            if other_id == ego_id:
                continue

            other_pos = self.k.vehicle.get_2d_position(other_id)
            if other_pos is None or other_pos == -1001:
                continue

            other_x, other_y = other_pos
            distance = np.sqrt((other_x - ego_x) ** 2 + (other_y - ego_y) ** 2)

            if not (self._is_conflicting(ego_id, other_id) and distance <= self.perception_radius):
                continue

            other_speed = max(self.k.vehicle.get_speed(other_id) or 0.0, 0.0)
            other_speed_norm = np.clip(other_speed / max_speed, 0.0, 1.0)

            other_heading = self.k.vehicle.get_heading(other_id)
            other_angle_rad = np.radians((-other_heading) + 90)
            other_sin = np.sin(other_angle_rad)
            other_cos = np.cos(other_angle_rad)

            edge = self.k.vehicle.get_edge(other_id)

            ego_line, ego_pos_on_edge = self._get_vehicle_polyline(ego_id)
            other_line, other_pos_on_edge = self._get_vehicle_polyline(other_id)

            intersection = ego_line.intersection(other_line)
            if intersection.is_empty:
                continue

            ego_dist_to_cp = 0.0
            other_dist_to_cp = 0.0
            ego_point = Point(ego_x, ego_y)
            other_point = Point(other_x, other_y)
            geom_type = intersection.geom_type

            if geom_type in ["Point", "MultiPoint"]:
                if geom_type == "MultiPoint":
                    ego_proj = min([ego_line.project(p) for p in intersection.geoms])
                    other_proj = min([other_line.project(p) for p in intersection.geoms])
                else:
                    ego_proj = ego_line.project(intersection)
                    other_proj = other_line.project(intersection)

                ego_dist_to_cp = max(0.0, ego_proj - ego_pos_on_edge)
                other_dist_to_cp = max(0.0, other_proj - other_pos_on_edge)

            elif geom_type in ["LineString", "MultiLineString", "GeometryCollection"]:
                is_car_following = False
                SAME_PATH_TOLERANCE = 2.0

                if ego_line.distance(other_point) < SAME_PATH_TOLERANCE:
                    other_proj = ego_line.project(other_point)
                    if other_proj >= ego_pos_on_edge:
                        leader_len = getattr(self.k.vehicle, "get_length", lambda _id: 5.0)(other_id)
                        ego_dist_to_cp = max(0.0, other_proj - ego_pos_on_edge - leader_len)
                        other_dist_to_cp = 0.0
                    else:
                        ego_len = getattr(self.k.vehicle, "get_length", lambda _id: 5.0)(ego_id)
                        other_dist_to_cp = max(0.0, ego_pos_on_edge - other_proj - ego_len)
                        ego_dist_to_cp = 0.0
                    is_car_following = True

                elif other_line.distance(ego_point) < SAME_PATH_TOLERANCE:
                    ego_proj = other_line.project(ego_point)
                    if ego_proj >= other_pos_on_edge:
                        ego_len = getattr(self.k.vehicle, "get_length", lambda _id: 5.0)(ego_id)
                        other_dist_to_cp = max(0.0, ego_proj - other_pos_on_edge - ego_len)
                        ego_dist_to_cp = 0.0
                    else:
                        leader_len = getattr(self.k.vehicle, "get_length", lambda _id: 5.0)(other_id)
                        ego_dist_to_cp = max(0.0, other_pos_on_edge - ego_proj - leader_len)
                        other_dist_to_cp = 0.0
                    is_car_following = True

                if not is_car_following:
                    if hasattr(intersection, "geoms"):
                        candidates = []
                        for geom in intersection.geoms:
                            if hasattr(geom, "coords") and len(geom.coords) > 0:
                                candidates.append(Point(geom.coords[0]))
                            elif hasattr(geom, "geoms"):
                                for g in geom.geoms:
                                    if hasattr(g, "coords") and len(g.coords) > 0:
                                        candidates.append(Point(g.coords[0]))
                        if candidates:
                            overlap_start = min(candidates, key=lambda p: ego_line.project(p))
                        else:
                            overlap_start = Point(intersection.coords[0])
                    else:
                        overlap_start = Point(intersection.coords[0])

                    ego_dist_to_cp = max(0.0, ego_line.project(overlap_start) - ego_pos_on_edge)
                    other_dist_to_cp = max(0.0, other_line.project(overlap_start) - other_pos_on_edge)

            ego_dist_to_cp_norm = np.clip(ego_dist_to_cp / self.perception_radius, 0.0, 1.0)
            other_dist_to_cp_norm = np.clip(other_dist_to_cp / self.perception_radius, 0.0, 1.0)

            ego_eta = ego_dist_to_cp / max(ego_speed, 0.5)
            other_eta = other_dist_to_cp / max(other_speed, 0.5)
            delta_eta = ego_eta - other_eta
            delta_eta_norm = np.tanh(delta_eta / 5.0)

            neighbors_info.append({
                "ego_dist_to_cp_norm": ego_dist_to_cp_norm,
                "other_dist_to_cp_norm": other_dist_to_cp_norm,
                "other_speed": other_speed_norm,
                "other_sin": other_sin,
                "other_cos": other_cos,
                "d_eta": delta_eta_norm,
                "edge": edge,
                "distance": distance,
            })

        # Sort by distance and take top k
        neighbors_info.sort(key=lambda n: n["distance"])
        neighbors_info = neighbors_info[:self.max_neighbours]

        for neighbor in neighbors_info:
            obs_vector.extend([
                neighbor["ego_dist_to_cp_norm"],
                neighbor["other_dist_to_cp_norm"],
                neighbor["other_speed"],
                neighbor["other_sin"],
                neighbor["other_cos"],
            ])

        # Pad missing neighbors
        num_actual = len(neighbors_info)
        if num_actual < self.max_neighbours:
            for _ in range(self.max_neighbours - num_actual):
                obs_vector.extend([1.0, 0.0, 1.0, 0.0, 0.0])

        # Attention mask (1.0 for real neighbor, 0.0 for padding)
        neighbor_mask = [1.0] * num_actual + [0.0] * (self.max_neighbours - num_actual)
        obs_vector.extend(neighbor_mask)

        obs_array = np.array(obs_vector, dtype=np.float32)
        return obs_array, neighbors_info

    def _is_conflicting(self, veh1, veh2):
        edge1 = self.k.vehicle.get_edge(veh1)
        edge2 = self.k.vehicle.get_edge(veh2)

        if edge1 == edge2:
            pos1 = self.k.vehicle.get_position(veh1)
            pos2 = self.k.vehicle.get_position(veh2)
            if pos1 == -1001 or pos2 == -1001:
                return True
            return pos2 > pos1

        if edge1.startswith("E#X"):
            return False

        route1 = self.routes.get(veh1, [])
        route2 = self.routes.get(veh2, [])
        if not route1 or not route2:
            return False

        if edge2.startswith("E#X") and edge2 not in route1:
            return False

        pattern_1 = (route1[0], route1[-1])
        pattern_2 = (route2[0], route2[-1])
        conflicting_patterns = self.conflict_map.get(pattern_1, [])
        return pattern_2 in conflicting_patterns

    def _build_conflict_map(self):
        N_in, N_out = "E#T-X", "E#X-T"
        S_in, S_out = "E#D-X", "E#X-D"
        E_in, E_out = "E#R-X", "E#X-R"
        W_in, W_out = "E#L-X", "E#X-L"

        NS = (N_in, S_out)
        SN = (S_in, N_out)
        EW = (E_in, W_out)
        WE = (W_in, E_out)

        NE = (N_in, E_out)
        SW = (S_in, W_out)
        WN = (W_in, N_out)
        ES = (E_in, S_out)

        NW = (N_in, W_out)
        SE = (S_in, E_out)
        EN = (E_in, N_out)
        WS = (W_in, S_out)

        mapping = {}
        mapping[NS] = [WE, EW, SW, WN, ES, WS, NS]
        mapping[SN] = [WE, EW, NE, WN, ES, EN, SN]
        mapping[EW] = [NS, SN, WN, NE, SW, NW, EW]
        mapping[WE] = [NS, NE, ES, SN, SW, SE, WE]

        mapping[NE] = [SN, WE, EW, WN, ES, SE, NE]
        mapping[SW] = [NS, WE, EW, WN, ES, NW, SW]
        mapping[WN] = [EW, EN, SN, SW, NS, NE, WN]
        mapping[ES] = [WE, NS, SN, NE, SW, EN, ES]

        mapping[NW] = [EW, SW, NW]
        mapping[SE] = [WE, NE, SE]
        mapping[EN] = [SN, WN, EN]
        mapping[WS] = [NS, ES, WS]
        return mapping

    def _apply_rl_actions(self, rl_action):
        max_accel = self.env_params.additional_params["max_accel"]
        max_decel = self.env_params.additional_params["max_decel"]

        try:
            action_val = float(rl_action[0]) if isinstance(rl_action, (list, np.ndarray)) else float(rl_action)
        except (TypeError, ValueError):
            action_val = 0.0

        if np.isnan(action_val) or np.isinf(action_val):
            action_val = 0.0

        if action_val >= 0:
            real_action = action_val * max_accel
        else:
            real_action = action_val * max_decel

        if self.agent_id in self.k.vehicle.get_ids():
            self.k.vehicle.apply_acceleration([self.agent_id], [real_action])

    def additional_command(self):
        for veh_id in self.k.vehicle.get_human_ids():
            self.k.vehicle.set_observed(veh_id)
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)
            if this_pos == -1001:
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = (
                    (self.absolute_position.get(veh_id, this_pos) + change) % self.k.network.length()
                )
                self.prev_pos[veh_id] = this_pos

    def _get_abs_position(self, veh_id):
        return self.absolute_position.get(veh_id, -1001)

    def compute_conflict_features(self, neighbors_info):
        if not neighbors_info:
            return {
                "is_conflict": False,
                "min_ttc": float("inf"),
                "min_d_eta": 1.0,
                "min_gap": float("inf"),
                "conflict_risk": 0.0,
                "conflicting_neighbors_count": 0,
            }

        min_d_eta = 1.0
        min_ttc = float("inf")
        min_gap = float("inf")
        conflicting_count = 0

        ego_speed = 0.0
        if self.agent_id in self.k.vehicle.get_ids():
            ego_speed = max(self.k.vehicle.get_speed(self.agent_id) or 0.0, 0.0)

        for n in neighbors_info:
            d_eta = abs(float(n.get("d_eta", 1.0)))
            dist = float(n.get("distance", self.perception_radius))
            ego_d_norm = float(n.get("ego_dist_to_cp_norm", 1.0))
            other_d_norm = float(n.get("other_dist_to_cp_norm", 1.0))
            other_speed_norm = float(n.get("other_speed", 0.0))

            ego_dist_cp = ego_d_norm * self.perception_radius
            other_dist_cp = other_d_norm * self.perception_radius
            other_speed = other_speed_norm * self.k.network.max_speed()

            ego_time_to_cp = ego_dist_cp / max(ego_speed, 0.5)
            other_time_to_cp = other_dist_cp / max(other_speed, 0.5)

            if abs(ego_time_to_cp - other_time_to_cp) < 2.0:
                ttc_est = min(ego_time_to_cp, other_time_to_cp)
            else:
                ttc_est = dist / max(ego_speed - other_speed, 0.5) if ego_speed > other_speed else float("inf")

            if ttc_est > 0:
                min_ttc = min(min_ttc, ttc_est)
            min_d_eta = min(min_d_eta, d_eta)
            min_gap = min(min_gap, dist)

            if ttc_est < self.ttc_threshold or dist < self.danger_distance:
                conflicting_count += 1

        is_conflict = (
            min_ttc < self.ttc_threshold
            or min_gap < self.danger_distance
        )

        ttc_risk = float(max(0.0, 1.0 - (min_ttc / self.ttc_threshold))) if min_ttc < self.ttc_threshold else 0.0
        gap_risk = float(max(0.0, 1.0 - (min_gap / self.danger_distance))) if min_gap < self.danger_distance else 0.0

        conflict_risk = float(np.clip(max(ttc_risk, gap_risk), 0.0, 1.0))

        return {
            "is_conflict": bool(is_conflict),
            "min_ttc": float(min_ttc),
            "min_d_eta": float(min_d_eta),
            "min_gap": float(min_gap),
            "conflict_risk": float(conflict_risk),
            "conflicting_neighbors_count": int(conflicting_count),
        }

    def compute_decomposed_reward(self, agent_id, fail, goal_reached, neighbors_info, current_action=None, conflict_info=None):
        r_prog = 0.0
        r_goal = 0.0
        r_time = 0.0
        r_gap = 0.0
        r_col = 0.0
        progress_delta = 0.0

        if fail:
            # Collision failure: no progress reward, catastrophic crash penalty
            r_col = -float(self.collision_penalty)
            r_l = 0.0
            r_s = r_col
            return r_l, r_s, progress_delta, r_prog, r_goal, r_time, r_gap, r_col

        if goal_reached:
            # Terminal goal reached: vehicle traversed destination edge and left SUMO
            prev_p = getattr(self, "prev_progress", 0.0)
            progress_delta = max(0.0, 1.0 - prev_p)
            self.prev_progress = 1.0
            r_prog = float(self.progress_weight * progress_delta)
            r_goal = float(self.goal_reward)
            r_time = -float(self.time_cost)
            r_l = float(r_prog + r_goal + r_time)
            r_s = 0.0
            return r_l, r_s, progress_delta, r_prog, r_goal, r_time, r_gap, r_col

        if agent_id not in self.k.vehicle.get_ids():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Normal active driving step
        ego_dis = self.k.vehicle.get_distance(agent_id)
        if ego_dis == -1001 or ego_dis is None:
            ego_dis = getattr(self, "last_valid_distance", 0.0)
        else:
            self.last_valid_distance = ego_dis

        total_len = max(getattr(self, "total_route_length", 100.0), 1.0)
        progress_norm = float(np.clip(ego_dis / total_len, 0.0, 1.0))

        prev_p = getattr(self, "prev_progress", 0.0)
        progress_delta = max(0.0, progress_norm - prev_p)
        self.prev_progress = progress_norm

        r_prog = float(self.progress_weight * progress_delta)
        r_time = -float(self.time_cost)
        r_l = float(r_prog + r_time)

        # Proximity and TTC safety penalties (strictly zero when vehicles are outside danger threshold)
        safety_gap_penalty = 0.0
        for n in neighbors_info:
            dist = float(n.get("distance", self.perception_radius))
            if dist < self.danger_distance:
                safety_gap_penalty += -float(1.0 - (dist / self.danger_distance))

        r_gap = float(self.gap_penalty_weight * safety_gap_penalty)

        ttc_penalty = 0.0
        if conflict_info is not None:
            min_ttc = float(conflict_info.get("min_ttc", float("inf")))
            if min_ttc < self.ttc_threshold:
                ttc_penalty = -float(self.ttc_penalty_weight * (1.0 - (min_ttc / self.ttc_threshold)))

        r_s = float(r_gap + ttc_penalty)
        return r_l, r_s, progress_delta, r_prog, r_goal, r_time, r_gap, r_col

    def compute_reward(self, agent_id, fail, goal_reached, current_action=None):
        # Called once per step by super().step() in base_env_single
        neighbors_info = getattr(self, "last_neighbors_info", []) or []
        conflict_info = self.compute_conflict_features(neighbors_info)
        self.last_conflict_info = conflict_info

        r_l, r_s, progress_delta, r_prog, r_goal, r_time, r_gap, r_col = self.compute_decomposed_reward(
            agent_id, fail=fail, goal_reached=goal_reached,
            neighbors_info=neighbors_info, current_action=current_action,
            conflict_info=conflict_info
        )

        if self.mode == "ablation_reward_adaptation":
            is_conflict = conflict_info.get("is_conflict", False)
            lam = self.lambda_danger if is_conflict else self.lambda_normal
            scalar_reward = float(r_l + lam * r_s)
        elif self.mode == "baseline":
            scalar_reward = float(r_l + r_s)
        else:
            scalar_reward = float(self.weight_l * r_l + self.weight_s * r_s)

        self._last_step_cache = {
            "r_l": r_l,
            "r_s": r_s,
            "progress_delta": progress_delta,
            "r_progress": r_prog,
            "r_goal": r_goal,
            "r_time": r_time,
            "r_gap": r_gap,
            "r_collision": r_col,
            "scalar_reward": scalar_reward,
            "conflict_info": conflict_info,
            "crashed": bool(fail),
            "goal_reached": bool(goal_reached),
        }
        return scalar_reward

    def step(self, action):
        obs, raw_scalar_reward, terminated, truncated, infos = super().step(action)

        cache = getattr(self, "_last_step_cache", None)
        if cache is not None:
            r_l = cache["r_l"]
            r_s = cache["r_s"]
            progress_delta = cache["progress_delta"]
            r_prog = cache["r_progress"]
            r_goal = cache["r_goal"]
            r_time = cache["r_time"]
            r_gap = cache["r_gap"]
            r_col = cache["r_collision"]
            scalar_reward = cache["scalar_reward"]
            conflict_info = cache["conflict_info"]
            crashed = cache["crashed"]
            goal_reached = cache["goal_reached"]
        else:
            neighbors_info = getattr(self, "last_neighbors_info", []) or []
            conflict_info = self.compute_conflict_features(neighbors_info)
            self.last_conflict_info = conflict_info
            crashed = bool(self.telemetry.get("agent_collision", False))
            goal_reached = bool(self.telemetry.get("agent_success", False))
            r_l, r_s, progress_delta, r_prog, r_goal, r_time, r_gap, r_col = self.compute_decomposed_reward(
                self.agent_id, fail=crashed, goal_reached=goal_reached,
                neighbors_info=neighbors_info, current_action=action,
                conflict_info=conflict_info
            )
            scalar_reward = float(self.weight_l * r_l + self.weight_s * r_s)

        vector_reward = np.array([r_l, r_s], dtype=np.float32)
        self.last_vector_reward = vector_reward

        self._update_mo_telemetry(
            action, r_l, r_s, scalar_reward, conflict_info, crashed, goal_reached,
            progress_delta=progress_delta, r_prog=r_prog, r_goal=r_goal, r_time=r_time, r_gap=r_gap, r_col=r_col
        )

        infos["vector_reward"] = vector_reward
        infos["reward_dict"] = {
            "progress_reward": float(r_prog),
            "goal_reward": float(r_goal),
            "time_penalty": float(r_time),
            "gap_penalty": float(r_gap),
            "collision_penalty": float(r_col),
            "total_long_term_reward": float(r_l),
            "total_safety_reward": float(r_s),
            "r_l": float(r_l),
            "r_s": float(r_s),
            "scalar_reward": float(scalar_reward),
            "progress_delta": float(progress_delta),
        }
        infos["conflict_info"] = conflict_info

        if terminated or truncated:
            infos["mo_telemetry"] = self._compile_final_telemetry(crashed, goal_reached)

        return obs, scalar_reward, terminated, truncated, infos

    def _update_mo_telemetry(self, action, r_l, r_s, scalar_reward, conflict_info, crashed, goal_reached,
                             progress_delta=0.0, r_prog=0.0, r_goal=0.0, r_time=0.0, r_gap=0.0, r_col=0.0):
        self.mo_telemetry["total_steps"] += 1
        self.mo_telemetry["reward_progress"] += float(r_prog)
        self.mo_telemetry["reward_goal"] += float(r_goal)
        self.mo_telemetry["reward_time"] += float(r_time)
        self.mo_telemetry["reward_gap"] += float(r_gap)
        self.mo_telemetry["reward_collision"] += float(r_col)
        self.mo_telemetry["reward_l_total"] += float(r_l)
        self.mo_telemetry["reward_s_total"] += float(r_s)
        self.mo_telemetry["reward_total"] += float(scalar_reward)
        self.mo_telemetry["cumulative_progress"] += float(progress_delta)

        if conflict_info["is_conflict"]:
            self.mo_telemetry["conflict_steps_count"] += 1
            self.mo_telemetry["unsafe_interactions_count"] += conflict_info["conflicting_neighbors_count"]

        if conflict_info["min_d_eta"] < self.d_eta_threshold or conflict_info["min_ttc"] < 1.5:
            self.mo_telemetry["near_collision_count"] += 1

        self.mo_telemetry["min_ttc"] = min(self.mo_telemetry["min_ttc"], conflict_info["min_ttc"])
        self.mo_telemetry["min_safe_gap"] = min(self.mo_telemetry["min_safe_gap"], conflict_info["min_d_eta"])
        self.mo_telemetry["min_distance_gap"] = min(self.mo_telemetry["min_distance_gap"], conflict_info["min_gap"])

        if self.agent_id in self.k.vehicle.get_ids():
            speed = float(self.k.vehicle.get_speed(self.agent_id) or 0.0)
            accel = float(self.k.vehicle.get_accel(self.agent_id) or 0.0)

            self.mo_telemetry["speeds"].append(speed)
            self.mo_telemetry["accelerations"].append(accel)

            if accel < -3.0:
                self.mo_telemetry["emergency_braking_count"] += 1
            if accel < self.mo_telemetry["max_deceleration"]:
                self.mo_telemetry["max_deceleration"] = accel

            if speed < 0.2:
                self.mo_telemetry["waiting_time"] += self.sim_step
                if not self._was_stopped:
                    self.mo_telemetry["stops_count"] += 1
                    self._was_stopped = True
            else:
                self._was_stopped = False

            if len(self.mo_telemetry["accelerations"]) > 1:
                prev_accel = self.mo_telemetry["accelerations"][-2]
                jerk = (accel - prev_accel) / max(self.sim_step, 1e-4)
                self.mo_telemetry["jerks"].append(float(jerk))

        if crashed:
            self.mo_telemetry["collision"] = True
        if goal_reached:
            self.mo_telemetry["goal_reached"] = True

    def _compile_final_telemetry(self, crashed, goal_reached):
        accels = np.array(self.mo_telemetry["accelerations"], dtype=np.float32) if self.mo_telemetry["accelerations"] else np.zeros(1)
        jerks = np.array(self.mo_telemetry["jerks"], dtype=np.float32) if self.mo_telemetry["jerks"] else np.zeros(1)
        speeds = np.array(self.mo_telemetry["speeds"], dtype=np.float32) if self.mo_telemetry["speeds"] else np.zeros(1)

        traversal_time = self.telemetry.get("agent_finish_time") or self.time_counter
        spawn_time = self.telemetry.get("agent_spawn_time") or 0.0
        duration = max(0.0, traversal_time - spawn_time)

        min_ttc = self.mo_telemetry["min_ttc"]
        if np.isinf(min_ttc):
            min_ttc = 99.0

        return {
            "collision": 1 if crashed else 0,
            "success": 1 if goal_reached else 0,
            "cumulative_progress": float(self.mo_telemetry["cumulative_progress"]),
            # Explicit Section reward diagnostics
            "progress_reward": float(self.mo_telemetry["reward_progress"]),
            "goal_reward": float(self.mo_telemetry["reward_goal"]),
            "time_penalty": float(self.mo_telemetry["reward_time"]),
            "gap_penalty": float(self.mo_telemetry["reward_gap"]),
            "collision_penalty": float(self.mo_telemetry["reward_collision"]),
            "total_long_term_reward": float(self.mo_telemetry["reward_l_total"]),
            "total_safety_reward": float(self.mo_telemetry["reward_s_total"]),
            "total_reward": float(self.mo_telemetry["reward_total"]),
            # Aliases
            "reward_efficiency_total": float(self.mo_telemetry["reward_l_total"]),
            "reward_safety_total": float(self.mo_telemetry["reward_s_total"]),
            # Physical safety & efficiency metrics
            "near_collision_count": int(self.mo_telemetry["near_collision_count"]),
            "min_ttc": float(min_ttc),
            "min_safe_gap": float(self.mo_telemetry["min_safe_gap"]),
            "min_distance_gap": float(self.mo_telemetry["min_distance_gap"] if not np.isinf(self.mo_telemetry["min_distance_gap"]) else 99.0),
            "emergency_braking_count": int(self.mo_telemetry["emergency_braking_count"]),
            "unsafe_interactions_count": int(self.mo_telemetry["unsafe_interactions_count"]),
            "conflict_fraction": float(self.mo_telemetry["conflict_steps_count"] / max(1, self.mo_telemetry["total_steps"])),
            "traversal_time": float(duration),
            "waiting_time": float(self.mo_telemetry["waiting_time"]),
            "average_speed": float(np.mean(speeds)),
            "stops_count": int(self.mo_telemetry["stops_count"]),
            "total_distance": float(self.telemetry.get("agent_total_distance", 0.0)),
            "mean_acceleration": float(np.mean(accels)),
            "max_deceleration": float(self.mo_telemetry["max_deceleration"]),
            "accel_variance": float(np.var(accels)),
            "mean_abs_jerk": float(np.mean(np.abs(jerks))),
            "jerk_variance": float(np.var(jerks)),
        }
