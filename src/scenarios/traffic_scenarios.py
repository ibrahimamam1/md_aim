"""
traffic_scenarios.py
────────────────────
Suite of 5 evaluation traffic scenarios:

  S1 — All Low:           All non-West edges (N, S, E) at low inflow (150 veh/h).
  S2 — Mixed Medium-Low:  Two random non-West edges at medium (275 veh/h) and the other at low (150 veh/h).
  S3 — All Medium:        All non-West edges (N, S, E) at medium inflow (275 veh/h).
  S4 — Mixed Medium-High: Two random non-West edges at medium (275 veh/h) and the other at high (400 veh/h).
  S5 — All High:          All non-West edges (N, S, E) at high inflow (400 veh/h).

Note: The West edge (E#L-X) where the ego vehicle spawns has no background (NonRL)
inflow by design; only the RL agent spawns there.
"""

import os
import random
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from networks.asymetric_random import AsymmetricRandomNetwork
from networks.uniform_random import UniformRandomNetwork
from networks.deterministic_scenario import DeterministicSouthNorthNetwork

from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController

LOW_RATE = 150
MEDIUM_RATE = 275
HIGH_RATE = 400
NON_WEST_EDGES = ("N", "S", "E")


def build_inflows(traffic_rate: Dict[str, float], rl_prob: float = 0.8, warmup_steps: int = 50) -> InFlows:
    """Helper to build InFlows from directional traffic rates (veh/hour).

    Note: the West edge (E#L-X) is the RL agent's spawn edge, so no
    background (NonRL) inflow is created there, regardless of traffic_rate["W"].
    """
    inflow = InFlows()
    inflow.add(veh_type="NonRL", edge="E#T-X", probability=traffic_rate["N"] / 3600.0,
               depart_lane=0, depart_speed=0, begin=1, color="green")
    inflow.add(veh_type="NonRL", edge="E#R-X", probability=traffic_rate["E"] / 3600.0,
               depart_lane=0, depart_speed=0, begin=1, color="green")
    inflow.add(veh_type="NonRL", edge="E#D-X", probability=traffic_rate["S"] / 3600.0,
               depart_lane=0, depart_speed=0, begin=1, color="green")
    # No background traffic on the ego's spawn edge (West / E#L-X): no
    # inflow is created there at all.

    # RL agent spawns from West edge
    inflow.add(veh_type="RL", edge="E#L-X", probability=rl_prob,
               depart_lane=0, depart_speed=0, begin=warmup_steps, color="blue")
    return inflow


def get_scenario_definition(scenario_id: str, root_dir: str) -> Dict[str, Any]:
    """
    Returns the complete parameter configuration for a specified scenario S1..S5.
    """
    scen = scenario_id.upper()
    net_dir = os.path.join(root_dir, "networks")

    # Base physics parameters
    min_gap = 2.5
    max_accel = 2.6
    max_decel = 4.5
    max_speed = 55.0
    horizon = 180
    sim_step = 0.25
    warmup_steps = 50

    # Default network file
    net_file = os.path.join(net_dir, "100m_right_before_left.net.xml")
    net_class = AsymmetricRandomNetwork
    perception_radius_override = None

    # Normal IDM parameters for background traffic
    non_rl_min_gap = 2.5
    non_rl_tau = 0.8
    non_rl_accel = 2.6
    non_rl_decel = 4.5
    non_rl_speed_factor = 1.0
    non_rl_speed_dev = 0.1
    non_rl_impatience = 0.0

    if scen == "S1":
        # 1 - All low
        traffic_rates = {"N": LOW_RATE, "S": LOW_RATE, "W": 0, "E": LOW_RATE}
        desc = f"S1 — All Low ({LOW_RATE} veh/h on N, S, E)"

    elif scen == "S2":
        # 2 - Mixed Medium Low: Two random edges are medium and the other is low
        low_edge = random.choice(NON_WEST_EDGES)
        traffic_rates = {
            edge: (LOW_RATE if edge == low_edge else MEDIUM_RATE)
            for edge in NON_WEST_EDGES
        }
        traffic_rates["W"] = 0
        desc = f"S2 — Mixed Medium-Low (two random edges {MEDIUM_RATE} veh/h, one {LOW_RATE} veh/h)"

    elif scen == "S3":
        # 3 - All medium
        traffic_rates = {"N": MEDIUM_RATE, "S": MEDIUM_RATE, "W": 0, "E": MEDIUM_RATE}
        desc = f"S3 — All Medium ({MEDIUM_RATE} veh/h on N, S, E)"

    elif scen == "S4":
        # 4 - Mixed Medium High: Two random edges are medium and the other is high
        high_edge = random.choice(NON_WEST_EDGES)
        traffic_rates = {
            edge: (HIGH_RATE if edge == high_edge else MEDIUM_RATE)
            for edge in NON_WEST_EDGES
        }
        traffic_rates["W"] = 0
        desc = f"S4 — Mixed Medium-High (two random edges {MEDIUM_RATE} veh/h, one {HIGH_RATE} veh/h)"

    elif scen == "S5":
        # 5 - All High
        traffic_rates = {"N": HIGH_RATE, "S": HIGH_RATE, "W": 0, "E": HIGH_RATE}
        desc = f"S5 — All High ({HIGH_RATE} veh/h on N, S, E)"

    else:
        raise ValueError(f"Unknown scenario ID: {scenario_id}. Choose S1..S5.")

    inflows = build_inflows(traffic_rates, rl_prob=0.8, warmup_steps=warmup_steps)

    # Vehicles setup
    vehicles = VehicleParams()
    rl_cfp = SumoCarFollowingParams(
        speed_mode=0,
        accel=max_accel,
        decel=max_decel,
        sigma=0.0,
        tau=0.8,
        min_gap=min_gap,
        max_speed=max_speed,
        speed_factor=1.0,
        speed_dev=0.1,
        impatience=0.0,
        car_follow_model="IDM",
    )
    non_rl_cfp = SumoCarFollowingParams(
        speed_mode=31,  # respects basic SUMO collision avoidance
        accel=non_rl_accel,
        decel=non_rl_decel,
        sigma=0.0,
        tau=non_rl_tau,
        min_gap=non_rl_min_gap,
        max_speed=max_speed,
        speed_factor=non_rl_speed_factor,
        speed_dev=non_rl_speed_dev,
        impatience=non_rl_impatience,
        car_follow_model="IDM",
    )
    vehicles.add(
        veh_id="RL",
        acceleration_controller=(RLController, {}),
        initial_speed=0,
        num_vehicles=0,
        car_following_params=rl_cfp,
        lane_change_params=None,
        color="blue",
    )
    vehicles.add(
        veh_id="NonRL",
        acceleration_controller=(IDMController, {}),
        initial_speed=0,
        num_vehicles=0,
        car_following_params=non_rl_cfp,
        lane_change_params=None,
        color="red",
    )

    net_params = NetParams(osm_path=None, template=net_file, inflows=inflows)
    initial_config = InitialConfig(
        shuffle=False,
        spacing="uniform",
        min_gap=12,
        perturbation=5.0,
        x0=5,
        bunching=0,
        lanes_distribution=float("inf"),
        edges_distribution=["E#D-X", "E#L-X", "E#R-X", "E#T-X"],
    )
    env_params = EnvParams(
        additional_params={
            "max_accel": max_accel,
            "max_decel": max_decel,
            "target_velocity": max_speed,
            "sort_vehicles": False,
        },
        horizon=horizon,
        warmup_steps=warmup_steps,
        sims_per_step=1,
        evaluate=False,
        clip_actions=True,
    )
    sim_params = SumoParams(
        port=None,
        sim_step=sim_step,
        lateral_resolution=None,
        no_step_log=True,
        render=False,
        save_render=False,
        sight_radius=25,
        show_radius=False,
        pxpm=2,
        force_color_update=False,
        overtake_right=False,
        seed=42,
        restart_instance=True,
        print_warnings=False,
        teleport_time=0,
        num_clients=1,
        color_by_speed=False,
        use_ballistic=False,
    )

    return {
        "scenario_id": scen,
        "description": desc,
        "traffic_rates": traffic_rates,
        "net_file": net_file,
        "net_class": net_class,
        "vehicles": vehicles,
        "net_params": net_params,
        "initial_config": initial_config,
        "env_params": env_params,
        "sim_params": sim_params,
        "perception_radius_override": perception_radius_override,
    }


def create_scenario_env(
    scenario_id: str,
    root_dir: str,
    render: bool = False,
    mode: str = "multi_objective",
    weight_l: float = 0.5,
    weight_s: float = 0.5,
    seed: Optional[int] = None,
    **env_kwargs,
):
    """
    Factory function instantiating an AlphaEnv_MO_SD configured for the specified scenario.
    """
    from src.envs.alpha_env_mo_sd import AlphaEnv_MO_SD

    scen_def = get_scenario_definition(scenario_id, root_dir)
    sim_params = deepcopy(scen_def["sim_params"])
    sim_params.render = render
    if seed is not None:
        sim_params.seed = int(seed)

    net = scen_def["net_class"](
        name=f"Scenario_{scenario_id}",
        vehicles=deepcopy(scen_def["vehicles"]),
        net_params=scen_def["net_params"],
        initial_config=scen_def["initial_config"],
        traffic_lights=TrafficLightParams(),
    )

    env = AlphaEnv_MO_SD(
        env_params=scen_def["env_params"],
        sim_params=sim_params,
        network=net,
        simulator="traci",
        mode=mode,
        weight_l=weight_l,
        weight_s=weight_s,
        perception_radius_override=scen_def["perception_radius_override"],
        seed=seed,
        **env_kwargs,
    )
    return env

