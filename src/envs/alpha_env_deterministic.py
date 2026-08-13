"""
alpha_env_deterministic.py
──────────────────────────
Deterministic environment factory for velocity-profile analysis.

Design
──────
All deterministic logic lives in `DeterministicMixin`. The mixin is combined
with the *correct* base environment class for each version tag via the factory
function `make_deterministic_env(version)`.

This ensures that:
  - The observation space matches the checkpoint exactly
    (heuristic = 29 dims, attention = 34 dims)
  - The action space matches the checkpoint exactly
    (continuous Box vs. Discrete-3/5/10)

Key properties of every resulting environment
─────────────────────────────────────────────
1. Ego vehicle
   • Spawns on the WEST lane (E#L-X) via a single-shot RL inflow at
     begin = ego_spawn_time.
   • Its route is set deterministically at episode start:
       - Even episodes  → West → North  (E#L-X → E#X-T)
       - Odd  episodes  → West → East   (E#L-X → E#X-R)

2. Background vehicles
   • Spawned on the SOUTH lane (E#D-X) at a fixed period
     (default: one vehicle every BG_SPAWN_PERIOD seconds).
   • All follow the S→N route (E#D-X → E#X-T).
   • Created via traci.vehicle.add() in additional_command(), so timing
     is perfectly reproducible regardless of SUMO's RNG.

3. No other stochastic elements — sigma=0, fixed speed_dev=0.

Usage
─────
    from envs.alpha_env_deterministic import make_deterministic_env

    EnvClass = make_deterministic_env("attention_continous")   # or any version tag
    env = EnvClass(env_params, sim_params, network, simulator='traci',
                   bg_spawn_period=4.0, ego_spawn_time=20.0)
"""

import os
import sys
import numpy as np
from typing import List, Type

sys.path.append(os.path.dirname(__file__))

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_BG_SPAWN_PERIOD = 4.0
DEFAULT_EGO_SPAWN_TIME  = 20.0

EGO_ROUTE_NORTH = ["E#L-X", "E#X-T"]
EGO_ROUTE_EAST  = ["E#L-X", "E#X-R"]

BG_ROUTE_ID    = "bg_route_SN"
BG_ROUTE_EDGES = ["E#D-X", "E#X-T"]
BG_VEH_TYPE    = "NonRL"


# ══════════════════════════════════════════════════════════════════════════════
# Mixin — pure deterministic logic, no base-class dependency
# ══════════════════════════════════════════════════════════════════════════════

class DeterministicMixin:
    """
    Drop-in mixin that adds deterministic spawning and route assignment to
    any AlphaEnv_v01 descendant.

    Constructor keyword arguments
    ─────────────────────────────
    bg_spawn_period : float
        Seconds between consecutive background vehicles (default 4.0).
    ego_spawn_time  : float
        Simulation time (s) at which the ego vehicle is released (default 20.0).

    These must be passed as keyword arguments *before* calling super().__init__().
    The factory class handles this correctly.
    """

    def _det_init(self, bg_spawn_period: float, ego_spawn_time: float):
        """Called by the factory subclass __init__ before super().__init__."""
        self._bg_spawn_period = bg_spawn_period
        self._ego_spawn_time  = ego_spawn_time
        self._episode_count   = 0
        self._bg_veh_counter  = 0
        self._bg_route_added  = False

    # ── Route helpers ──────────────────────────────────────────────────────────

    def _get_ego_route_for_episode(self) -> List[str]:
        """Return the ego route for the current episode (alternating N/E)."""
        return EGO_ROUTE_NORTH if self._episode_count % 2 == 0 else EGO_ROUTE_EAST

    def _ensure_bg_route(self):
        """Register the S→N background route with SUMO once per sim instance."""
        if self._bg_route_added:
            return
        try:
            existing = self.k.kernel_api.route.getIDList()
            if BG_ROUTE_ID not in existing:
                self.k.kernel_api.route.add(BG_ROUTE_ID, BG_ROUTE_EDGES)
            self._bg_route_added = True
        except Exception as exc:
            pass  # will retry next step

    # ── Background spawning ────────────────────────────────────────────────────

    def _spawn_bg_vehicle_if_due(self):
        """
        Spawn one background vehicle per spawn-period using time_counter.
        First vehicle appears at t = 0.
        """
        self._ensure_bg_route()
        expected = int(self.time_counter / self._bg_spawn_period) + 1
        while self._bg_veh_counter < expected:
            veh_id = f"bg_{self._episode_count}_{self._bg_veh_counter}"
            try:
                self.k.kernel_api.vehicle.add(
                    vehID=veh_id,
                    routeID=BG_ROUTE_ID,
                    typeID=BG_VEH_TYPE,
                    depart="now",
                    departLane="0",
                    departPos="base",
                    departSpeed="max",
                )
            except Exception:
                pass  # edge full or duplicate — skip gracefully
            self._bg_veh_counter += 1

    # ── Overrides ──────────────────────────────────────────────────────────────

    def _apply_non_rl_controls(self):
        """
        Called at every simulation micro-step, including during the reset()
        wait loop before the ego spawns. Spawning bg vehicles here guarantees
        they enter at t=1 (or whenever the first spawn is due) regardless of
        whether the ego has appeared yet.
        """
        self._spawn_bg_vehicle_if_due()
        super()._apply_non_rl_controls()

    def additional_command(self):
        """Run parent housekeeping (observed vehicle bookkeeping, etc.)."""
        super().additional_command()

    def reset(self, *, seed=None, options=None):
        """Reset and force the deterministic ego route."""
        self._bg_veh_counter = 0
        self._bg_route_added = False

        obs, info = super().reset(seed=seed, options=options)

        if self.agent_id is not None:
            desired_route = self._get_ego_route_for_episode()
            try:
                self.k.kernel_api.vehicle.setRoute(self.agent_id, desired_route)
            except Exception as exc:
                print(f"[DeterministicMixin] Could not set ego route: {exc}")

        self._episode_count += 1
        return obs, info

    def _compute_telemetry_stats(self):
        stats = super()._compute_telemetry_stats()
        ep_idx = self._episode_count - 1
        stats["ego_route"]      = "north" if ep_idx % 2 == 0 else "east"
        stats["bg_spawn_period"] = self._bg_spawn_period
        stats["episode_index"]  = ep_idx
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════

def _get_base_class(version: str):
    """
    Return the correct base environment class for a given version tag.
    Mirrors the logic in v0_1_evaluate.py::_get_env_class().
    """
    v = version.lower().strip()

    if v in ("heuristic_continuous", "heuristic_continous"):
        from alpha_env_v01 import AlphaEnv_v01
        return AlphaEnv_v01

    elif v in ("heuristic_discrete", "heuristic_discrete_5"):
        from alpha_env_v01_discrete import AlphaEnv_v01_Discrete_5
        return AlphaEnv_v01_Discrete_5

    elif v == "heuristic_discrete_3":
        from alpha_env_v01_discrete import AlphaEnv_v01_Discrete_3
        return AlphaEnv_v01_Discrete_3

    elif v == "heuristic_discrete_10":
        from alpha_env_v01_discrete import AlphaEnv_v01_Discrete_10
        return AlphaEnv_v01_Discrete_10

    elif v in ("attention_continuous", "attention_continous"):
        from alpha_env_v01_attention_continous import AlphaEnv_v01_Attention
        return AlphaEnv_v01_Attention

    elif v in ("attention_discrete", "attention_discrete_5"):
        from alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete_5
        return AlphaEnv_v01_AttentionDiscrete_5

    elif v == "attention_discrete_3":
        from alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete_3
        return AlphaEnv_v01_AttentionDiscrete_3

    elif v == "attention_discrete_10":
        from alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete_10
        return AlphaEnv_v01_AttentionDiscrete_10

    else:
        raise ValueError(
            f"Unknown version tag: '{version}'. "
            "Expected one of: heuristic_continous, heuristic_discrete_[3|5|10], "
            "attention_continous, attention_discrete_[3|5|10]."
        )


def make_deterministic_env(version: str) -> Type:
    """
    Dynamically create a DeterministicEnv class that inherits from the correct
    base class for the given version tag.

    Parameters
    ----------
    version : str
        Version tag, e.g. 'attention_continous', 'heuristic_discrete_10'.

    Returns
    -------
    A class (not an instance) that can be instantiated as:
        EnvClass(env_params, sim_params, network, simulator='traci',
                 bg_spawn_period=4.0, ego_spawn_time=20.0)
    """
    BaseClass = _get_base_class(version)

    class DeterministicEnv(DeterministicMixin, BaseClass):
        f"""Deterministic variant of {BaseClass.__name__} for velocity-profile analysis."""

        def __init__(
            self,
            env_params,
            sim_params,
            network,
            simulator="traci",
            bg_spawn_period: float = DEFAULT_BG_SPAWN_PERIOD,
            ego_spawn_time:  float = DEFAULT_EGO_SPAWN_TIME,
        ):
            # Initialise mixin state BEFORE calling super().__init__()
            # so that additional_command() doesn't crash on first call.
            self._det_init(bg_spawn_period, ego_spawn_time)
            super().__init__(env_params, sim_params, network, simulator)

    DeterministicEnv.__name__     = f"Deterministic_{BaseClass.__name__}"
    DeterministicEnv.__qualname__ = f"Deterministic_{BaseClass.__name__}"
    return DeterministicEnv
