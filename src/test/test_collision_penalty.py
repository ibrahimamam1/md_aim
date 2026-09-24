"""
test_collision_penalty.py
─────────────────────────
Verifies that the ego (RL) vehicle receives the sparse collision penalty on a
crash, exercising the REAL reward path (Env_N.step -> AlphaEnv_MO_SD.step ->
compute_reward -> compute_decomposed_reward) with only the SUMO/flow kernel
boundary stubbed (no simulator needed).

Checks:
  1. On collision, step() returns exactly fail_penalty (-15.0 by default).
  2. A custom fail_penalty (-25.0) is honored.
  3. On success, step() returns exactly goal_reward (+20.0).
  4. On dense (non-terminal) steps, the scalar is r_l only (progress +
     waiting penalty); no collision penalty is added.
  5. Reward plumbing: info["reward_dict"], info["vector_reward"], and
     final mo_telemetry all carry the penalty on the collision step.
"""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _install_simulator_import_stubs():
    """
    Make base_env_single.py importable without the flow / traci / sumolib
    packages installed. These stubs ONLY satisfy import-time dependencies;
    every code path exercised by the tests (Env_N.step, the reward chain,
    telemetry) is the real implementation. If the real packages are
    available they are used instead.
    """
    try:
        import flow  # noqa: F401
        import traci  # noqa: F401
        import sumolib  # noqa: F401
        return
    except Exception:
        pass

    class FatalFlowError(Exception):
        pass

    class FatalTraCIError(Exception):
        pass

    class TraCIException(Exception):
        pass

    miscutils = types.ModuleType("sumolib.miscutils")
    miscutils.getFreeSocketPort = lambda: 0
    sumolib = types.ModuleType("sumolib")
    sumolib.miscutils = miscutils

    pyglet_renderer = types.ModuleType("flow.renderer.pyglet_renderer")
    pyglet_renderer.PygletRenderer = object
    flow_warnings = types.ModuleType("flow.utils.flow_warnings")
    flow_warnings.deprecated_attribute = lambda *a, **k: None
    flow_exceptions = types.ModuleType("flow.utils.exceptions")
    flow_exceptions.FatalFlowError = FatalFlowError
    core_util = types.ModuleType("flow.core.util")
    core_util.ensure_dir = lambda p: None
    kernel_mod = types.ModuleType("flow.core.kernel")
    kernel_mod.Kernel = object
    traci_exceptions = types.ModuleType("traci.exceptions")
    traci_exceptions.FatalTraCIError = FatalTraCIError
    traci_exceptions.TraCIException = TraCIException

    stubs = {
        "flow": types.ModuleType("flow"),
        "flow.renderer": types.ModuleType("flow.renderer"),
        "flow.renderer.pyglet_renderer": pyglet_renderer,
        "flow.utils": types.ModuleType("flow.utils"),
        "flow.utils.flow_warnings": flow_warnings,
        "flow.utils.exceptions": flow_exceptions,
        "flow.core": types.ModuleType("flow.core"),
        "flow.core.util": core_util,
        "flow.core.kernel": kernel_mod,
        "traci": types.ModuleType("traci"),
        "traci.exceptions": traci_exceptions,
        "sumolib": sumolib,
        "sumolib.miscutils": miscutils,
    }
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)


_install_simulator_import_stubs()

from src.envs.alpha_env_mo_sd import AlphaEnv_MO_SD


# --------------------------------------------------------------------------- #
# Stubs for the flow/SUMO boundary
# --------------------------------------------------------------------------- #
class StubAccController:
    def get_action(self, env):
        return 0.0


class StubVehicles:
    """Mimics the subset of k.vehicle / kernel_api.vehicle used by the env."""

    def __init__(self, ids, ego_id="rl_0"):
        self._ids = list(ids)
        self._ego = ego_id

    def get_ids(self):
        return list(self._ids)

    def get_rl_ids(self):
        return [self._ego] if self._ego in self._ids else []

    def get_human_ids(self):
        return [i for i in self._ids if i != self._ego]

    def get_controlled_ids(self):
        return self.get_human_ids()

    def get_controlled_lc_ids(self):
        return []

    def get_2d_position(self, vid):
        return {"rl_0": (5.0, 0.0), "hv_1": (30.0, 0.0)}.get(vid, (0.0, 0.0))

    def get_speed(self, vid):
        return 8.0

    def get_accel(self, vid):
        return 0.0

    def get_heading(self, vid):
        return 90.0

    def get_edge(self, vid):
        return "E#R-X"

    def get_distance(self, vid):
        return 40.0

    def get_position(self, vid):
        return 10.0

    def get_x_by_id(self, vid):
        return 42.0

    def get_route(self, vid):
        return ["E#R-X", "E#X-R"]

    def get_length(self, vid):
        return 5.0

    def get_type(self, vid):
        return "passenger"

    def get_initial_speed(self, vid):
        return 0.0

    def get_acc_controller(self, vid):
        return StubAccController()

    def apply_acceleration(self, ids, accels):
        pass

    def apply_lane_change(self, ids, direction=None):
        pass

    def set_color(self, vid, color):
        pass

    def update_vehicle_colors(self):
        pass

    def set_observed(self, vid):
        pass

    def reset(self):
        pass

    def add(self, *args):
        pass

    def remove(self, vid):
        if vid in self._ids:
            self._ids.remove(vid)


class StubJunction:
    def get_ids(self):
        return []


class StubLane:
    def getIDList(self):
        return ["E#R-X_0", "E#X-R_0"]

    def getShape(self, lane_id):
        return [(0.0, 0.0), (100.0, 0.0)]

    def getLength(self, lane_id):
        return 100.0

    def getLinks(self, lane_id):
        return []


class StubSimulation:
    def __init__(self, colliding=None):
        self._colliding = list(colliding or [])

    def simulation_step(self):
        pass

    def getCollidingVehiclesIDList(self):
        return list(self._colliding)


class StubKernelApi:
    """Object passed as k.kernel_api (traci-style domains)."""

    def __init__(self, colliding=None):
        self.vehicle = StubVehicles(["rl_0", "hv_1"])
        self.lane = StubLane()
        self.simulation = StubSimulation(colliding)


class StubKernel:
    """Replaces the flow Kernel (k) after Env_N.__init__ is bypassed."""

    def __init__(self, colliding=None):
        self.vehicle = StubVehicles(["rl_0", "hv_1"])
        self.simulation = StubSimulation(colliding)
        self.junction = StubJunction()
        self.lane = StubLane()
        self.kernel_api = StubKernelApi(colliding)
        # Share vehicle/simulation instances between k and kernel_api so the
        # tests only have to mutate one place (Env_N reads from both).
        self.kernel_api.vehicle = self.vehicle
        self.kernel_api.simulation = self.simulation
        self.network = None  # wired in make_env

    def pass_api(self, api):
        self.kernel_api = api

    def update(self, reset=False):
        pass

    def close(self):
        pass


class StubNetwork:
    """Mimics the flow network object used by Env_N / AlphaEnv_MO_SD."""

    def __init__(self):
        self.net_params = mock.Mock()
        self.net_params.template = None  # forces DEFAULT_INTERNAL_CONNECTIONS
        self.initial_config = mock.Mock()
        self.initial_config.edges_distribution = ["E#R-X"]
        self.initial_config.shuffle = False
        self.name = "stub_net"

    def max_speed(self):
        return 55.0

    def edge_length(self, edge):
        return 100.0

    def length(self):
        return 400.0

    def generate_network(self, network):
        pass

    def generate_starting_positions(self, initial_config, num_vehicles):
        return [((0, 0), 0.0)] * num_vehicles, [0] * num_vehicles


class StubSimulationParams:
    """Mimics the subset of flow SumoParams the env touches."""

    def __init__(self):
        self.render = False
        self.save_render = False
        self.sight_radius = 25
        self.show_radius = False
        self.pxpm = 2
        self.port = None
        self.sim_step = 0.25
        self.restart_instance = False
        self.print_warnings = False
        self.emission_path = None
        self.seed = 42
        self.num_clients = 1


def make_env(fail_penalty=-15.0, goal_reward=20.0, colliding_ids=("rl_0",)):
    """
    Builds a real AlphaEnv_MO_SD with the flow Kernel and base-class
    constructor stubbed out, then manually replicates the base-class
    post-kernel initialization (telemetry, connection maps, initial state).
    """
    env_params = mock.Mock()
    env_params.sims_per_step = 1
    env_params.horizon = 180
    env_params.additional_params = {"max_accel": 2.6, "max_decel": 4.5}

    net = StubNetwork()
    env = AlphaEnv_MO_SD.__new__(AlphaEnv_MO_SD)

    # Seed attributes the subclass __init__ touches BEFORE super().__init__()
    env.prev_pos = {}
    env.absolute_position = {}
    env.max_neighbours = 5
    env.perception_radius = 100.0
    env.ego_obs_features = 4
    env.neighbour_obs_features = 5
    env.routes = {}
    env.last_progress = 0.0

    with mock.patch.object(AlphaEnv_MO_SD.__mro__[1], "__init__",
                           lambda self, *a, **k: None):
        AlphaEnv_MO_SD.__init__(
            env,
            env_params=env_params,
            sim_params=StubSimulationParams(),
            network=net,
            simulator="traci",
            fail_penalty=fail_penalty,
            goal_reward=goal_reward,
        )

    # Manually replicate Env_N.__init__ post-kernel logic
    env.env_params = env_params
    env.network = net
    env.net_params = net.net_params
    env.initial_config = net.initial_config
    sim_params = StubSimulationParams()
    sim_params.port = 0
    env.sim_params = sim_params
    env.should_render = False
    env.time_counter = 0.0
    env.step_counter = 0
    env.step_counter_within_rl_step = 0
    env.initial_state = {}
    env.state = None
    env.rl_agent_spawned = False
    env.sim_step = 0.25
    env.simulator = "traci"
    env._init_telemetry()

    env.k = StubKernel(colliding_ids)
    env.k.network = net
    env.available_routes = {}
    env.initial_ids = ["rl_0", "hv_1"]
    env.initial_vehicles = env.k.vehicle
    env.initial_junction = env.k.junction

    env.internal_connections, env.internal_to_out = env._build_connection_maps()
    env.setup_initial_state()

    env.agent_id = "rl_0"
    return env


class TestEgoCollisionPenalty(unittest.TestCase):
    """The ego vehicle must receive the sparse collision penalty on crash."""

    def test_collision_step_returns_fail_penalty(self):
        """Ego crashes: the returned scalar reward is exactly fail_penalty."""
        env = make_env(fail_penalty=-15.0)
        env.k.kernel_api.simulation._colliding = ["rl_0"]
        obs, reward, terminated, truncated, info = env.step(0.0)
        self.assertEqual(float(reward), -15.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_collision_penalty_configurable(self):
        """A custom fail_penalty (-25.0) is honored exactly."""
        env = make_env(fail_penalty=-25.0)
        env.k.kernel_api.simulation._colliding = ["rl_0"]
        _, reward, terminated, _, _ = env.step(0.0)
        self.assertEqual(float(reward), -25.0)
        self.assertTrue(terminated)

    def test_success_step_returns_goal_reward(self):
        """Ego exits the network (not crashed, no longer present): +goal_reward."""
        env = make_env()
        env.k.kernel_api.simulation._colliding = []
        env.k.vehicle._ids = ["hv_1"]  # ego departed
        _, reward, terminated, _, _ = env.step(0.0)
        self.assertEqual(float(reward), 20.0)
        self.assertTrue(terminated)

    def test_dense_step_has_no_collision_penalty(self):
        """Normal dense step: scalar == r_l (progress + waiting); r_col == 0."""
        env = make_env()
        env.k.kernel_api.simulation._colliding = []
        _, reward, terminated, _, info = env.step(0.0)
        self.assertFalse(terminated)
        rd = info["reward_dict"]
        self.assertEqual(rd["collision_penalty"], 0.0)
        self.assertEqual(rd["goal_reward"], 0.0)
        # scalar == r_l + r_col with r_col == 0
        self.assertEqual(float(reward), rd["total_long_term_reward"])
        # Dense steps stay small: no sparse terminal leaking into them
        self.assertLess(abs(float(reward)), 5.0)

    def test_reward_dict_vector_and_mo_telemetry_carry_penalty(self):
        """collision_penalty / vector_reward / final mo_telemetry all agree."""
        env = make_env(fail_penalty=-15.0)
        env.k.kernel_api.simulation._colliding = ["rl_0"]
        _, reward, terminated, _, info = env.step(0.0)

        rd = info["reward_dict"]
        self.assertEqual(rd["collision_penalty"], -15.0)
        self.assertEqual(rd["total_safety_reward"], -15.0)
        self.assertEqual(rd["r_s"], -15.0)
        # scalar == r_l + r_col
        self.assertEqual(float(reward), rd["r_l"] + rd["collision_penalty"])

        # vector_reward = [r_l, r_s] -> [0.0, -15.0]
        vr = info["vector_reward"]
        self.assertEqual(float(vr[0]), 0.0)
        self.assertEqual(float(vr[1]), -15.0)

        # Final-step mo_telemetry
        mo = info["mo_telemetry"]
        self.assertEqual(mo["collision"], 1)
        self.assertEqual(mo["success"], 0)
        self.assertEqual(mo["collision_penalty"], -15.0)
        self.assertEqual(mo["total_safety_reward"], -15.0)
        self.assertEqual(mo["total_reward"], -15.0)

    def test_base_telemetry_flags_collision(self):
        """Env_N telemetry marks the agent as collided on the crash step."""
        env = make_env()
        env.k.kernel_api.simulation._colliding = ["rl_0"]
        _, _, _, _, info = env.step(0.0)
        base = info["telemetry"]
        self.assertTrue(base["agent_collision"])
        self.assertFalse(base["agent_success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
