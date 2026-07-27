"""Base environment class. This is the parent of all other environments."""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import os
import atexit
import numpy as np
import random
import gymnasium as gym
from flow.renderer.pyglet_renderer import PygletRenderer as Renderer
from flow.utils.flow_warnings import deprecated_attribute
from gymnasium.spaces import Box
from traci.exceptions import FatalTraCIError
from traci.exceptions import TraCIException
from shapely.geometry import LineString, Point
import sumolib
from flow.core.util import ensure_dir
from flow.core.kernel import Kernel
from flow.utils.exceptions import FatalFlowError
from shapely.geometry import LineString, Point

# ANSI color codes for debugging
BLUE = '\033[94m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'

class Env_N(gym.Env, metaclass=ABCMeta):
    """
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 env_params,
                 sim_params,
                 network=None,
                 simulator='traci',
                 scenario=None,
                 render_mode=None
                 ):
        
        self.agent_id = None 
        self.env_params = env_params
        if scenario is not None:
            deprecated_attribute(self, "scenario", "network")
        self.network = scenario if scenario is not None else network
        self.net_params = self.network.net_params
        self.initial_config = self.network.initial_config
        self.sim_params = deepcopy(sim_params)
        
        # Rendering setup
        self.should_render = self.sim_params.render
        self.sim_params.render = False 
        
        # Unique port generation to prevent collisions during parallel training
        self.sim_params.port = sumolib.miscutils.getFreeSocketPort()
        
        self.time_counter = 0
        self.step_counter = 0
        self.step_counter_within_rl_step = 0
        self.initial_state = {}
        self.state = None
        self.rl_agent_spawned = False

        self.sim_step = sim_params.sim_step
        self.simulator = simulator

        # Telemetry Accumulators
        self._init_telemetry()

        # --- FLOW KERNEL INITIALIZATION ---
        self.k = Kernel(simulator=self.simulator, sim_params=self.sim_params)
        self.k.network.generate_network(self.network)
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))
        
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)
        
        self.k.pass_api(kernel_api)
        self.available_routes = self.k.network.rts
        self.initial_ids = deepcopy(self.network.vehicles.ids)

        # Snapshot for restarts
        self.k.vehicle.kernel_api = None
        self.k.vehicle.master_kernel = None
        self.initial_vehicles = deepcopy(self.k.vehicle)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        # Snapshot for junctions
        self.k.junction.kernel_api = None
        self.k.junction.master_kernel = None
        self.initial_junction = deepcopy(self.k.junction)
        self.k.junction.kernel_api = self.k.kernel_api
        self.k.junction.master_kernel = self.k

        self.setup_initial_state()
        
        self.internal_connections = {
            ('E#D-X', 'E#X-R'): ':X_6',
            ('E#D-X', 'E#X-T'): ':X_7',
            ('E#D-X', 'E#X-L'): ':X_8',
            ('E#L-X', 'E#X-D'): ':X_9',
            ('E#L-X', 'E#X-R'): ':X_10',
            ('E#L-X', 'E#X-T'): ':X_11',
            ('E#R-X', 'E#X-T'): ':X_3',
            ('E#R-X', 'E#X-L'): ':X_4',
            ('E#R-X', 'E#X-D'): ':X_5',
            ('E#T-X', 'E#X-L'): ':X_0',
            ('E#T-X', 'E#X-D'): ':X_1',
            ('E#T-X', 'E#X-R'): ':X_2',
        }
        # Renderer Setup
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            save_render = self.sim_params.save_render
            sight_radius = self.sim_params.sight_radius
            pxpm = self.sim_params.pxpm
            show_radius = self.sim_params.show_radius
            network = []
            for lane_id in self.k.kernel_api.lane.getIDList():
                _lane_poly = self.k.kernel_api.lane.getShape(lane_id)
                lane_poly = [i for pt in _lane_poly for i in pt]
                network.append(lane_poly)
            self.renderer = Renderer(
                network,
                self.sim_params.render,
                save_render,
                sight_radius=sight_radius,
                pxpm=pxpm,
                show_radius=show_radius)
            self.render(reset=True)
            self.path = os.path.expanduser('~')+'/flow_rendering/' + self.network.name
            os.makedirs(self.path, exist_ok=True)
        elif self.sim_params.render in [True, False]:
            self.path = os.path.expanduser('~')+'/flow_rendering/' + self.network.name
            os.makedirs(self.path, exist_ok=True)
        else:
             raise FatalFlowError('Mode %s is not supported!' % self.sim_params.render)
        
        atexit.register(self.terminate)

    def restart_simulation(self, sim_params, render=None):
        """Restart simulation logic (Kept identical to original)."""
        self.k.close()
        if self.simulator == 'traci':
            self.k.simulation.sumo_proc.kill()

        if render is not None:
            self.sim_params.render = render
        if sim_params.emission_path is not None:
            ensure_dir(sim_params.emission_path)
            self.sim_params.emission_path = sim_params.emission_path

        self.k.network.generate_network(self.network)
        self.k.vehicle.initialize(deepcopy(self.network.vehicles))
        kernel_api = self.k.simulation.start_simulation(
            network=self.k.network, sim_params=self.sim_params)
        self.k.pass_api(kernel_api)
        self.setup_initial_state()

    def _is_in_control_zone(self, veh_id):
        """
        Determines if a vehicle is in the control zone.
        """
        position = self.k.vehicle.get_2d_position(veh_id)
        in_box_x = -12 <= position[0] <= 12
        in_box_y = -12 <= position[1] <= 12
            
        return in_box_x and in_box_y
             
    # --- TELEMETRY HELPERS ---
    def _init_telemetry(self):
        """Resets telemetry storage for a new episode (Agent Only)."""
        self.telemetry = {
            # --- Agent-only telemetry ---
            "agent_speeds": [],          # List of speeds for every step agent is alive
            "agent_accelerations": [],   # List of accels for every step agent is alive
            "agent_waiting_time": 0.0,   # Accumulated time agent speed < 0.1
            "agent_spawn_time": None,    # Time step agent first appeared
            "agent_finish_time": None,   # Time step agent left (success or crash)
            "agent_collision": False,    # Did agent collide?
            "agent_success": False,      # Did agent reach goal?
            "agent_total_distance": 0.0, # Approximate distance travelled
            "reward_speed_total": 0.0,
            "reward_time_total": 0.0,
            "reward_action_total": 0.0,
            "reward_terminal_total": 0.0,
        }

    def _update_telemetry_step(self):
        """
        Updates internal accumulators for the specific RL agent only.
        """
        current_time = self.time_counter
        
        # If agent hasn't spawned or is already gone, do nothing
        if self.agent_id is None:
            return

        # Check if agent is currently in the network
        if self.agent_id in self.k.vehicle.get_ids():
            # 1. Capture Spawn Time
            if self.telemetry["agent_spawn_time"] is None:
                self.telemetry["agent_spawn_time"] = current_time

            # 2. Get Agent Physics
            speed = self.k.vehicle.get_speed(self.agent_id)
            accel = self.k.vehicle.get_accel(self.agent_id)
            
            # 3. Update Stats
            if speed is not None:
                self.telemetry["agent_speeds"].append(speed)
                # Track waiting time (speed < 0.1 m/s)
                if speed < 0.1:
                    self.telemetry["agent_waiting_time"] += self.sim_step
                # Track distance (Speed * Time)
                self.telemetry["agent_total_distance"] += speed * self.sim_step

            if accel is not None:
                self.telemetry["agent_accelerations"].append(accel)

        # 4. Check Collisions specifically for this agent
        colliding_ids = self.k.kernel_api.simulation.getCollidingVehiclesIDList()
        if self.agent_id in colliding_ids:
            self.telemetry["agent_collision"] = True
            self.telemetry["agent_finish_time"] = current_time

    def _compute_telemetry_stats(self):
        """
        Returns the raw agent statistics.
        Called only when terminated is True.
        """
        import numpy as np
        
        # Calculate Averages
        avg_speed = np.mean(self.telemetry["agent_speeds"]) if self.telemetry["agent_speeds"] else 0.0
        avg_accel = np.mean(self.telemetry["agent_accelerations"]) if self.telemetry["agent_accelerations"] else 0.0

        # Calculate Duration
        spawn_time = self.telemetry["agent_spawn_time"]
        # If finish time wasn't set (e.g. timeout), use current time
        finish_time = self.telemetry["agent_finish_time"] if self.telemetry["agent_finish_time"] else self.time_counter
        
        duration = 0.0
        if spawn_time is not None:
            duration = finish_time - spawn_time

        return {
            "agent_success": self.telemetry["agent_success"],
            "agent_collision": self.telemetry["agent_collision"],
            "agent_travel_time": duration,
            "agent_waiting_time": self.telemetry["agent_waiting_time"],
            "agent_avg_speed": float(avg_speed),
            "agent_avg_accel": float(avg_accel),
            "agent_total_distance": self.telemetry["agent_total_distance"],
            "episode_length": self.time_counter,
            "reward_speed": self.telemetry["reward_speed_total"],
            "reward_time": self.telemetry["reward_time_total"],
            "reward_action": self.telemetry["reward_action_total"],
            "reward_terminal": self.telemetry["reward_terminal_total"]
        }
    
    def step(self, action):
        """
        Advance the environment by one step.
        """
        self.step_counter_within_rl_step = 0
        
        # Snapshot of agents before step
        sorted_ids = set(self.sorted_ids)
        if self.agent_id in sorted_ids:
            self.apply_rl_actions(action) 
        if hasattr(self, "additional_command"):
            self.additional_command()
        
        # 2. Simulation Step (Inner Loop)
        for inner_step in range(self.env_params.sims_per_step):
            self.time_counter += self.sim_step
            self.step_counter += 1
            self.step_counter_within_rl_step = inner_step
            
            self._apply_non_rl_controls()
                
            # Advance Simulator
            self.k.simulation.simulation_step()
            self.k.update(reset=False)
            
            self._update_telemetry_step()
            
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()
       
        
        # 3. Retrieve Observations
        obs = self.get_state()
        colliding_ids = set(self.k.kernel_api.simulation.getCollidingVehiclesIDList())
        rl_ids_set = set(self.k.vehicle.get_rl_ids())
        rl_crash_ids = colliding_ids & rl_ids_set  # Only RL vehicles that actually crashed
        
        crashed = self.agent_id in rl_crash_ids
        goal_reached = (self.agent_id not in self.sorted_ids) and not crashed  #agent spawned then left
        truncated = (self.time_counter >= self.env_params.horizon)
       
        # Only terminate if an RL agent crashed OR successfully arrived
        terminated = crashed or goal_reached
        
        # Update agent-only telemetry flags
        if goal_reached:
           self.telemetry["agent_success"] = True
           if self.telemetry["agent_finish_time"] is None:
                self.telemetry["agent_finish_time"] = self.time_counter
        if crashed:
            self.telemetry["agent_collision"] = True
        
        reward = self.compute_reward(self.agent_id, crashed, goal_reached, current_action=action)
        
        # --- COMPUTE TELEMETRY ---
        telemetry_stats = None
        if (terminated or truncated):
            telemetry_stats = self._compute_telemetry_stats()
        
        infos = {}
        if telemetry_stats is not None:
            infos["telemetry"] = telemetry_stats
        infos["neighbors"] = self.last_neighbors_info 
        return obs, reward, terminated, truncated, infos

    def _apply_non_rl_controls(self):
        """Helper to handle IDM/LaneChange controllers for non-RL vehicles."""
        if len(self.k.vehicle.get_controlled_ids()) > 0:
            accel = []
            for veh_id in self.k.vehicle.get_controlled_ids():
                action = self.k.vehicle.get_acc_controller(veh_id).get_action(self)
                accel.append(action)
            self.k.vehicle.apply_acceleration(
                self.k.vehicle.get_controlled_ids(), accel)

        if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
            direction = []
            for veh_id in self.k.vehicle.get_controlled_lc_ids():
                target_lane = self.k.vehicle.get_lane_changing_controller(veh_id).get_action(self)
                direction.append(target_lane)
            self.k.vehicle.apply_lane_change(
                self.k.vehicle.get_controlled_lc_ids(), direction=direction)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment with a spawn safety valve.
        """
        # --- RESET TELEMETRY ---
        self._init_telemetry()
        # -----------------------

        # Call parent reset (if using gymnasium.Env, though Env_N inherits directly from gym.Env)
        super().reset(seed=seed)
        
        self.last_action = 0.0
        self.last_progress = 0.0
        self.last_neighbors_info = []
        # Ensure observation space is respected (Box vs Discrete check might be needed depending on subclass)
        if hasattr(self.observation_space, 'shape'):
            self.last_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        else:
            self.last_obs = np.zeros(0, dtype=np.float32)

        self.time_counter = 0
        self.rl_agent_spawned = False
        self.agent_id = None

        if self.should_render:
            self.sim_params.render = True
            self.restart_simulation(self.sim_params)

        # Standard Flow restart logic
        if self.sim_params.restart_instance or (self.step_counter > 2e6 and self.simulator != 'aimsun'):
            self.step_counter = 0
            # Handle seeding for deterministic behavior if needed
            if seed is not None:
                self.sim_params.seed = seed
            else:
                self.sim_params.seed = random.randint(0, 100000)
            
            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            self.k.junction = deepcopy(self.initial_junction)
            self.k.junction.master_kernel = self.k
            self.restart_simulation(self.sim_params)
        elif self.initial_config.shuffle:
            self.setup_initial_state()

        if self.simulator == 'traci':
            try:
                for veh_id in self.k.kernel_api.vehicle.getIDList():
                    self.k.vehicle.remove(veh_id)
            except:
                pass

        self.k.vehicle.reset()

        # Re-add initial vehicles
        for veh_id in self.initial_ids:
            type_id, edge, lane_index, pos, speed = self.initial_state[veh_id]
            try:
                self.k.vehicle.add(veh_id, type_id, edge, lane_index, pos, speed)
            except (FatalTraCIError, TraCIException):
                self.k.vehicle.remove(veh_id)
                if self.simulator == 'traci':
                    self.k.kernel_api.vehicle.remove(veh_id)
                self.k.vehicle.add(veh_id, type_id, edge, lane_index, pos, speed)

        self.k.simulation.simulation_step()
        self.k.update(reset=True)
        
        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()
        
        
        while not self.k.vehicle.get_rl_ids():
            self._apply_non_rl_controls()
            self.k.simulation.simulation_step()
            self.k.update(reset=False)
            self.time_counter += self.sim_step
            self.step_counter += 1

        # Now that the agent exists, grab the FIRST real observation
        rl_ids = self.k.vehicle.get_rl_ids()
        self.agent_id = rl_ids[0]  
        self.rl_agent_spawned = True
        self.k.vehicle.set_color(self.agent_id, (255, 0, 0))
            
        obs = self.get_state()
        return obs, {}
    

    def _get_vehicle_polyline(self, veh_id):
        """
        Builds a continuous Shapely LineString of the vehicle's future path, 
        stitching the gap across the junction using known internal connections.
        """
        current_edge = self.k.vehicle.get_edge(veh_id)
        route = self.k.vehicle.get_route(veh_id)
        pos = self.k.vehicle.get_position(veh_id)
     
        # 1. Start with the current edge's shape
        # Note: If Flow complains about missing shapes, you may need to append '_0' 
        # to target the specific lane, e.g., self.k.network.get_edge_shape(f"{current_edge}_0")
        coords = list(self.k.kernel_api.lane.getShape(current_edge + "_0"))
     
        # 2. Check if we need to stitch the intersection gap
        if not current_edge.startswith(':'):
            try:
                current_idx = route.index(current_edge)
                if current_idx + 1 < len(route):
                    next_edge = route[current_idx + 1]
                    
                    # 3. Lookup the internal edge from our dictionary
                    internal_edge = self.internal_connections.get((current_edge, next_edge))
                    
                    if internal_edge:
                        # Fetch internal junction shape. Append '_0' to target the lane 
                        # if Flow's wrapper requires lane IDs instead of edge IDs.
                        internal_lane = f"{internal_edge}_0" 
                        coords.extend(self.k.kernel_api.lane.getShape(internal_lane))
                    
                    # 4. Add the next macro edge's shape
                    coords.extend(self.k.kernel_api.lane.getShape(next_edge + "_0"))
            except ValueError:
                pass # Vehicle is likely at the very end of its route
                
        # Handle the case where the vehicle is already inside the intersection (on an internal edge)
        elif current_edge.startswith(':'):
            try:
                # If inside the intersection, route[0] is usually the target outgoing edge
                next_edge = route[0] if route else None
                if next_edge:
                    coords.extend(self.k.kernel_api.lane.getShape(next_edge + "_0"))
            except IndexError:
                pass
    
        # Fallback to avoid Shapely crashing on single-coordinate lines
        if len(coords) < 2:
            x, y = self.k.vehicle.get_2d_position(veh_id)
            # Create a tiny arbitrary line indicating a stopped/lost vehicle
            return LineString([(x, y), (x+0.1, y+0.1)]), pos
            
        return LineString(coords), pos 

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.""" 
        return self.k.vehicle.get_rl_ids()
    
    def apply_rl_actions(self, action):
        self._apply_rl_actions(action)

    @abstractmethod
    def _apply_rl_actions(self, rl_actions):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def compute_reward(self, agent_id, fail, goal_reached, **kwargs):
        pass

    def setup_initial_state(self):
        if isinstance(self.initial_config.edges_distribution, list):
            random.shuffle(self.initial_config.edges_distribution)

        if self.initial_config.shuffle:
            random.shuffle(self.initial_ids)

        start_pos, start_lanes = self.k.network.generate_starting_positions(
            initial_config=self.initial_config,
            num_vehicles=len(self.initial_ids))

        occupied_edges = set()

        for i, veh_id in enumerate(self.initial_ids):
            type_id = self.k.vehicle.get_type(veh_id)
            pos = start_pos[i][1]
            speed = self.k.vehicle.get_initial_speed(veh_id)

            available_edges = [e for e in self.initial_config.edges_distribution if e not in occupied_edges]
            if available_edges:
                edge = random.choice(available_edges)
            else:
                edge = random.choice(self.initial_config.edges_distribution)
            occupied_edges.add(edge)

            self.initial_state[veh_id] = (type_id, edge, 0, pos, speed)

    def additional_command(self):
        pass
    
    def terminate(self):
        try:
            self.k.close()
            if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
                self.renderer.close()
        except:
            pass
    def render(self, reset=False, buffer_length=5):
        """Render a frame.

        Parameters
        ----------
        reset : bool
            set to True to reset the buffer
        buffer_length : int
            length of the buffer
        """
        if self.sim_params.render in ['gray', 'dgray', 'rgb', 'drgb']:
            # render a frame
            self.pyglet_render()

            # cache rendering
            if reset:
                self.frame_buffer = [self.frame.copy() for _ in range(5)]
                self.sights_buffer = [self.sights.copy() for _ in range(5)]
            else:
                if self.step_counter % int(1/self.sim_step) == 0:
                    self.frame_buffer.append(self.frame.copy())
                    self.sights_buffer.append(self.sights.copy())
                if len(self.frame_buffer) > buffer_length:
                    self.frame_buffer.pop(0)
                    self.sights_buffer.pop(0)
        elif (self.sim_params.render is True) and self.sim_params.save_render:
            # sumo-gui render
            self.k.kernel_api.gui.screenshot("View #0", self.path+"/frame_%06d.png" % self.time_counter)

    def pyglet_render(self):
        """Render a frame using pyglet."""
        # get human and RL simulation status
        human_idlist = self.k.vehicle.get_human_ids()
        machine_idlist = self.k.vehicle.get_rl_ids()
        human_logs = []
        human_orientations = []
        human_dynamics = []
        machine_logs = []
        machine_orientations = []
        machine_dynamics = []
        max_speed = self.k.network.max_speed()
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if 'track' in id:
                machine_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                machine_orientations.append(
                    self.k.vehicle.get_orientation(id))
                machine_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
            else:
                human_logs.append(
                    [self.k.vehicle.get_timestep(id),
                     self.k.vehicle.get_timedelta(id),
                     id])
                human_orientations.append(
                    self.k.vehicle.get_orientation(id))
                human_dynamics.append(
                    self.k.vehicle.get_speed(id)/max_speed)
        for id in machine_idlist:
            machine_logs.append(
                [self.k.vehicle.get_timestep(id),
                 self.k.vehicle.get_timedelta(id),
                 id])
            machine_orientations.append(
                self.k.vehicle.get_orientation(id))
            machine_dynamics.append(
                self.k.vehicle.get_speed(id)/max_speed)

        # step the renderer
        self.frame = self.renderer.render(human_orientations,
                                          machine_orientations,
                                          human_dynamics,
                                          machine_dynamics,
                                          human_logs,
                                          machine_logs)

        # get local observation of RL vehicles
        self.sights = []
        for id in human_idlist:
            # Force tracking human vehicles by adding "track" in vehicle id.
            # The tracked human vehicles will be treated as machine vehicles.
            if "track" in id:
                orientation = self.k.vehicle.get_orientation(id)
                sight = self.renderer.get_sight(
                    orientation, id)
                self.sights.append(sight)
        for id in machine_idlist:
            orientation = self.k.vehicle.get_orientation(id)
            sight = self.renderer.get_sight(
                orientation, id)
            self.sights.append(sight)


