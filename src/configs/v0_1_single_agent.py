import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime

parser = argparse.ArgumentParser(description="Train or evaluate the AlphaEnv PPO agent.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--train", action="store_true", help="Run training loop.")
group.add_argument("--eval",  metavar="CHECKPOINT_PATH",
                   help="Path to a checkpoint zip file to load and evaluate.")
parser.add_argument("--version", choices=["heuristic_discrete", "heuristic_continuous", "attention_discrete", "attention_continous"],
                    default="heuristic_discrete", help="Environment version to use.")
args = parser.parse_args()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from networks.uniform_random import UniformRandomNetwork as myNet
from flow.core.params import (
    VehicleParams, NetParams, InitialConfig, TrafficLightParams,
    EnvParams, SumoParams, SumoCarFollowingParams, InFlows,
)
from flow.controllers import RLController, IDMController
from src.utils.plot_train_curves import plot_results

# ---------------------------------------------
# SB3 Imports
# ---------------------------------------------
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

IDM_acceleration_controller = IDMController
RL_vehicle_acceleration_controller = RLController

myTag = "AlphaV0.1_Heuristic_Discrete"
min_gap       = 2.5
max_accel     = 2.6
max_decel     = 4.5
max_speed     = 55
initial_speed = 0
speed_factor  = 1.0
speed_dev     = 0.1
impatience    = 0.0
car_follow_model = "IDM"
sigma = 0
tau   = 0.8
horizon = 180
sim_step = 0.25
warmup_steps = 50
number_of_sim_steps_per_RlAction_step = 1

############### VEHICLE Configuration ##########################
num_rl_vehicles      = 0
num_non_rl_vehicles  = 0

rl_speed_mode    = 0
non_rl_speed_mode = 0

vehicles = VehicleParams()

RL_car_following_params = SumoCarFollowingParams(
    speed_mode=rl_speed_mode,
    accel=max_accel, decel=max_decel,
    sigma=sigma, tau=tau,
    min_gap=min_gap, max_speed=max_speed,
    speed_factor=speed_factor, speed_dev=speed_dev,
    impatience=impatience, car_follow_model=car_follow_model,
)
NonRL_car_following_params = SumoCarFollowingParams(
    speed_mode=non_rl_speed_mode,
    accel=max_accel, decel=max_decel,
    sigma=sigma, tau=tau,
    min_gap=min_gap, max_speed=max_speed,
    speed_factor=speed_factor, speed_dev=speed_dev,
    impatience=impatience, car_follow_model=car_follow_model,
)

vehicles.add(
    veh_id="RL",
    acceleration_controller=(RL_vehicle_acceleration_controller, {}),
    initial_speed=0,
    num_vehicles=num_rl_vehicles,
    car_following_params=RL_car_following_params,
    lane_change_params=None,
    color="blue",
)
vehicles.add(
    veh_id="NonRL",
    acceleration_controller=(IDM_acceleration_controller, {}),
    initial_speed=initial_speed,
    num_vehicles=num_non_rl_vehicles,
    car_following_params=NonRL_car_following_params,
    lane_change_params=None,
    color="red",
)

############################# InFlow Configuration #########################
high = 400
medium = 275
low = 150
traffic_rate = {"N": high, "S": high, "W": medium, "E": high}

inflow = InFlows()
 
inflow.add(veh_type="NonRL", edge="E#T-X", probability=traffic_rate["N"]/3600,
           depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
inflow.add(veh_type="NonRL", edge="E#R-X", probability=traffic_rate["E"]/3600,
           depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
inflow.add(veh_type="NonRL", edge="E#D-X", probability=traffic_rate["S"]/3600,
           depart_lane=0, depart_speed=initial_speed, begin=1, color="green")
inflow.add(veh_type="NonRL", edge="E#L-X", probability=traffic_rate["W"]/3600,
           depart_lane=0, depart_speed=initial_speed, begin=1, color="green")

inflow.add(veh_type="RL", edge="E#L-X", probability=0.3,
           depart_lane=0, depart_speed=initial_speed, begin=warmup_steps,
            color="green")

root_dir        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_file_dir = os.path.join(root_dir, "results")
net_file_dir    = os.path.join(root_dir, "networks")

net_file_name = "100m_right_before_left.net.xml"
net_file= os.path.join(net_file_dir, net_file_name)

net_params = NetParams(osm_path=None, template=net_file, inflows=inflow)

EDGES_DISTRIBUTION = ["E#D-X", "E#L-X", "E#R-X", "E#T-X"]

initial_config = InitialConfig(
    shuffle=False, spacing="uniform", min_gap=12, perturbation=5.0,
    x0=5, bunching=0, lanes_distribution=float("inf"),
    edges_distribution=EDGES_DISTRIBUTION, additional_params=None,
)
env_params = EnvParams(
    additional_params={"max_accel": max_accel, "max_decel": max_decel,
                       "target_velocity": max_speed, "sort_vehicles": False},
    horizon=horizon, warmup_steps=5, sims_per_step=number_of_sim_steps_per_RlAction_step,
    evaluate=False, clip_actions=True,
)

sim_params = SumoParams(
    port=None, sim_step=sim_step, emission_path=None,
    lateral_resolution=None, no_step_log=True, render=False, save_render=False,
    sight_radius=25, show_radius=False, pxpm=2, force_color_update=False,
    overtake_right=False, seed=42, restart_instance=True, print_warnings=False,
    teleport_time=0, num_clients=1, color_by_speed=False, use_ballistic=False,
)

flow_params = dict(
    exp_tag=myTag, network=myNet, simulator="traci",
    sim=sim_params, env=env_params, net=net_params, veh=vehicles, initial=initial_config,
)

# ---------------------------------------------
# SB3 Environment Setup & Callbacks
# ---------------------------------------------

def create_flow_env(env_config):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.version == "heuristic_continuous":
        from src.envs.alpha_env_v01 import AlphaEnv_v01 as EnvClass
    elif args.version == "heuristic_discrete":
        from src.envs.alpha_env_v01_discrete import AlphaEnv_v01_Discrete as EnvClass
    elif args.version == "attention_discrete":
        from src.envs.alpha_env_v01_attention_discrete import AlphaEnv_v01_AttentionDiscrete as EnvClass
    elif args.version == "attention_continous":
        from src.envs.alpha_env_v01_attention_continous import AlphaEnv_v01_Attention as EnvClass

    params       = flow_params
    _vehicles    = deepcopy(params["veh"])
    _net_params  = params["net"]
    _sim_params  = deepcopy(params["sim"])
    _sim_params.render = env_config.get("render", False)
    network_class = params["network"]
    _initial_config = params.get("initial", InitialConfig())
    traffic_lights  = params.get("tls", TrafficLightParams())

    network = network_class(
        name="AlphaEnv-Check",
        vehicles=_vehicles,
        net_params=_net_params,
        initial_config=_initial_config,
        traffic_lights=traffic_lights,
    )
    
    env = EnvClass(
        env_params=params["env"],
        sim_params=_sim_params,
        network=network,
        simulator=params["simulator"],
    )
    # Wrap in Monitor to log episode returns/lengths standard to SB3
    return Monitor(env)

class TrafficCallback(BaseCallback):
    """
    Custom callback for logging telemetry metrics to TensorBoard
    """
    def __init__(self, verbose=0):
        super(TrafficCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if environment provided an info dictionary at this step
        for info in self.locals.get("infos", []):
            if "telemetry" in info:
                telemetry = info["telemetry"]
                if telemetry is None:
                    continue
                
                # Log metrics to TensorBoard (averaged automatically over the rollout)
                self.logger.record("custom_metrics/collision", 1.0 if telemetry.get("agent_collision", False) else 0.0)
                self.logger.record("custom_metrics/success", 1.0 if telemetry.get("agent_success", False) else 0.0)
                self.logger.record("custom_metrics/avg_speed", float(telemetry.get("agent_avg_speed", 0.0)))
                
        return True

def linear_schedule_with_floor(initial_value: float, min_value: float):
    """
    Linear learning rate schedule that decays to a minimum floor.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 down to 0.0
        decayed_lr = progress_remaining * initial_value
        return max(min_value, decayed_lr)
    return func

# ---------------------------------------------
# Checkpoint helpers
# ---------------------------------------------
ENV_NAME  = "alpha_env_v01_" + args.version
ALGO_NAME = "PPO"

CHECKPOINT_ROOT = os.path.join(
    os.getcwd(), "checkpoints/v0_1",
    f"{args.version}_{ENV_NAME}_{ALGO_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
TENSORBOARD_DIR = os.path.join(os.getcwd(), "tensorboard_logs/v0_1_",f"{args.version}")
RUN_NAME = f"flow_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
TENSORBOARD_RUN_DIR = os.path.join(TENSORBOARD_DIR, RUN_NAME)


from src.models.attention_model import AttentionFeatureExtractor

def train():
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    
    print(f"\n--- TRAINING START (Discrete - SB3) ---")
    print(f"TensorBoard → {TENSORBOARD_RUN_DIR}")
    
    num_workers = 8
    n_steps = 1024 
    
    # 800 iterations * 8192 batch size = 6,553,600 total timesteps
    total_timesteps = 1500000
    
    # Vectorized environments for multi-processing
    def make_env():
        return create_flow_env({"render": False})
    
    policy_kwargs = None
    # FIX 2: Correct string method (startswith)
    if args.version.startswith("attention"):
        policy_kwargs = dict(
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                ego_features=4, 
                neighbor_features=5, 
                max_neighbors=5,
                embed_dim=64, 
                num_heads=4
            ),
            net_arch=dict(pi=[256, 256], vf=[256, 256]) 
        )

    vec_env = SubprocVecEnv([make_env for _ in range(num_workers)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule_with_floor(3e-4, 1e-5),
        n_steps=n_steps,
        batch_size=256,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.25,
        max_grad_norm=0.5,
        ent_coef=0.01,
        tensorboard_log=TENSORBOARD_RUN_DIR,
        verbose=1,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps, 
        callback=TrafficCallback(),
        progress_bar=True
    )

    # Save
    final_model_path = os.path.join(CHECKPOINT_ROOT, "final_model")
    model.save(final_model_path)

    print("\n--- TRAINING COMPLETE ---")
    print(f"Saved Model  → {final_model_path}.zip")
    print(f"TensorBoard → {TENSORBOARD_RUN_DIR}")
    
    # Optional: If your plot_results supports SB3 tensorboard formatting
    plot_out = os.path.join(root_dir, "outputs", "train", RUN_NAME)
    try:
        plot_results(logdir=TENSORBOARD_RUN_DIR, output_dir=plot_out, exp_name=RUN_NAME)
    except Exception as e:
        print(f"Note: Could not run plot_results. Check if it's strictly compatible with RLlib tensorboard formatting. Error: {e}")

    vec_env.close()

def _risk_bar(value, width=10):
    """value in [0,1] where 0=dangerous, 1=safe. Returns a colored bar string."""
    filled = int((1 - value) * width)
    bar = "█" * filled + "░" * (width - filled)
    if value < 0.3:
        color = "\033[91m"   # red
    elif value < 0.6:
        color = "\033[93m"   # yellow
    else:
        color = "\033[92m"   # green
    return f"{color}{bar}\033[0m"

def _angle_arrow(sin_val, cos_val):
    import math
    angle_deg = math.degrees(math.atan2(sin_val, cos_val))
    # atan2: east=0°, north=90°, west=±180°, south=-90°
    # Shift so that east (0°) maps to index 0, going CCW
    idx = round(angle_deg / 45) % 8
    arrows = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
    return arrows[idx]

def print_neighbor_table(step_num, obs, reward, neighbors_info, terminated, truncated):
    os.system("cls" if os.name == "nt" else "clear")

    # --- Ego stats from obs vector ---
    # SB3 DummyVecEnv wraps obs in an extra array dimension: obs[0][0]
    dis_to_goal = obs[0][0]
    ego_speed = obs[0][1]
    ego_sin, ego_cos = obs[0][2], obs[0][3]
    ego_dir = _angle_arrow(ego_sin, ego_cos)

    print(f"╔{'═'*72}╗")
    print(f"║  Step {step_num:<6}   Reward: {reward:+.3f}   "
          f"{'TERMINATED' if terminated else 'TRUNCATED' if truncated else 'running  ':<12}║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  EGO   dir:{ego_dir}  speed:{ego_speed:.2f}  dist_to_goal:{dis_to_goal:.2f}   ║")
    print(f"╠══════════════════════════════════════════════════════════╣")

    if not neighbors_info:
        print(f"║  No conflicting neighbors in perception radius.          ║")
    else:
        print(f"║  {'#':<3} {'dir':<4} {'dist':>6} {'speed':>6} {'ego_d':>6} {'delta_eta':>6} {'edge':<12}║")
        print(f"║  {'─'*70}║")
        for i, n in enumerate(neighbors_info):
            direction = _angle_arrow(n['sin'], n['cos'])
            dist_norm = n['distance']
            speed_pct = n['v']
            ego_d = n['ego_dist_to_cp']
            delta_eta = n['delta_eta'] # FIX 4: Corrected key lookup 
            bar = _risk_bar(ego_d)
            edge = n['edge'][:12].ljust(12)

            print(f"║  {i+1:<3} {direction:<4} {dist_norm:>6.2f} {speed_pct:>6.2f} "
                  f" {ego_d:>6.2f} {delta_eta:>6.2f}   {edge}║")
    
    print(f"╚{'═'*72}╝")
    print()

def evaluate(checkpoint_path: str, num_iterations: int = 20):
    if not checkpoint_path.endswith('.zip'):
        checkpoint_path += '.zip'
        
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n--- EVALUATION START (Discrete - SB3) ---")
    print(f"Loaded checkpoint: {checkpoint_path}")

    eval_env = DummyVecEnv([lambda: create_flow_env({"render": True})])
    model = PPO.load(checkpoint_path, env=eval_env)

    rewards = []
    for episode in range(num_iterations):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            
            # Extract from SB3's vectorized returns
            done = dones[0]
            step_reward = reward[0]
            info = infos[0]
            total_reward += step_reward
            step += 1
            
            # Print the visual table
            neighbors = info.get("neighbors", [])
            print_neighbor_table(step, obs, step_reward, neighbors, done, False)
            
        print(f"  Episode {episode+1}: reward={total_reward:.3f}")
        rewards.append(total_reward)

    avg = sum(rewards) / max(len(rewards), 1)
    print(f"\n  Average reward: {avg:.3f}")
    print("--- EVALUATION COMPLETE ---\n")
    eval_env.close()

if __name__ == "__main__":
    if args.train:
        train()
    else:
        evaluate(checkpoint_path=args.eval)
