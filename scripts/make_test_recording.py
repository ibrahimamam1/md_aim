"""Generate a synthetic schema-v2 episode recording for renderer testing.

Creates output/eval_mo_sd/episode_recordings/S3TEST/run_9000_coll1_succ0.json
with a head-on approach scenario: the ego drives east along the west arm and a
neighbor crosses south along the north arm; both approach the shared conflict
point near the intersection centre, with |d_eta| shrinking into the danger
window (a collision with the ego at the end).

Usage:
    python -m scripts.make_test_recording
"""

import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERCEPTION = 100.0
MAX_SPEED = 13.9          # m/s, typical SUMO urban max
ROUTE_LEN = 200.0         # m, total route length
SIM_STEP = 0.25           # s
N_STEPS = 40              # 10 s of simulation

EGO_START = (-60.0, -5.0)   # west arm, eastbound
NB_START = (5.0, 60.0)      # north arm, southbound


def _sumo_heading_deg(dx, dy):
    """SUMO heading in degrees (0 = north, 90 = east)."""
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _state(pos, vel):
    x, y = pos
    vx, vy = vel
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "speed": round(math.hypot(vx, vy), 2),
        "heading": round(_sumo_heading_deg(vx, vy), 1),
        "accel": 0.0,
    }


def make_episode():
    frames = []
    # Simple constant-velocity kinematics towards the origin area.
    ego_pos, ego_vel = list(EGO_START), [8.0, 0.0]
    nb_pos, nb_vel = list(NB_START), [0.0, -8.0]

    for step in range(N_STEPS + 1):
        t = step * SIM_STEP
        # Ego brakes late; neighbor keeps speed -> collision at final step.
        if step >= 30:
            ego_vel[0] = max(2.0, ego_vel[0] - 2.0)

        ego = _state(ego_pos, ego_vel)
        ego["id"] = "rl_0"
        ego["accel"] = -2.0 if step >= 30 else 0.0
        nb = _state(nb_pos, nb_vel)
        nb["id"] = "flow_1"
        nb["accel"] = 0.0

        # Distances travelled along each route (for d_eta realism)
        ego_d_cp = max(0.0, math.hypot(ego_pos[0], ego_pos[1]) - 3.0)
        nb_d_cp = max(0.0, math.hypot(nb_pos[0], nb_pos[1]) - 3.0)

        # Normalized features exactly as AlphaEnv_MO_SD builds them
        # d_goal = dis_to_goal / route_length (ego spawns 60 m before the CP)
        ego_dis = max(0.0, ego_pos[0] - EGO_START[0])
        d_goal_norm = max(0.0, (ROUTE_LEN - ego_dis) / ROUTE_LEN)
        ego_v_norm = ego["speed"] / MAX_SPEED
        ego_h = math.radians(90.0 - ego["heading"])
        nb_v_norm = nb["speed"] / MAX_SPEED
        nb_h = math.radians(90.0 - nb["heading"])

        ego_eta = ego_d_cp / max(ego["speed"], 0.5)
        nb_eta = nb_d_cp / max(nb["speed"], 0.5)
        d_eta_norm = math.tanh((ego_eta - nb_eta) / 5.0)

        slot = [
            round(min(1.0, ego_d_cp / PERCEPTION), 4),
            round(min(1.0, nb_d_cp / PERCEPTION), 4),
            round(nb_v_norm, 4),
            round(math.sin(nb_h), 4),
            round(math.cos(nb_h), 4),
        ]
        padding = [1.0, 0.0, 1.0, 0.0, 0.0]   # exact pad pattern of the env
        obs = [
            round(d_goal_norm, 4), round(ego_v_norm, 4),
            round(math.sin(ego_h), 4), round(math.cos(ego_h), 4),
            *slot,                               # slot 0: the real neighbor
            *(padding * 4),                      # slots 1-4: padding
            1.0, 0.0, 0.0, 0.0, 0.0,             # attention mask: slot 0 live
        ]

        terminated = step == N_STEPS
        frames.append({
            "t": round(t, 3),
            "step": step,
            "ego": ego,
            "vehicles": [nb],
            "action": -0.6 if step >= 30 else 0.35,
            "reward": 0.01 if not terminated else -15.0,
            "neighbors": ["flow_1"],
            "obs": obs,
            "neighbors_info": [{
                "veh_id": "flow_1",
                "ego_dist_to_cp_norm": slot[0],
                "other_dist_to_cp_norm": slot[1],
                "other_speed": slot[2],
                "other_sin": slot[3],
                "other_cos": slot[4],
                "d_eta": round(d_eta_norm, 4),
                "edge": "E#T-X",
                "distance": round(math.hypot(ego_pos[0] - nb_pos[0],
                                             ego_pos[1] - nb_pos[1]), 2),
            }],
            "terminated": terminated,
            "truncated": False,
        })

        # Integrate
        ego_pos[0] += ego_vel[0] * SIM_STEP
        ego_pos[1] += ego_vel[1] * SIM_STEP
        nb_pos[0] += nb_vel[0] * SIM_STEP
        nb_pos[1] += nb_vel[1] * SIM_STEP

    return {
        "schema": "md_aim_episode_recording_v2",
        "scenario_id": "S3TEST",
        "run_index": 9000,
        "collision": 1,
        "success": 0,
        "timeout": 0,
        "sim_step": SIM_STEP,
        "perception_radius": PERCEPTION,
        "max_neighbours": 5,
        "norms": {
            "max_speed": MAX_SPEED,
            "perception_radius": PERCEPTION,
            "route_length": ROUTE_LEN,
        },
        "metadata": {"synthetic": True},
        "final_info": {"mo_telemetry": {"collision": 1, "success": 0}},
        "frames": frames,
    }


def main():
    out = os.path.join(REPO, "output", "eval_mo_sd", "episode_recordings",
                       "S3TEST", "run_9000_coll1_succ0.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(make_episode(), f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
