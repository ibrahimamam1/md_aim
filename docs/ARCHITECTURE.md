# MD-AIM Architecture

Single-agent reinforcement learning for autonomous **intersection crossing** with SUMO/Flow.
One RL-controlled "ego" vehicle (the agent) approaches a 4-way, right-before-left intersection
from the west and must cross to the east while interacting with non-RL (IDM) traffic from the
other approaches. The agent selects an **acceleration** every RL step; there are no traffic
lights in the loop — the vehicle itself decides when it is safe to go.

The training algorithm is a custom **dual-discount PPO** ("Multi-Discount PPO", the `md_aim`
wandb project) built on Stable-Baselines3 2.0.0: a standard PPO whose value function is split
into a **short-term head** (dense, per-step shaping) and a **long-term head** (terminal
goal/crash outcomes), each with its own discount factor and its own GAE stream.

---

## 1. System overview

```
                    ┌──────────────────────────────────────────────┐
                    │                SUMO (TraCI)                   │
                    │  4-way intersection, IDM background traffic  │
                    └───────────────────▲──────────────────────────┘
                                        │ kernel API (flow)
                    ┌───────────────────┴──────────────────────────┐
                    │            Flow environment (gym)            │
                    │  Env_N ─> AlphaEnv_v01 ─> variants           │
                    └───────────────────▲──────────────────────────┘
                                        │ obs / reward / done
                    ┌───────────────────┴──────────────────────────┐
                    │           SB3 PPO machinery                  │
                    │  DualHeadPPO (SubprocVecEnv, 8 workers)      │
                    │   └─ DualHeadActorCriticPolicy               │
                    │        └─ AttentionFeatureExtractor (opt.)   │
                    │   └─ MultiDiscountRolloutBuffer              │
                    │        (dual reward streams, dual GAE)       │
                    └──────────────────────────────────────────────┘
```

- **Simulator:** SUMO via flow's TraCI kernel, `sim_step = 0.25 s`, episode `horizon = 180 s`
  (720 sim steps), 1 sim step per RL action.
- **Training net:** `networks/100m_skewed_right_before_left.net.xml`
- **Eval net:** `networks/100m_right_before_left.net.xml` (different internal-edge numbering —
  the env parses junction geometry from whichever net is loaded, see §2.4).
- **Files:**
  - `src/envs/base_env_single.py` — `Env_N`, the flow/gym base class.
  - `src/envs/alpha_env_v01.py` — `AlphaEnv_v01` (continuous, heuristic obs).
  - `src/envs/alpha_env_v01_discrete.py` — discrete-action variants (`_Discrete`, `_3`, `_5`, `_10`).
  - `src/envs/alpha_env_v01_attention_continous.py` — attention obs variant.
  - `src/envs/alpha_env_v01_attention_discrete.py` — attention + discrete variants.
  - `src/models/multi_discount_ppo.py` — `DualHeadPPO`, `DualHeadActorCriticPolicy`,
    `MultiDiscountRolloutBuffer`.
  - `src/models/attention_model.py` — `AttentionFeatureExtractor`.
  - `src/configs/v0_1_single_agent.py` — training entry point, hyperparameters, callbacks.
  - `src/test/v0_1_evaluate*.py`, `src/test/plot_eval_results.py` — evaluation + plotting.

---

## 2. Environment

### 2.1 Class hierarchy

```
Env_N (gym.Env, abstract)                 # base_env_single.py
└── AlphaEnv_v01                          # alpha_env_v01.py         (continuous, heuristic obs)
    ├── AlphaEnv_v01_Discrete             # discrete 5 bins          (ACCEL_BINS)
    │    ├── AlphaEnv_v01_Discrete_3      #   3 bins
    │    ├── AlphaEnv_v01_Discrete_5      #   5 bins
    │    └── AlphaEnv_v01_Discrete_10     #  10 bins
    └── AlphaEnv_v01_Attention            # attention obs (34-dim)
         └── AlphaEnv_v01_AttentionDiscrete (+ _3/_5/_10)
```

All variants share `Env_N`'s step/reset/telemetry machinery and `AlphaEnv_v01`'s observation
builder, conflict model, and reward.

### 2.2 Observation space

**Heuristic (continuous) variants — 29-dim** `Box(-1, 1)`:

```
[ ego ]           4 features:  dis_to_goal_norm, ego_speed_norm, ego_sin, ego_cos
[ neighbor i ]    5 features each, up to max_neighbours=5, sorted by distance:
                   ego_dist_to_cp_norm, other_dist_to_cp_norm, other_speed_norm, other_sin, other_cos
[ padding ]       missing neighbors padded with [1.0, 0.0, 1.0, 0.0, 0.0]
```

where `dis_to_goal_norm = clip((route_len − dist) / route_len, −1, 1)`,
`ego_speed_norm = clip(v / v_max, −1, 1)`, and the sin/cos are the heading converted from
SUMO's compass convention (`θ_rad = (−heading + 90)·π/180`).

**Attention variants — 34-dim**: the same 29 features plus a 5-dim **validity mask**
(`1.0` for a real neighbor, `0.0` for padding), and a larger `perception_radius = 100 m`
(vs 50 m for the heuristic env).

The per-neighbor "distance to conflict point" (`dist_to_cp`) is computed geometrically:
each vehicle's future path is stitched into a continuous Shapely `LineString` (see §2.4),
paths are intersected, and each vehicle's distance to the intersection point is measured
along its own path. Overlapping paths (car-following / merging) are handled by the
"same-path tolerance" branch in `_get_local_observation`.

**Conflict predicate** (`_is_conflicting`): a neighbor is considered only if it is (a) on the
same edge and ahead of ego, or (b) its `(source, destination)` route pair appears in the
static `conflict_map` for ego's route pair (§2.5). Vehicles already past the intersection
(on `E#X-*` outflow edges) are ignored.

### 2.3 Action space & reward

**Action** — a normalized acceleration in `[−1, 1]`, denormalized to `[−max_decel, max_accel]`
=`[−4.5, 2.6] m/s²` and applied to the ego vehicle. Discrete variants map the index through
`ACCEL_BINS` (e.g. `[−1, −0.5, 0, 0.5, 1]` for 5 bins).

**Reward** (identical across all variants):

| Term | Value |
|---|---|
| Goal reached (success) | **+15.0** (terminal) |
| Collision (crash) | **−10.0** (terminal) |
| Progress shaping | `+10.0 · Δprogress_norm` per step |
| Safety penalty | `Σ −exp(−10·\|d_η\|)` for each neighbor with `\|d_η\| < 0.2` |
| Time penalty | `−0.01` per step |

- `progress_norm = clip(ego_dist / route_length, 0, 1)`; `Δ` is the per-step increase.
- `d_η` is the normalized time-gap to a conflict point:
  `η_v = dist_to_cp / max(v, 0.5)`, `Δη = η_ego − η_other`, `d_η = tanh(Δη / 5) ∈ [−1, 1]`.
  `|d_η| → 0` means simultaneous arrival at the conflict point (dangerous); `→ 1` is safe.
- Terminal checks are ordered **crash → goal → departed guard**: a successful agent has
  already been removed from the network by SUMO, so the "not in `get_ids()`" guard must
  come *after* the `goal_reached` check (this ordering was a verified bug fix).

### 2.4 Junction geometry (connection maps)

Internal junction edges (`:X_n`) are numbered by netconvert and differ between the training
and evaluation nets. `Env_N._build_connection_maps()` parses the loaded net.xml's
`<connection from to via>` table at runtime to build:

- `internal_connections: (from_edge, to_edge) → internal_edge`
- `internal_to_out: internal_edge → to_edge`

`_get_vehicle_polyline()` stitches macro-edge shapes + internal-edge shapes into one
`LineString` (with consecutive-duplicate coordinate dedup so Shapely projections never NaN).
When a vehicle is already inside the junction (on an internal edge), the **outgoing** macro
edge is looked up via `internal_to_out` — `get_route()` returns only macro edges, so
`route[0]` would be the origin, not the destination (another verified bug fix).

### 2.5 Conflict map

`AlphaEnv_v01._build_conflict_map()` hardcodes the 12 route patterns of a 4-way intersection
(N/S/E/W sources × destinations, including straights, left and right turns) and maps each
`(source, destination)` pair to the set of conflicting pairs — crossing straights, opposing
left turns, merging right turns, and the vehicle's own pattern (to track car-following leaders
through the junction). `_is_conflicting` then does a dict lookup on
`(route[0], route[-1])`.

### 2.6 Telemetry

`Env_N` accumulates per-episode telemetry (speeds, accelerations, jerks, travel/waiting time,
per-step safe gap `min|d_η|`, success/collision flags) and returns it in
`info["telemetry"]` on the terminal step. The training callback aggregates these into
episode-based `custom_metrics/*`; the eval scripts write them to per-scenario CSVs.

---

## 3. Neural network pieces

Everything below lives in `src/models/` plus SB3's own layers. SB3 2.0.0 defaults that apply:
Tanh activations, orthogonal init, Adam optimizer.

### 3.1 `DualHeadActorCriticPolicy` (multi_discount_ppo.py)

Subclasses `stable_baselines3.common.policies.ActorCriticPolicy`; the only change is
replacing the value head:

```python
self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 2)   # instead of Linear(..., 1)
self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
```

The optimizer is re-instantiated because `super()._build()` already created it against the
old 1-output head (the new `value_net` params must be in the optimizer).

### 3.2 Feature extractor / MLP trunk

**Heuristic variants** (`policy_kwargs=None`): SB3's `FlattenExtractor` (identity over the
29-dim obs) → `MlpExtractor` with the default `net_arch = [64, 64]`, normalized by SB3 to
`dict(pi=[64, 64], vf=[64, 64])` — i.e. **separate** policy and value MLPs, each
`Linear(29→64) → Tanh → Linear(64→64) → Tanh`; `latent_dim_pi = latent_dim_vf = 64`.

**Attention variants** (`policy_kwargs` in the config): `AttentionFeatureExtractor` produces a
256-dim latent → `MlpExtractor(net_arch=dict(pi=[256, 256], vf=[256, 256]))`, so each trunk is
`Linear(256→256) → Tanh → Linear(256→256) → Tanh`; `latent_dim_pi = latent_dim_vf = 256`.

### 3.3 `AttentionFeatureExtractor` (attention_model.py)

Cross-attention over neighbor vehicles:

```
ego_raw (4)      ── Linear(4→64) → LayerNorm → ReLU ──┐  (ego_embed, 64)
neighbor_raw (5×5)── Linear(5→64) → LayerNorm → ReLU ──┴─┐
mask_raw (5)     ── key_padding_mask = (mask < 0.5)     │
                                                         ▼
              nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
              query = ego_embed.unsqueeze(1), key = value = neighbor_embeds
              ── output zeroed when all neighbors masked (safety) ──
                                                         │
context = concat([ego_embed, attn_out]) (128) → LayerNorm → Linear(128→256) → ReLU
```

Output: 256-dim latent consumed by the pi/vf trunks.

### 3.4 Action heads

- **Continuous** (Box(1,)): `DiagGaussianDistribution` → `mean_actions = Linear(latent→1)`
  plus a free `log_std` parameter initialized to `0.0` (per-dim, not action-dependent).
  The mean is *not* squashed (SB3 `squash_output=False` default).
- **Discrete** (n bins): `CategoricalDistribution` → `action_logits = Linear(latent→n)`,
  probabilities via softmax.

### 3.5 Value head (the "dual head")

`value_net = Linear(latent_dim_vf → 2)`:

- output `[0]` — **short-term value** `V_short(s)`: predicts the discounted dense-shaping stream.
- output `[1]` — **long-term value** `V_long(s)`: predicts the discounted terminal-outcome
  (goal/crash) stream.

---

## 4. Rollout buffer and reward decomposition

`MultiDiscountRolloutBuffer` extends SB3's `RolloutBuffer` with **two reward streams**, two
value/return streams, and one total advantage:

- `rewards_short`, `rewards_long` — the per-step reward is split at `add()` time by *magic
  numbers*: `is_long = isclose(r, 15.0) | isclose(r, -10.0)`. Terminal goal/crash rewards go
  to the long stream, everything else (dense shaping) to the short stream.
- `values[buffer, n_envs, 2]`, `returns[buffer, n_envs, 2]` — dual-head predictions and
  targets.
- `advantages[buffer, n_envs]` — **scalar** total advantage (see §5).

This is what makes the value head trainable as two separate value functions: each stream is a
standard discounted return over its own reward channel, and the two heads learn to attribute
credit independently (short-term shaping vs. sparse terminal outcome).

---

## 5. Advantage estimation (dual GAE)

`compute_returns_and_advantage(last_values, dones)` runs two independent GAE recursions
backwards through the buffer, one per discount factor:

```
γ_short = gamma      (CLI default 0.90)
γ_long  = gamma_long (CLI default 0.99)
λ       = gae_lambda (0.95, shared by both streams)
```

For each stream `x ∈ {short, long}`:

```
δ_x(t) = r_x(t) + γ_x · V_x(s_{t+1}) · (1 − done_{t+1}) − V_x(s_t)
A_x(t) = δ_x(t) + γ_x · λ · (1 − done_{t+1}) · A_x(t+1)
R_x(t) = A_x(t) + V_x(s_t)          # stored as the value target
```

Notes:

- At the buffer's last step, `V_x(s_{t+1})` uses `last_values` (the critic bootstrap), masked
  by `dones`; elsewhere the stored `values[step+1]` is used.
- **No bootstrap on horizon truncation**: SB3 2.0.0's `PPO.collect_rollouts` adds
  `rewards[idx] += γ · terminal_value` whenever `TimeLimit.truncated` is set in an info — with
  the dual head, `predict_values` returns `[n_envs, 2]` and this broadcast corrupts the scalar
  reward (verified empirically). The training config's `DisableBootstrapCallback` sets
  `TimeLimit.truncated = False` in infos, and the custom buffer never reads the flag, so a
  truncated episode is treated as a true terminal (V = 0).
- The **total advantage** passed to the policy update is simply the sum:

```
A_total(t) = A_short(t) + A_long(t)      # scalar per transition
```

- The dual value targets are kept separate: `returns[:, :, 0]` and `returns[:, :, 1]`.
  `_get_samples()` **flattens** `values` and `returns` from `(batch, 2)` to `(batch*2,)`
  while observations/actions/advantages stay `(batch,)`. This "stacking trick" lets SB3's
  stock `F.mse_loss(returns, values_pred)` train both heads with one loss, because the
  flattened tensors align exactly (`[V_s0, V_l0, V_s1, V_l1, …]` vs `[R_s0, R_l0, …]`).

---

## 6. Loss functions

The training loop is SB3 2.0.0's stock `PPO.train` (no override); the dual-head machinery is
entirely in the buffer/policy. With `A = A_total`, per minibatch:

**Policy loss** — clipped surrogate objective:

```
ratio(θ) = exp(log π_θ(a|s) − log π_old(a|s))
L_policy  = −E[ min( A·ratio,  A·clip(ratio, 1−ε, 1+ε) ) ],   ε = 0.25 (clip_range)
```

**Value loss** — MSE on the flattened dual head:

```
L_value = E[ (V_θ(s) − R)² ]      # R = flattened [R_short, R_long] targets
```

Because `values` and `returns` are both flattened `(batch*2,)`, this single MSE is computed
over both heads simultaneously — no custom value-loss code needed.

**Entropy bonus** (to encourage exploration):

```
L_entropy = −E[ H[π_θ(·|s)] ]
```

**Total loss:**

```
L = L_policy + ent_coef · L_entropy + vf_coef · L_value
  = L_policy + 0.01 · L_entropy + 0.5 · L_value
```

Optimization: Adam (SB3 default `eps=1e-5`), gradients clipped to `max_grad_norm = 0.5`.
Learning rate: `linear_schedule_with_floor(3e-4, 1e-5)` — linear decay in `progress_remaining`
with a `1e-5` floor (instead of SB3's decay-to-zero).

---

## 7. Training configuration

From `src/configs/v0_1_single_agent.py` (`train()`):

| Setting | Value |
|---|---|
| Algorithm | `DualHeadPPO` (PPO subclass) |
| Policy | `DualHeadActorCriticPolicy` |
| Vec env | `SubprocVecEnv`, **8** workers (fork-server; each env gets a fresh SUMO port) |
| Env wrapper | `Monitor` |
| `n_steps` (per env) | 1024 → rollout = 8192 transitions |
| `batch_size` | 256 |
| `n_epochs` | 10 |
| `gamma` (short) | 0.90 (CLI `--gamma`) |
| `gamma_long` | 0.99 (CLI `--gamma_long`) |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.25 |
| `ent_coef` | 0.01 |
| `vf_coef` | 0.5 (SB3 default) |
| `max_grad_norm` | 0.5 |
| `learning_rate` | 3e-4 → 1e-5 (linear floor) |
| Total timesteps | 1,000,000 |
| Logging | TensorBoard + `wandb` (`sync_tensorboard=True`), checkpoints via `WandbCallback` |

**Callbacks:**
- `TrafficCallback` — aggregates per-**episode** telemetry at each rollout end and records
  `custom_metrics/{episodes, success_rate, collision_rate, avg_speed, avg_safe_gap}`.
- `WandbCallback(model_save_path=…)` — SB3↔wandb bridge (saves checkpoints, syncs TB).
- `DisableBootstrapCallback` — suppresses `TimeLimit.truncated` bootstrapping (§5).
  (`RemoveTruncatedWrapper` is defined in the config but currently unused.)

---

## 8. Evaluation

- `src/test/v0_1_evaluate.py` — main eval: loads a checkpoint, runs `n_sims` (default 42)
  episodes per traffic scenario (`Sc1…Sc6` rate regimes) on the eval net, writes per-scenario
  CSVs with success/collision/avg_speed/safe_gap/travel_time + time/distance/velocity/jerk
  profiles, plots a 3×3 dashboard (`plot_eval_results.py`), and optionally logs per-scenario
  aggregates to wandb (`--wandb`).
- `src/test/v0_1_evaluate_all_rl.py`, `v0_1_evaluate_deterministic.py` — fleet-wide and
  deterministic variants.
- Inference uses `model.predict(obs, deterministic=True)` against a `DummyVecEnv` driven
  manually (no `set_env()`, because training used 8 envs and SB3 would reject a count change).

---

## 9. Design notes / known quirks

1. **Magic-number reward split.** The buffer classifies terminal rewards with
   `isclose(r, 15.0) | isclose(r, -10.0)`. This is brittle: changing the reward constants,
   or any dense reward that happens to equal exactly ±15/±10, silently misroutes the stream.
2. **Padding contradiction.** The heuristic obs pads missing neighbors with
   `[1.0, 0.0, 1.0, 0.0, 0.0]` — `ego_dist_to_cp=1.0` (safe) but `other_dist_to_cp=0.0`,
   which contradicts the "1 = safe/no neighbor" convention used for `ego_dist_to_cp`.
3. **Truncation = terminal.** Suppressing `TimeLimit.truncated` means horizon-truncated
   episodes get no bootstrap; the critic's terminal value is assumed 0 at the horizon.
4. **Eval ≠ train distribution.** Training uses the skewed net; evaluation uses
   `100m_right_before_left.net.xml` with different junction geometry and traffic regimes —
   eval numbers do not measure training-time conditions exactly.
5. **Reward-order / geometry bug fixes** (verified empirically with live SUMO):
   terminal `goal_reached` must be checked before the departed-agent guard (§2.3), and
   junction polylines must use the runtime-parsed connection map, not a hardcoded dict
   (§2.4). Existing checkpoints predate these fixes, so retrain before comparing.
