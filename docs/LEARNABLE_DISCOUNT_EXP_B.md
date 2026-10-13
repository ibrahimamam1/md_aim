# Learnable Discount Module & Architecture: Single-Objective Experiment B

This document details the **Learnable Discount Module** and the neural network architecture of the **Learnable Single Discount Net** ($\gamma_\phi(s)$) implemented in the MD-AIM framework. Per specification, this document focuses strictly on the **single-objective formulation evaluated in Experiment B**, contrasting it where relevant with baseline discounting and rule-based discounting.

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Credit-Assignment Dilemma in Autonomous Intersections
In unsignalized autonomous intersection management (AIM), an RL-controlled ego vehicle must cross an unsignalized 4-way junction while negotiating right-of-way with non-communicating background traffic (IDM vehicles). The agent receives a continuous observation of vehicle kinematics and outputs a continuous longitudinal acceleration $a_t \in [-4.5, 2.6] \text{ m/s}^2$ every $\Delta t = 0.25\text{ s}$.

In standard reinforcement learning, return estimation relies on a fixed scalar discount factor $\gamma_0 \in (0, 1)$:
$$R_t = \sum_{k=0}^{\infty} \gamma_0^k r_{t+k}$$

This static formulation enforces an invariant effective planning horizon:
$$H_{\text{eff}} = \frac{1}{1 - \gamma_0} \text{ steps} \quad \left(\tau_{\text{eff}} = \frac{\Delta t}{1 - \gamma_0} \text{ seconds}\right)$$

This creates a fundamental conflict:
- **Long Horizon ($\gamma_0 = 0.99 \implies H_{\text{eff}} = 100 \text{ steps} = 25.0\text{ s}$):** Necessary for progress shaping, traffic flow maintenance, and anticipating distant terminal goal rewards ($+20.0$). However, during sudden, imminent conflict situations (small Time-to-Collision or critical arrival gap $|\Delta \eta| \to 0$), credit assignment over 100 steps diffuses immediate collision danger signals, causing delayed emergency braking.
- **Short Horizon ($\gamma_0 \le 0.90 \implies H_{\text{eff}} \le 10 \text{ steps} = 2.5\text{ s}$):** Highly responsive to imminent safety penalties, but induces severe myopia—the agent stalls indefinitely at the intersection approach line because long-term traversal rewards are extinguished.

### 1.2 Experiment B Formulation: Learnable Single-Discounting
**Experiment B (`exp_b`)** resolves this trade-off within a **single-objective RL framework** (single critic $V(s) \in \mathbb{R}$, unified scalar reward $r_t \in \mathbb{R}$). 

Instead of fixing $\gamma$ or relying on rigid heuristic rules (Experiment A), Experiment B introduces a dedicated neural module—the **`LearnableSingleDiscountNet`** ($\gamma_\phi(s)$)—which dynamically adjusts the agent's discount factor as a function of the local multi-vehicle interaction state:
$$\gamma_t = \gamma_\phi(s_t), \quad \gamma_{\min} \le \gamma_\phi(s_t) \le \gamma_{\max}$$

The agent preserves a long horizon ($\gamma_0 = 0.99$) during nominal cruising and approach, but autonomously contracts the discount horizon toward a calibrated danger floor $\gamma_{\min}$ when conflict risk is detected.

---

## 2. System Architecture & Information Flow

The interaction between the environment, the policy/critic, the rollout buffer, and the learnable discount network is shown below:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SUMO Simulation Environment                     │
│                  (AlphaEnv_MO_SD in "baseline" mode)                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (s_t, r_t, d_t, info)
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │                Rollout Collection (Worker)               │
       │                                                          │
       │  1. Observation s_t (34-dim: ego + 5 neighbors + mask)   │
       │  2. Policy: a_t, V(s_t), log π(a_t|s_t)                   │
       │  3. Discount Module:                                     │
       │     γ_t, ρ_pred = LearnableSingleDiscountNet(s_t)        │
       │  4. Extract Target Risk ρ_target from info["conflict"]   │
       └────────────────────────────┬─────────────────────────────┘
                                    │ transitions {s_t, a_t, r_t, V_t, γ_t, ρ_target}
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │             MultiObjectiveRolloutBuffer                  │
       │                   (critic_dim = 1)                       │
       │                                                          │
       │  Dynamic Single-Stream GAE:                              │
       │  δ_t = r_t + γ_t · V(s_{t+1}) · (1 - d_{t+1}) - V(s_t)   │
       │  A_t = δ_t + γ_t · λ · (1 - d_{t+1}) · A_{t+1}           │
       │  R_t = A_t + V(s_t)                                      │
       └────────────────────────────┬─────────────────────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
               ▼                                         ▼
┌──────────────────────────────┐       ┌─────────────────────────────────┐
│        PPO Optimizer         │       │     Discount Net Optimizer      │
│  (Policy π_θ & Critic V_ψ)   │       │          (Parameters φ)         │
│                              │       │                                 │
│  - Clipped Surrogate L^CLIP  │       │  L_total = L_reg + L_risk       │
│  - Single Critic MSE L^VF    │       │  - Anti-cheating L2: ||γ - γ_0||│
│  - Entropy Bonus L^ent       │       │  - Smooth L1 on target risk ρ   │
└──────────────────────────────┘       └─────────────────────────────────┘
```

---

## 3. Network Architecture: `LearnableSingleDiscountNet`

The `LearnableSingleDiscountNet` is implemented in [`src/models/mo_sd_models.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_models.py#L342-L464).

### 3.1 Architectural Decoupling
The discount network is built with a completely independent feature trunk (`CrossAttentionTrunk`), isolated from the actor-critic network. 

**Design Rationale:** If the discount network shared feature layers with the policy or value networks, gradients from the discount loss would interfere with policy feature representations (and vice versa), leading to representation collapse and destabilized value targets. Architectural isolation guarantees that the discount net operates strictly as an adaptive horizon provider.

### 3.2 Input Observation Space (34 Dimensions)
The network consumes the 34-dimensional continuous observation vector produced by [`AlphaEnv_MO_SD`](file:///home/rgb/Desktop/md_aim/src/envs/alpha_env_mo_sd.py):

| Slice | Features | Dimensionality | Description |
|---|---|:---:|---|
| `obs[0:4]` | $S_{\text{ego}}$ | 4 | $[d_{\text{goal}}, v_{\text{ego}}, \sin\theta_{\text{ego}}, \cos\theta_{\text{ego}}]$ (distance to goal in m, speed in m/s, heading) |
| `obs[4:29]` | $S_{\text{nbr}}$ | 25 | 5 neighbors $\times$ 5 features: $[d^{\text{ego}}_{\text{cp}}, d^{\text{other}}_{\text{cp}}, v_{\text{other}}, \sin\theta_{\text{other}}, \cos\theta_{\text{other}}]$ sorted by physical distance |
| `obs[29:34]` | $M_{\text{nbr}}$ | 5 | Binary validity mask: $1.0$ if vehicle slot is occupied, $0.0$ if padded |

### 3.3 Layer-by-Layer Architecture Diagram

```
Observations obs ∈ R^34
  ├── obs[:, 0:4]   ── Ego Features (4)
  ├── obs[:, 4:29]  ── Neighbor Features (5 × 5)
  └── obs[:, 29:34] ── Neighbor Mask (5)
          │
          ├── [Ego Features] ──> Linear(4 → 64) ──> LayerNorm(64) ──> ReLU ──> ego_embed (B, 1, 64)
          │
          ├── [Neighbor Features] ──> Linear(5 → 64) ──> LayerNorm(64) ──> ReLU ──> neighbor_embeds (B, 5, 64)
          │
          ├── [Neighbor Mask] ──> key_padding_mask = (mask < 0.5) (B, 5)
          │
          ▼
    MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
      - Query : ego_embed.unsqueeze(1)    [B, 1, 64]
      - Key   : neighbor_embeds           [B, 5, 64]
      - Value : neighbor_embeds           [B, 5, 64]
      - Mask  : safe_mask (handles all-padding safely)
          │
          ▼
    attn_output ∈ R^(B, 64)  (Zeroed out if all neighbors masked)
          │
          ▼
    Context Concatenation: concat([ego_embed, attn_output]) ∈ R^(B, 128)
          │
          ▼
    LayerNorm(128) ──> Linear(128 → 128) ──> ReLU ──> feat ∈ R^(B, 128)
          │
          ▼
    Structural Risk Head: Linear(128 → 1) ──> Sigmoid
          │
          ▼
    Predicted Conflict Risk: ρ_φ(s) ∈ [0, 1]
          │
          ▼
    Structural Discount Prior:
    γ_φ(s) = γ_max - (γ_max - γ_min) · ρ_φ(s)
          │
          ▼
    Output: γ_φ(s) ∈ [γ_min, γ_max],  ρ_φ(s) ∈ [0, 1]
```

### 3.4 Cross-Attention Trunk Details
Implemented in `CrossAttentionTrunk` ([`src/models/mo_sd_models.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_models.py#L122-L201)):
1. **Ego Encoder:**
   $$\mathbf{e}_{\text{ego}} = \text{ReLU}\left(\text{LayerNorm}(\mathbf{W}_e \mathbf{s}_{\text{ego}} + \mathbf{b}_e)\right), \quad \mathbf{W}_e \in \mathbb{R}^{64 \times 4}$$
2. **Neighbor Encoder:**
   $$\mathbf{e}_{n, i} = \text{ReLU}\left(\text{LayerNorm}(\mathbf{W}_n \mathbf{s}_{n, i} + \mathbf{b}_n)\right), \quad \mathbf{W}_n \in \mathbb{R}^{64 \times 5}, \quad i \in \{1, \dots, 5\}$$
3. **Multi-Head Cross-Attention:**
   Two attention heads ($d_k = 32$) compute the relevance of surrounding vehicles to ego's path crossing:
   $$\mathbf{q} = \mathbf{e}_{\text{ego}} \mathbf{W}^Q, \quad \mathbf{k}_i = \mathbf{e}_{n, i} \mathbf{W}^K, \quad \mathbf{v}_i = \mathbf{e}_{n, i} \mathbf{W}^V$$
   $$\alpha_i = \frac{\exp\left(\frac{\mathbf{q} \mathbf{k}_i^\top}{\sqrt{d_k}}\right) \cdot \mathbb{I}[m_i = 1]}{\sum_{j=1}^5 \exp\left(\frac{\mathbf{q} \mathbf{k}_j^\top}{\sqrt{d_k}}\right) \cdot \mathbb{I}[m_j = 1]}$$
   $$\mathbf{c}_{\text{attn}} = \sum_{i=1}^5 \alpha_i \mathbf{v}_i$$
   *Empty Scenario Guard:* If an episode has no surrounding vehicles ($\sum m_i = 0$), `all_masked` evaluates to true, preventing PyTorch NaN propagation, and $\mathbf{c}_{\text{attn}}$ is explicitly set to $\mathbf{0}$.
4. **Context Fusion & Latent Projection:**
   $$\mathbf{h} = \text{LayerNorm}\left([\mathbf{e}_{\text{ego}} \,\|\, \mathbf{c}_{\text{attn}}]\right) \in \mathbb{R}^{128}$$
   $$\mathbf{z} = \text{ReLU}\left(\mathbf{W}_p \mathbf{h} + \mathbf{b}_p\right) \in \mathbb{R}^{128}$$

### 3.5 Structural Risk Prior vs Direct Discount Head
The network supports two output modes governed by the boolean flag `use_structural_prior` (set to `True` in Experiment B):

#### Option A: Structural Risk Prior Mode (`use_structural_prior = True`, Default)
The network outputs a conflict risk logit via a linear layer `risk_head`:
$$\rho_\phi(s) = \sigma\left(\mathbf{W}_{\text{risk}} \mathbf{z} + b_{\text{risk}}\right) \in [0, 1]$$
The scalar discount factor is then strictly constrained via the linear affine mapping:
$$\gamma_\phi(s) = \gamma_{\max} - (\gamma_{\max} - \gamma_{\min}) \cdot \rho_\phi(s)$$

**Why this matters (Preventing "Reward Cheating"):**
In an unconstrained setup, an RL agent can "cheat" by predicting $\gamma \to 0$ whenever an impending crash is inevitable, collapsing future value to 0 and hiding the collision penalty ($-15.0$ / $-20.0$). The structural prior forces an invariant monotonicity: **high risk ($\rho \to 1$) contracts $\gamma$ to $\gamma_{\min}$, whereas low risk ($\rho \to 0$) forces $\gamma \to \gamma_{\max}$**. It is mathematically impossible for the network to invert this relationship or output an out-of-bounds discount.

#### Option B: Direct Discount Head Mode (`use_structural_prior = False`)
A direct linear head produces a raw discount logit:
$$\gamma_\phi(s) = \gamma_{\min} + (\gamma_{\max} - \gamma_{\min}) \cdot \sigma\left(\mathbf{W}_{\text{disc}} \mathbf{z} + b_{\text{disc}}\right)$$
The risk is then inversely inferred:
$$\rho_\phi(s) = 1.0 - \frac{\gamma_\phi(s) - \gamma_{\min}}{\max(\gamma_{\max} - \gamma_{\min}, 10^{-6})}$$

---

## 4. Mathematical Formulation & Optimization

The discount network is optimized using a dual-objective loss function computed in `compute_loss` ([`src/models/mo_sd_models.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_models.py#L432-L463)):
$$\mathcal{L}_{\text{total}}(\phi) = \mathcal{L}_{\text{reg}}(\phi) + \mathcal{L}_{\text{risk}}(\phi)$$

### 4.1 Anti-Cheating Regularization Loss ($\mathcal{L}_{\text{reg}}$)
To prevent the network from needlessly drifting into myopic regimes in ambiguous states, an $L_2$ regularization penalty anchors predictions to the baseline nominal discount $\gamma_0 = 0.99$:
$$\mathcal{L}_{\text{reg}}(\phi) = \lambda_{\text{reg}} \cdot \frac{1}{B} \sum_{i=1}^B \left(\gamma_\phi(s_i) - \gamma_0\right)^2$$
Where $\lambda_{\text{reg}} = 0.01$ by default.

### 4.2 Auxiliary Conflict Risk Supervision Loss ($\mathcal{L}_{\text{risk}}$)
To accelerate convergence and align the cross-attention layers with physical conflict geometry, the predicted risk $\rho_\phi(s)$ is supervised against continuous ground-truth conflict targets $\rho_{\text{target}}(s) \in [0, 1]$ generated by the simulator:
$$\mathcal{L}_{\text{risk}}(\phi) = \lambda_{\text{risk}} \cdot \frac{1}{B} \sum_{i=1}^B \text{SmoothL1}\left(\rho_\phi(s_i) - \rho_{\text{target}}(s_i)\right)$$
Where $\lambda_{\text{risk}} = 1.0$, and $\text{SmoothL1}$ (Huber loss with $\beta = 1.0$) prevents gradient explosion from extreme outliers:
$$\text{SmoothL1}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

### 4.3 Ground-Truth Kinematic Risk Definition
During environment execution ([`compute_conflict_features`](file:///home/rgb/Desktop/md_aim/src/envs/alpha_env_mo_sd.py#L574-L635)), the environment evaluates:
1. **Time-to-Collision ($TTC$):**
   $$TTC_{\text{est}} = \begin{cases} \min(t_{\text{cp}}^{\text{ego}}, t_{\text{cp}}^{\text{other}}) & \text{if } |t_{\text{cp}}^{\text{ego}} - t_{\text{cp}}^{\text{other}}| < 2.0\text{ s} \\ \frac{d_{\text{dist}}}{\max(v_{\text{ego}} - v_{\text{other}}, 0.5)} & \text{if } v_{\text{ego}} > v_{\text{other}} \\ \infty & \text{otherwise} \end{cases}$$
2. **Kinematic Proximity Risk Components:**
   $$TTC_{\text{risk}} = \max\left(0, 1.0 - \frac{\min TTC}{TTC_{\text{thresh}}}\right), \quad TTC_{\text{thresh}} = 3.0\text{ s}$$
   $$gap_{\text{risk}} = \max\left(0, 1.0 - \frac{\min d_{\text{dist}}}{d_{\text{danger}}}\right), \quad d_{\text{danger}} = 20.0\text{ m}$$
3. **Composite Continuous Target Risk:**
   $$\rho_{\text{target}}(s) = \text{clip}\left(\max(TTC_{\text{risk}}, gap_{\text{risk}}), 0.0, 1.0\right)$$

---

## 5. Single-Objective Integration into MOSDPPO & GAE

In Experiment B, the RL policy and critic optimize a **single objective**.

### 5.1 Scalar Reward Formulation
In Experiment B (`env_mode = "baseline"`), the environment computes a unified scalar reward:
$$r_t = \Delta p_t + r_{\text{wait}} \cdot \mathbb{I}[v_{\text{ego}} < 2.0\text{ m/s}] + \lambda_{\text{gap}} \sum_{i \in \text{nbrs}} -e^{-\frac{|\Delta \eta_i|}{0.5}} + r_{\text{term}}$$
Where:
- $\Delta p_t = \text{progress}_t - \text{progress}_{t-1}$ is normalized path progress.
- $r_{\text{wait}} = -0.01$ per step when stalled.
- $\lambda_{\text{gap}} = 0.25$ weights the raw arrival-time gap penalty $|\Delta \eta_i| < 2.0\text{ s}$.
- $r_{\text{term}} = +20.0$ for goal arrival, $-15.0$ for collision.

### 5.2 Dynamic State-Dependent GAE Recursion
During rollout collection ([`mo_sd_ppo.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_ppo.py#L148-L192)), transitions store the per-step discount factor $\gamma_t = \gamma_\phi(s_t)$. 

At the end of the rollout, generalized advantage estimation is computed backward along each trajectory ($t = T-1, \dots, 0$) using the **step-varying discount** $\gamma_t$:
$$\delta_t = r_t + \gamma_t \cdot V(s_{t+1}) \cdot (1 - d_{t+1}) - V(s_t)$$
$$A_t = \delta_t + \gamma_t \cdot \lambda_{\text{GAE}} \cdot (1 - d_{t+1}) \cdot A_{t+1}$$
$$R_t = A_t + V(s_t)$$
Where:
- $\lambda_{\text{GAE}} = 0.95$.
- $d_{t+1}$ is the episode done mask.
- $V(s_t)$ is the scalar value prediction from `MultiObjectiveActorCriticPolicy` (`critic_dim = 1`).

### 5.3 PPO and Discount Network Training Sequence
Every PPO update cycle operates in two phases ([`mo_sd_ppo.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_ppo.py#L472-L576)):

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: PPO Updates (n_epochs = 4, batch_size = 128)                  │
│                                                                        │
│ For each minibatch:                                                    │
│   1. Compute policy ratio: r_t(θ) = exp(log π_θ(a_t|s_t) - log π_old)  │
│   2. Surrogate Loss:                                                   │
│      L^CLIP(θ) = -E[ min(r_t · A_t, clip(r_t, 1±ε) · A_t) ]           │
│   3. Value Loss:                                                       │
│      L^VF(ψ) = E[ (V_ψ(s_t) - R_t)^2 ]                                 │
│   4. Entropy Bonus:                                                    │
│      L^ent(θ) = -E[ H[π_θ(·|s_t)] ]                                    │
│   5. Update θ, ψ via Adam (clip_grad_norm = 0.5)                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Learnable Discount Net Update (Post-PPO Step)                 │
│                                                                        │
│ 1. Collect full buffer tensors: all_obs, all_risks                     │
│ 2. Forward pass: γ_pred, ρ_pred = LearnableSingleDiscountNet(all_obs)  │
│ 3. Loss computation:                                                   │
│    L_total = 0.01 · ||γ_pred - γ_0||^2 + 1.0 · SmoothL1(ρ_pred, risks)│
│ 4. Optimizer step:                                                     │
│    optimizer.zero_grad() ──> L_total.backward() ──> optimizer.step()   │
│ 5. Record TensorBoard/Wandb metrics:                                   │
│    discount_net/loss, reg_loss, risk_loss, mean_gamma, mean_risk       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Physics of the Danger Horizon Floor ($\gamma_{s\_danger} = \gamma_{\min}$)

In Experiment B, $\gamma_{\max}$ is fixed to $\gamma_0 = 0.99$, while the lower bound $\gamma_{\min} = \gamma_{s\_danger}$ dictates the maximum allowable horizon contraction during peak conflict.

As defined in [`scripts/run_danger_horizon_sweep.sh`](file:///home/rgb/Desktop/md_aim/scripts/run_danger_horizon_sweep.sh#L9-L18), $\gamma_{\min}$ is swept across $\{0.0, 0.50, 0.70, 0.80, 0.85\}$:

| Parameter $\gamma_{\min}$ | Effective Steps $H_{\text{eff}} = \frac{1}{1 - \gamma_{\min}}$ | Physical Horizon $\tau_{\text{eff}} = H_{\text{eff}} \cdot 0.25\text{ s}$ | Physical Implication & Vehicle Dynamics |
|---|:---:|:---:|---|
| **0.00** | 1.0 step | **0.25 s** | **Extreme Myopia:** Agent sees only the current time-step transition. It becomes completely blind to collisions occurring $> 0.25\text{ s}$ ahead, failing to decelerate in time. |
| **0.50** | 2.0 steps | **0.50 s** | Sub-second reaction horizon; still shorter than human/IDM perception-reaction time ($\approx 0.8\text{ s}$). |
| **0.70** | 3.3 steps | **0.83 s** | Threshold of typical driver reaction time. Begins to exhibit emergency braking. |
| **0.80** | 5.0 steps | **1.25 s** | **Calibrated Braking Horizon:** Matches emergency braking dynamics from speed $v = 10\text{ m/s}$ at $a_{\max} = -4.5\text{ m/s}^2$ ($t_{\text{stop}} \approx 2.2\text{ s}$; deceleration initiated well in advance). |
| **0.85** | 6.7 steps | **1.67 s** | **Early Decelerative Margin:** Provides smooth deceleration before entering conflict points while dampening long-term reward noise. |

---

## 7. Comparison: Experiment B vs Other Experimental Modes

To clarify Experiment B's scope within the MD-AIM framework, the table below highlights architectural differences across modes:

| Feature / Dimension | Baseline | Experiment A | **Experiment B (This Document)** | Experiment C (formerly Exp B) | Experiment D (formerly Exp C) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Objective Type** | Single | Single | **Single** | Multi-Objective (Dual) | Multi-Objective (Dual) |
| **Critic Heads** | 1 head ($V$) | 1 head ($V$) | **1 head ($V$)** | 2 heads ($V_l, V_s$) | 2 heads ($V_l, V_s$) |
| **Reward Representation** | Scalar | Scalar | **Scalar** | Vector $[r_l, r_s]$ | Vector $[r_l, r_s]$ |
| **Discount Mechanism** | Fixed $\gamma_0$ | Rule-based $\gamma(s)$ | **Learned $\gamma_\phi(s)$** | Rule-based $[\gamma_l, \gamma_s(s)]$ | Learned $[\gamma_l(s), \gamma_s(s)]$ |
| **Discount Network** | None | None | **`LearnableSingleDiscountNet`** | None | `LearnableDiscountNet` |
| **Trunk Isolation** | N/A | N/A | **Independent Cross-Attention** | N/A | Independent Cross-Attention |
| **Structural Prior** | N/A | Heuristic | **$\gamma_{\max} - \Delta\gamma \cdot \rho_\phi(s)$** | Heuristic | $\gamma_s = \gamma_{\max} - \Delta\gamma \cdot \rho_\phi(s)$ |
| **Auxiliary Supervision** | None | None | **Smooth L1 on $\rho_{\text{target}}$** | None | Smooth L1 on $\rho_{\text{target}}$ |
| **Anti-Cheating Loss** | None | None | **$\lambda \|\gamma - \gamma_0\|^2$** | None | $\lambda \sum \|\gamma_i - \gamma_{0, i}\|^2$ |

---

## 8. Source Code Reference Guide

| Component | File Path | Key Classes / Functions | Primary Lines |
|---|---|---|---|
| **Discount Net Definition** | [`src/models/mo_sd_models.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_models.py) | `LearnableSingleDiscountNet`, `CrossAttentionTrunk` | Lines 122–201, 342–464 |
| **PPO Algorithm & GAE** | [`src/models/mo_sd_ppo.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_ppo.py) | `MOSDPPO`, `MultiObjectiveRolloutBuffer` | Lines 33–220, 291–299, 353–358, 561–576 |
| **Actor-Critic Policy** | [`src/models/mo_sd_models.py`](file:///home/rgb/Desktop/md_aim/src/models/mo_sd_models.py) | `MultiObjectiveActorCriticPolicy` (`critic_dim=1`) | Lines 38–120 |
| **Environment & Kinematic Risk** | [`src/envs/alpha_env_mo_sd.py`](file:///home/rgb/Desktop/md_aim/src/envs/alpha_env_mo_sd.py) | `compute_conflict_features`, `compute_reward` | Lines 574–635, 744–773 |
| **Training Entry Point** | [`src/configs/train_mo_sd.py`](file:///home/rgb/Desktop/md_aim/src/configs/train_mo_sd.py) | `parse_args`, `train_run` (`--mode exp_b`) | Lines 9, 22, 267, 421–444 |
| **Danger Horizon Sweep** | [`scripts/run_danger_horizon_sweep.sh`](file:///home/rgb/Desktop/md_aim/scripts/run_danger_horizon_sweep.sh) | Sweep runner (`MODE=exp_b`) | Lines 12–28, 44–99 |

