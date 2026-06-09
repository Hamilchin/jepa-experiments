# Pixel JEPA World Models on OGBench: Isolating the Bottleneck

### What we found, the experiments we ran to challenge ourselves, and what actually helps

*2026-06-07 · keyword: driftwood · follows the [2026-06 report](../2026-06_ogbench-lewm/)*

---

## TL;DR (main findings)

1. **Low-level control — not the loss, world model, or cost — is the wall on OGBench antmaze/cube.** On the *same maze layout, same official navigate data, same world-model + CEM-MPC pipeline*, swapping the **ant** (8-DOF locomotion) for a **point mass** (2-D control) takes success from **0% → up to 63%**. We isolated this by re-rendering the official `pointmaze-medium-navigate-v0` state trajectories to pixels.
2. **No planning strategy rescues antmaze.** Greedy, longer horizon, frequent replan, and subgoal/graph-search over the quasimetric *all* stay at **0% success** (progress 8–14% vs a **random floor of 8.9%**). This **falsified our own hypothesis** that short-horizon/greedy planning would unlock it.
3. **On the solvable env, inverse-dynamics is the best loss (63%), not the quasimetric (47%)** — even though the **quasimetric has the best geodesic geometry** (ρ0.97). So **"better geodesic geometry ⇏ better planning."** We report this rather than the tidier story we expected.
4. **Two negative results, reported plainly:** our subgoal/graph-search planner *hurt* in both envs; and SIGReg's static latent-distance looks like **noise** (ρ≈0) yet it still reaches 37% on point-maze — an inconsistency we flag rather than explain away.

### Final success / progress (n=125 = 25 episodes × 5 official tasks; `rh1` planner)

| Environment | control | random | SIGReg | VICReg | contrastive | quasimetric | **inv-dynamics** |
|---|---|---|---|---|---|---|---|
| visual antmaze-medium | 8-DOF locomotion | 0% / 8.9% | 0% / 12.9% | 0% | 0% | 0% / 14.1%* | 0% |
| visual cube-single | 7-DOF arm+grasp | 0% / 0% | 0% / 0% | 0% | 0% | 0% / 0% | 0% |
| **visual point-maze-medium** | **2-D point** | 0% / 13.4% | 36.8% / 58.6% | 21.6% / 45.3% | 19.2% / 46.9% | 47.2% / 69.4% | **63.2% / 83.8%** |

*\*antmaze "14.1%" is the best of a full planner sweep; all configs are 0% success. Antmaze rows for vicreg/contrastive/invdyn are 0% (prior report).*

![Results](figures/fig0_results.png)

---

## 1. Setup (ground-up)

**LeWM** = JEPA world model: an **encoder** maps a 64×64 image → 192-d latent `z`; an autoregressive **predictor** maps `(z_t, action) → ẑ_{t+1}`; planning is **CEM-MPC** — sample action sequences, roll the predictor forward, execute the one whose predicted final latent is **closest to the goal latent**. *The planner's cost is latent distance to the goal.* We evaluate under OGBench's **official protocol** (5 tasks, image goal, full 1000-step horizon, OGBench's success test). Because success was 0% on the hard envs, we also report **closest-approach progress** (1.0 = reached). The five losses (SIGReg, VICReg, inverse-dynamics, contrastive, quasimetric) are explained from scratch in the [prior report](../2026-06_ogbench-lewm/).

**Where we started:** the quasimetric LeWM has near-perfect latent geometry (ρ_geo = 0.97, [fig 2](figures/fig2_antmaze_quasimetric_rho097.png)) yet 0% antmaze success. *Why, and can we fix it?*

---

## 2. Experiments that challenged our assumptions

We tried to **falsify**, not confirm.

**(a) "Greedy/short-horizon planning will unlock it."** *Falsified.* Planner sweep (no retraining), all **0% success**:

| planner | greedy h1 | h3 rh1 | baseline h5 rh5 | h10 rh10 | rh1 (replan every step) | subgoal graph-search |
|---|---|---|---|---|---|---|
| antmaze progress | 9.6% | 10.5% | 12.9% | 12.5% | **14.1%** | 8.2% |

Shorter horizon *hurts*; the subgoal planner we built to chain reachable subgoals was the *worst* (8.2%, below random).

**(b) "Is ~12% progress even better than random?"** Random-action floor: antmaze **8.9% / 0%**. So the planner's best (14.1%) beats random by only ~1.6× and never reaches goals.

**(c) "Is the multi-step rollout the bottleneck?"** Open-loop rollout error: 1-step MSE tiny (6e-4, matches training); cost-agreement corr 0.99→0.85 (k=5)→0.21 (k=9) on in-distribution actions. Usable to ~5 steps — the baseline regime — so rollout fidelity is **not** the dominant wall. *(Caveat: measured on dataset actions; CEM's OOD actions likely worse, unmeasured.)*

**(d) The decisive control test.** Every clue pointed at low-level control. We built a **visual point-maze** (re-rendered official navigate data; [fig 4](figures/fig4_pointmaze_render.png)) — *identical maze + data, point mass instead of ant*. Result: **0% (ant) → up to 63% (point), across all five losses** (random floor 0%). The only change is control difficulty. **This confirms control is the wall on antmaze/cube — not the loss, world model, or cost.**

---

## 3. What actually helps (and what doesn't) — point-maze loss comparison

With control out of the way, the losses rank (success): **inv-dynamics 63.2 > quasimetric 47.2 > SIGReg 36.8 > VICReg 21.6 ≈ contrastive 19.2** (random 0). All beat random — every anti-collapse/distance loss yields *real* navigation when control allows.

**Geometry vs. planning success do NOT line up cleanly** (measured ρ of static latent-distance-to-goal vs true geodesic):

| loss (point-maze) | ρ(geo) | ρ(eucl) | success |
|---|---|---|---|
| inv-dynamics | 0.91 | 0.96 | **63.2%** |
| quasimetric | **0.94** | 0.92 | 47.2% |
| SIGReg | **−0.04** | 0.00 | 36.8% |

- The **quasimetric has the best geodesic geometry** ([fig 3](figures/fig3_pointmaze_quasimetric_rho094.png)) but **not** the best planning. Its learned MRN distance is a more complex cost than raw `‖z−z_g‖`; the extra fidelity didn't convert to more success here.
- **Inverse-dynamics** wins planning with a euclidean-leaning position code ([fig 6](figures/fig6_pointmaze_invdyn_geometry.png), ρ_eucl 0.96): it forces the latent to encode the *controllable* state (here, position), which is exactly the planner's signal.
- **SIGReg is the anomaly:** its static latent-distance is ~noise (ρ≈0, [fig 7](figures/fig7_pointmaze_sigreg_geometry.png)) yet it reaches 37%. We do **not** have a clean explanation — possibly the single-anchor, canonical-pose probe is unrepresentative of what the planner sees, so SIGReg's latent may carry usable position info the static probe missed. Flagged, not resolved.
- **Subgoal/graph-search planner: negative result** in both envs (antmaze 8.2% < baseline 12.9%; point-maze 10.4% < plain 47.2%). As implemented, chaining subgoals over the quasimetric *hurts*; plain receding-horizon MPC is better. We are not claiming it as a useful technique.

---

## 4. The methods, and an honest read on the "final" one

All five are anti-collapse/representation losses added to LeWM; details + update-step examples in the prior report. Two are most relevant here:

- **Inverse-dynamics** (empirical winner for navigation): a head predicts the action from `(z_t, z_{t+1})`; minimizing `‖head(z_t,z_{t+1}) − a_t‖²` forces the latent to keep **action-controllable** structure. On a point-maze the controllable variable *is* position, so the latent becomes a clean position code and raw `‖z−z_g‖` is a good planning cost → best success (63%).
- **Quasimetric (MRN + QRL)** (best *geometry*): an asymmetric distance head trained so `d(s,s')≤1` per transition and random pairs are pushed apart — the local rule + triangle inequality force `d` to **count shortest-path steps**, giving ρ0.94–0.97 with true geodesic distance. Best geometry, but not the best planner here.

**The honest conclusion:** there is no single "loss that solves OGBench." The decisive variable is **low-level control**. When control is easy (point-maze) every loss navigates and inverse-dynamics is best; when control is hard (ant locomotion, cube grasp) **no loss or planner reaches goals**, because torque-space CEM-MPC cannot realize the required motor skills.

---

## 5. What this means / next steps
- **The world model and the losses are not the limiting factor on hard OGBench envs.** The single change the evidence says is needed is a **learned low-level controller** (a goal-/subgoal-conditioned policy, HIQL-style) replacing torque-space CEM-MPC. A learned distance head (quasimetric/inv-dyn) is the right *high-level*; it needs a competent *low-level*.
- **Point-maze is the clean loss-iteration testbed** going forward (control is not a confound). Inverse-dynamics is the current best; understanding *why* SIGReg's near-noise geometry still navigates is an open thread.
- Negative results to keep in mind: our subgoal planner hurt; greedy planning hurt.

## Files
- `README.md`, `figures/`. Code: `distance_heads.py` (MRN+QRL, contrastive, inv-dyn), `train.py` (`loss.reg_type`), `eval_ogbench.py` / `eval_ogbench_subgoal.py`, `scripts/data/render_states_to_visual.py` (state→pixels), `scripts/probes/{rollout_error,random_floor,maze_one_heatmap}.py` (the assumption-challenging probes). *(Code is in the working tree on `main`; not yet committed.)*
