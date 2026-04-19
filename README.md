# Mixed-Strategy Equilibrium via Actor-Critic

An extension of [Martin & Sandholm (2022)](https://arxiv.org/abs/2211.15936) that replaces zeroth-order optimization with an **Actor-Critic framework** for computing approximate Nash equilibria in continuous-action games.

---

## Overview

The original paper computes mixed-strategy equilibria using **zeroth-order (gradient-free) optimization** — smoothed gradient estimators applied to randomized policy networks. This work replaces that optimization backbone with an **Actor-Critic architecture**, drawing inspiration from model-free deep RL (SAC / TD3), and evaluates it on two benchmark games with known analytical solutions.

### Method Comparison

|  | Martin & Sandholm (2022) | **This Work** |
|---|---|---|
| Strategy representation | Randomized policy network (noise → action) | Same (implicit generative policy) |
| Policy optimization | Zeroth-order smoothed gradient | **Actor-Critic (replicator dynamics)** |
| Value estimation | None (direct payoff queries) | **Double Q-Critic** Q(a; π) = E[u(a, a')] |
| Variance reduction | Central-difference stencil | **Double Q + conservative estimate** |
| Off-policy learning | ✗ | **Replay buffer (recent + historical mix)** |
| Gradient information | Not required | Not required (black-box payoff) |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Actor (Policy Network)              │
│   z ~ N(0, I_d)  →  [FC → Mish → FC → Mish → FC]   │
│                  →  a ∈ Action Space                  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│              Double Q-Critic                          │
│   a  →  [FC → LN → Mish → FC → LN → Mish → FC]     │
│      →  Q(a; π_opponent)   (× 2, independent)        │
│                                                       │
│   Conservative estimate:                             │
│   Q_cons = (Q1 + Q2)/2 − β |Q1 − Q2|                │
└──────────────────────────────────────────────────────┘

Training Loop:
  1. Actor samples actions → payoff oracle → replay buffer
  2. Critic (high-frequency): fit Q(a; π) via Monte Carlo targets
  3. Actor (low-frequency): maximize Q_cons (replicator dynamics)
     Loss = −(Q_cons − Q̄)  [zero-mean, prevents trivial solution]
```

**Key design choices:**
- **Mish activation** — smooth, non-monotonic, avoids dead neurons
- **LayerNorm in Critic** — stabilizes Q-function learning on non-stationary targets
- **Conservative Q (UCB-style)** — actor cannot exploit critic uncertainty
- **Mixed replay sampling** — 50% recent (tracks current policy), 50% historical (prevents forgetting)
- **Critic warmup phase** — pre-trains Q on random action pairs before actor training begins

---

## Games & Results

### 1. Colonel Blotto (2 Players, 3 Battlefields)

**Setup:** Both players allocate budget B=1 across 3 battlefields. The player who allocates more to a battlefield wins it. Payoff = number of battlefields won. No pure-strategy Nash equilibrium exists.

**Analytical equilibrium** (Gross & Wagner, 1950): Uniform distribution over the surface of a hemisphere inscribed in the simplex. Each battlefield's marginal allocation mean = 1/3.

| Metric | Single Actor | Two Independent Actors | Threshold |
|--------|:-----------:|:---------------------:|:---------:|
| Exploitability | 0.0490 | 0.0488 | < 0.05 ✅ |
| Nash Gap | 0.0023 | 0.0037 | < 0.10 ✅ |
| Payoff (theory: 1.5) | 1.4997 | — | — |
| BF marginal mean max deviation | 0.0152 | 0.0145 | — |

**Strategy distribution** converges to the expected hemispherical shape on the simplex, matching the analytical solution.

**Two-actor variant:** Each player has an independent policy network and critic — testing whether symmetry needs to be imposed or emerges naturally. Result: both variants converge to the same exploitability level; the two learned strategies have low KS divergence (BF1: 0.031, BF2: 0.041, BF3: 0.022), confirming that symmetry emerges without enforcement.

---

### 2. Visibility Game (2 Players)

**Setup:** Each player chooses x ∈ [0, 1]. Payoff to player i: distance to the next-higher point (or to 1 if i has the highest point). No pure-strategy Nash equilibrium exists.

**Analytical equilibrium** (Lotker et al., 2008):
- Density: p*(x) = 1/(1−x),  support [0, 1 − 1/e] ≈ [0, 0.632]
- Expected payoff = 1/e ≈ 0.3679

**Results:**

| Metric | Value | Reference |
|--------|-------|-----------|
| In-support Q std (indifference metric) | **0.0065** | → 0 at equilibrium |
| In-support Q mean | 0.3702 | Theory: 0.3679 |
| In-support Q range | 0.0322 | — |

**Indifference condition:** At Nash equilibrium, the Q-function must be constant over the support (no profitable deviation). The learned Q-function achieves near-perfect flatness within [0, 0.632], confirming convergence.

**Distribution shape:** The learned policy density closely tracks p*(x) = 1/(1−x), with a sharp cutoff near x = 0.632 (= 1 − 1/e), consistent with theory. A boundary spike appears near the support upper bound — a known artifact of neural approximations at discontinuities.

---

## Key Findings

1. **Actor-Critic successfully computes mixed-strategy Nash equilibria** in continuous-action games without gradient information, matching or approaching known analytical solutions.

2. **The indifference condition** (Q flat over support) provides a principled, geometry-free convergence criterion — more informative than Nash Gap alone.

3. **Double Q + conservative estimate** prevents the actor from exploiting critic errors, which is critical in the non-stationary multi-agent setting.

4. **Symmetry emerges naturally** in the two-actor variant without imposing it as a constraint (KS ≈ 0.03).

5. **Critic warmup is essential**: training without warmup leads to unstable early actor updates that derail convergence.

---

## Repository Structure

```
mixed-strategy-equilibrium-actor-critic/
├── README.md
├── notebooks/
│   ├── 01_colonel_blotto.ipynb       # Blotto game: single & two-actor variants
│   └── 02_visibility_game.ipynb      # Visibility game with indifference check
├── results/
│   └── figures/                      # Training curves, strategy distributions
├── reference/
│   └── martin2022.pdf                # Original paper (Martin & Sandholm, 2022)
└── requirements.txt
```

---

## Requirements

```
torch>=2.0
numpy
scipy
matplotlib
```

---

## Reference

Martin, C., & Sandholm, T. (2022). *Finding mixed-strategy equilibria of continuous-action games without gradients using randomized policy networks.* arXiv:2211.15936.
