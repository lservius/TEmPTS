# TEmPTS — Transition Estimation for Proportional Time Sequences

Python package based on Servius et al.,
*"State transition estimation in proportional data sequence with applications in immunology"*.

Original scripts: manuscript/appli_simData/mm_temptations_functions.py

---

## Installation

```bash
pip install .
# or, for development
pip install -e ".[dev]"
```

### Dependencies

| package | purpose |
|---|---|
| `numpy` | array maths |
| `pandas` | data wrangling |
| `scipy` | matrix exponential, optimisation, statistics |
| `multiprocess` | parallel multi-start optimisation |
| `progressbar2` | progress display (optional) |

---

## Conceptual overview

TEmPTS estimates a continuous-time Markov rate matrix **Q** from aggregate
proportional observations.  Given proportions p(t) observed at time points
t₀ < t₁ < … < t_T the package finds Q such that

```
p(uₗ) ≈ exp(Q · uₗ)ᵀ · p(tₗ₋₁)    where uₗ = tₗ − tₗ₋₁
```

via constrained nonlinear least-squares, with a biological prior that only
forward transitions are permitted (upper-triangular Q).

Uncertainty is quantified via a residual-resampling bootstrap that produces
both bootstrap-t and BCa confidence intervals.

---

## Quick-start example

```python
import numpy as np
import tempts

# ── 1. Reproducibility ───────────────────────────────────────────────────────
tempts.planting(42)

# ── 2. Define a true rate matrix (3 states) ──────────────────────────────────
k = 3
Q_true = np.array([
    [-0.3,  0.2,  0.1],
    [ 0.0, -0.15, 0.15],
    [ 0.0,  0.0,  0.0 ],
])

# ── 3. Simulate aggregate counts ─────────────────────────────────────────────
N0  = tempts.createN0(k, startCount=10_000, state_names=tempts.statNam(k))
sim = tempts.generateAggStates(N0, Q_true, n_generate=20, recruit=False)
print(sim)

# ── 4. Convert to proportions ────────────────────────────────────────────────
pi_hat, T = tempts.eqTin(sim.to_numpy())
u = np.ones(T - 1)          # unit time intervals

# ── 5. Build a Q template (upper-triangle structure) ─────────────────────────
Q_template = np.triu(np.ones((k, k)), k=1)
Q_template[:, -1] = 1       # last column = diagonal placeholder

n_param = int(np.count_nonzero(Q_template[:, :-1]))

# ── 6. Set bounds / constraints ───────────────────────────────────────────────
bounds      = tempts.def_bounds(n_param, k)
constraints = tempts.def_constraints(n_param, k)

# ── 7. Point estimation (single core, 30 random starts) ──────────────────────
optimiserArgs = {
    'costFunc':      tempts.calc_cost,
    'args':          (pi_hat, T, k, u, Q_template),
    'bounds':        bounds,
    'constraints':   constraints,
    'sampRange_ub':  1e-1,
    'ingressAccount': False,
}

theta_est, cost = tempts.oneCore_mc_optimiser(
    iter_samp=30,
    n_param=n_param,
    theta_generator=tempts.make_theta0,
    optimiserArgs=optimiserArgs,
    progress_bar=True,
)

Q_est = tempts.theta_to_Q(theta_est, k, Q_template)
print("\nEstimated Q:\n", Q_est)
print("True Q:\n", Q_true)
```

---

## Function reference

### Reproducibility
| function | description |
|---|---|
| `planting(s)` | Set numpy + stdlib random seeds |

### Simulation
| function | description |
|---|---|
| `createN0(n_states, startCount)` | Random initial state counts |
| `statNam(n_states)` | Default state name list |
| `generateAggStates(N0, Q, n_generate, ...)` | Simulate forward in time |
| `make_theta0(k, format, ub, options)` | Random starting parameter vector |

### Data preparation
| function | description |
|---|---|
| `formatCount(data, ...)` | Pivot raw data → proportion table |
| `eqTin(df, ...)` | Normalise counts → proportion arrays |

### Model building
| function | description |
|---|---|
| `theta_to_Q(theta_u, n_states, Q_template)` | Parameter vector → rate matrix |
| `transMat(Q, u)` | Rate matrix → transition matrix via matrix exponential |
| `def_bounds(n_param, k, ...)` | scipy `Bounds` for the optimiser |
| `def_constraints(n_param, k)` | Row-sum constraint for the optimiser |

### Cost functions
| function | description |
|---|---|
| `calc_cost(theta, pi_hat, T, k, u, Q_template)` | SSR for a single dataset |
| `calc_cost_donors(theta, pi_hat_donors, ...)` | Pooled SSR across donors |

### Optimisation
| function | description |
|---|---|
| `oneCore_mc_optimiser(iter_samp, n_param, ...)` | Serial multi-start optimiser |
| `parallel_mc_optimiser(iter_samp, n_cores, ...)` | Parallel multi-start optimiser |
| `single_mc_optimiser(i, ...)` | Single optimisation run (trust-constr) |

### Inference
| function | description |
|---|---|
| `residCalc(dataArgTuple, theta_point)` | Compute per-interval residuals |
| `calc_donorSD(pi_hat_donors, k, timepoint)` | Cross-donor SD at a timepoint |
| `residResample_perDonor(i, ...)` | One bootstrap replicate via residual resampling |
| `truncated_t_ci(theta_point, se, dof, alpha)` | Bootstrap-t CIs |
| `calc_BCa_CI(theta_point, theta_boot, alpha)` | BCa CIs |
| `bootstrap_resResamp_perDonor(...)` | Full bootstrap pipeline (parallel inner) |
| `bootstrap_resResamp_perDonor_oneiter(...)` | Single bootstrap iter (HPC use) |
| `parallel_bootstrap_resResamp(...)` | Fully parallel bootstrap |
