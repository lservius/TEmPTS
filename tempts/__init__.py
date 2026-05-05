"""
TEmPTS — Transition Estimation for Proportional Time Sequences
==============================================================

Python package based on Servius et al.,
"State transition estimation in proportional data sequence
with applications in immunology."

Original scripts in: ../manuscript/

Quick-start
-----------
>>> import tempts
>>> tempts.planting(42)                        # reproducibility seed
>>> N0 = tempts.createN0(3, 1000)              # random initial counts
>>> import numpy as np
>>> Q_true = np.array([[-0.2, 0.1, 0.1],
...                     [0.0, -0.1, 0.1],
...                     [0.0,  0.0, 0.0]])
>>> sim = tempts.generateAggStates(N0, Q_true, n_generate=10)
"""

from tempts.functions import (  # noqa: F401
    # reproducibility
    planting,
    # baseline
    theta_to_Q,
    transMat,
    # simulation
    createN0,
    statNam,
    generateAggStates,
    make_theta0,
    # data preparation
    formatCount,
    eqTin,
    # bounds / constraints
    def_bounds,
    def_constraints,
    # cost functions
    calc_cost,
    calc_cost_donors,
    # optimisers
    single_mc_optimiser,
    initGlobal,
    oneCore_mc_optimiser,
    parallel_mc_optimiser,
    # residuals
    residCalc,
    calc_donorSD,
    residResample_perDonor,
    # confidence intervals
    truncated_t_ci,
    calc_BCa_CI,
    # bootstrap pipelines
    bootstrap_resResamp_perDonor,
    bootstrap_resResamp_perDonor_oneiter,
    parallel_bootstrap_resResamp,
)

__version__ = "0.1.0"
__author__ = "Servius et al. (packaged)"
