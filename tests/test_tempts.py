"""
Tests for the tempts package.
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest

import tempts


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def seed():
    tempts.planting(0)


K = 3  # number of states used throughout

Q_TRUE = np.array([
    [-0.3,  0.2,  0.1],
    [ 0.0, -0.15, 0.15],
    [ 0.0,  0.0,  0.0],
])

# Q template: full upper triangular (includes diagonal entries in cols 0..k-2).
# This matches what make_theta0(format='Q') produces and what theta_to_Q expects.
def _make_template(k=K):
    Q_tmp = tempts.make_theta0(k, format='Q', ub=0.1)
    return (Q_tmp != 0).astype(float)


# ── planting / seed ───────────────────────────────────────────────────────────

def test_planting_reproducibility():
    tempts.planting(99)
    a = np.random.rand(5)
    tempts.planting(99)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


# ── createN0 ─────────────────────────────────────────────────────────────────

def test_createN0_shape():
    N0 = tempts.createN0(K, 1000)
    assert N0.shape == (K, 1)


def test_createN0_sum():
    N0 = tempts.createN0(K, 1000)
    assert abs(N0['d0'].sum() - 1000) <= K  # rounding slack


def test_createN0_custom_names():
    names = ['IgM', 'IgG', 'IgA']
    N0 = tempts.createN0(K, 1000, state_names=names)
    assert list(N0.index) == names


# ── statNam ──────────────────────────────────────────────────────────────────

def test_statNam():
    assert tempts.statNam(3) == ['State 1', 'State 2', 'State 3']


# ── transMat ─────────────────────────────────────────────────────────────────

def test_transMat_is_stochastic():
    P = tempts.transMat(Q_TRUE, 1.0)
    assert P.shape == (K, K)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(K), atol=1e-10)


def test_transMat_nonneg():
    P = tempts.transMat(Q_TRUE, 1.0)
    assert np.all(P >= -1e-10)


# ── theta_to_Q ───────────────────────────────────────────────────────────────

def test_theta_to_Q_roundtrip():
    tmpl = _make_template()
    theta = Q_TRUE[np.nonzero(Q_TRUE[:, :-1])]   # extract free params
    Q_rec = tempts.theta_to_Q(theta, K, tmpl)
    # off-diagonal entries should match
    np.testing.assert_allclose(Q_rec[:, :-1][np.nonzero(tmpl[:, :-1])],
                                theta, atol=1e-12)


def test_theta_to_Q_row_sums():
    tmpl = _make_template()
    theta = Q_TRUE[np.nonzero(Q_TRUE[:, :-1])]
    Q_rec = tempts.theta_to_Q(theta, K, tmpl)
    np.testing.assert_allclose(Q_rec.sum(axis=1), np.zeros(K), atol=1e-12)


# ── make_theta0 ───────────────────────────────────────────────────────────────

def test_make_theta0_length():
    tmpl = _make_template()
    n_param = int(np.count_nonzero(tmpl[:, :-1]))
    theta = tempts.make_theta0(K, format='theta', ub=1e-1)
    assert len(theta) == n_param


def test_make_theta0_Q_format():
    Q = tempts.make_theta0(K, format='Q')
    assert Q.shape == (K, K)
    # make_theta0 produces upper-triangular matrices (biological: forward only)
    assert np.all(Q[np.tril_indices(K, k=-1)] == 0), "Lower triangle should be zero"


# ── generateAggStates ─────────────────────────────────────────────────────────

def test_generateAggStates_shape():
    N0 = tempts.createN0(K, 1000)
    sim = tempts.generateAggStates(N0, Q_TRUE, n_generate=5, recruit=False)
    assert sim.shape == (K, 6)   # d0 + 5 steps


def test_generateAggStates_nonneg():
    N0 = tempts.createN0(K, 1000)
    sim = tempts.generateAggStates(N0, Q_TRUE, n_generate=5, recruit=False)
    assert (sim.values >= 0).all()


# ── eqTin ────────────────────────────────────────────────────────────────────

def test_eqTin_proportions_sum_to_one():
    N0 = tempts.createN0(K, 1000)
    sim = tempts.generateAggStates(N0, Q_TRUE, n_generate=5, recruit=False)
    pi_hat, T = tempts.eqTin(sim.to_numpy())
    col_sums = pi_hat.sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(T), atol=1e-10)


def test_eqTin_giveK():
    N0 = tempts.createN0(K, 1000)
    sim = tempts.generateAggStates(N0, Q_TRUE, n_generate=5, recruit=False)
    pi_hat, T, k = tempts.eqTin(sim.to_numpy(), giveK=True)
    assert k == K


# ── def_bounds / def_constraints ─────────────────────────────────────────────

def test_def_bounds_shape():
    tmpl = _make_template()
    n_param = int(np.count_nonzero(tmpl[:, :-1]))
    bounds = tempts.def_bounds(n_param, K)
    assert len(bounds.lb) == n_param


def test_def_constraints_shape():
    tmpl = _make_template()
    n_param = int(np.count_nonzero(tmpl[:, :-1]))
    con = tempts.def_constraints(n_param, K)
    assert con.A.shape[0] == K - 1


# ── calc_cost ─────────────────────────────────────────────────────────────────

def test_calc_cost_at_true_params_is_small():
    N0 = tempts.createN0(K, 100_000)
    sim = tempts.generateAggStates(N0, Q_TRUE, n_generate=10, recruit=False)
    pi_hat, T = tempts.eqTin(sim.to_numpy())
    u = np.ones(T - 1)
    tmpl = _make_template()
    theta = Q_TRUE[np.nonzero(Q_TRUE[:, :-1])]
    cost = tempts.calc_cost(theta, pi_hat, T, K, u, tmpl)
    assert cost < 1.0   # should be near-zero for large noise-free simulation


# ── truncated_t_ci ────────────────────────────────────────────────────────────

def test_truncated_t_ci_shape():
    theta = np.array([0.2, -0.15, 0.1])
    se    = np.array([0.02, 0.01, 0.015])
    ci = tempts.truncated_t_ci(theta, se, dof=99, alpha=0.05)
    assert ci.shape == (3, 2)


def test_truncated_t_ci_truncation():
    theta = np.array([0.1, -0.1])
    se    = np.array([0.01, 0.01])
    ci = tempts.truncated_t_ci(theta, se, dof=99)
    assert ci[0, 0] >= 0        # positive param: lower bound ≥ 0
    assert ci[1, 1] <= 0        # negative param: upper bound ≤ 0


# ── calc_BCa_CI ───────────────────────────────────────────────────────────────

def test_calc_BCa_CI_shape():
    np.random.seed(7)
    theta_point = np.array([0.2, 0.1])
    theta_boot  = np.random.normal(loc=theta_point, scale=0.02, size=(200, 2))
    ci, z0, a = tempts.calc_BCa_CI(theta_point, theta_boot, alpha=0.05)
    assert ci.shape == (2, 2)
    assert z0.shape == (2,)
    assert a.shape  == (2,)


def test_calc_BCa_CI_ordering():
    np.random.seed(7)
    theta_point = np.array([0.2, 0.1])
    theta_boot  = np.random.normal(loc=theta_point, scale=0.02, size=(200, 2))
    ci, _, _ = tempts.calc_BCa_CI(theta_point, theta_boot, alpha=0.05)
    assert (ci[:, 0] <= ci[:, 1]).all()


# ── oneCore_mc_optimiser (integration) ───────────────────────────────────────

def test_oneCore_mc_optimiser_recovers_params():
    """
    Verify that the single-core optimiser can recover Q_TRUE from clean data.
    Uses a large population so proportions are near-exact.
    """
    tempts.planting(1)
    N0  = tempts.createN0(K, 500_000)
    sim = tempts.generateAggStates(N0, Q_TRUE, n_generate=15, recruit=False)
    pi_hat, T = tempts.eqTin(sim.to_numpy())
    u    = np.ones(T - 1)
    tmpl = _make_template()
    n_param = int(np.count_nonzero(tmpl[:, :-1]))

    optimiserArgs = {
        'costFunc':      tempts.calc_cost,
        'args':          (pi_hat, T, K, u, tmpl),
        'bounds':        tempts.def_bounds(n_param, K),
        'constraints':   tempts.def_constraints(n_param, K),
        'sampRange_ub':  5e-1,
        'ingressAccount': False,
    }
    theta_est, cost = tempts.oneCore_mc_optimiser(
        iter_samp=20,
        n_param=n_param,
        theta_generator=tempts.make_theta0,
        optimiserArgs=optimiserArgs,
    )
    Q_est = tempts.theta_to_Q(theta_est, K, tmpl)

    # Check off-diagonal entries match to within 20 %
    for i in range(K):
        for j in range(K - 1):
            if tmpl[i, j]:
                np.testing.assert_allclose(
                    Q_est[i, j], Q_TRUE[i, j], rtol=0.20,
                    err_msg=f"Q[{i},{j}] mismatch"
                )
