"""
TEmPTS core functions
=====================
Assembled from mm_temptations_functions.py (Servius et al.) found in ../manuscript.

All original function names and signatures are preserved so examples in ../manuscript would be converted like so:

    # Before (script import)
    from mm_temptations_functions import *

    # After (package import)
    from tempts import *
    # or
    import tempts
"""

import pickle
import warnings
from functools import partial
from math import ceil

import numpy as np
import pandas as pd
import random as rand
from scipy.linalg import expm
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import norm, skew, t, truncnorm

# multiprocess (drop-in for multiprocessing with better pickling) is preferred;
# fall back to stdlib multiprocessing if not installed.
try:
    from multiprocess import Pool
    _HAS_MULTIPROCESS = True
except ImportError:
    from multiprocessing import Pool
    _HAS_MULTIPROCESS = False

try:
    import progressbar
    _HAS_PROGRESSBAR = True
except ImportError:
    _HAS_PROGRESSBAR = False

warnings.filterwarnings(
    "ignore",
    message="delta_grad == 0.0. Check if the approximated function is linear.",
)

# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------

def planting(s):
    """Seed both numpy and stdlib random for reproducibility."""
    np.random.seed(s)
    rand.seed(s)


# ---------------------------------------------------------------------------
# BASELINE CALCULATIONS
# ---------------------------------------------------------------------------

def theta_to_Q(theta_u, n_states, Q_template=np.array(None)):
    """
    Map an unconstrained parameter vector *theta_u* back to a rate matrix Q.

    Parameters
    ----------
    theta_u : array_like
        Flattened non-zero off-diagonal entries of Q (row-major, upper
        triangle only — biological constraint: no backwards transitions).
    n_states : int
        Number of states.
    Q_template : ndarray, shape (n_states, n_states)
        Template matrix whose non-zero entries (excluding the last column,
        which is the diagonal) indicate which entries of Q are free.

    Returns
    -------
    Q_new : ndarray, shape (n_states, n_states)
        Rate (generator) matrix with row sums == 0.
    """
    R = len(np.nonzero(Q_template[:, :-1])[0])
    Q_new = np.zeros((n_states, n_states))
    Q_to_theta = np.nonzero(Q_template[:, :-1])

    for r in range(R):
        i = Q_to_theta[0][r]
        j = Q_to_theta[1][r]
        Q_new[i, j] = theta_u[r]

    for r in range(n_states):
        Q_new[r, -1] = -sum(Q_new[r, :(n_states - 1)])

    return Q_new


def transMat(Q, u):
    """
    Compute the transition probability matrix P = expm(Q * u).

    Parameters
    ----------
    Q : ndarray, shape (k, k)
        Rate matrix.
    u : float
        Time interval length.

    Returns
    -------
    P_l : ndarray, shape (k, k)
    """
    P_l = expm(Q * u)
    return P_l


# ---------------------------------------------------------------------------
# SIMULATION FUNCTIONS
# ---------------------------------------------------------------------------

def createN0(n_states, startCount, state_names='None'):
    """
    Generate random initial state counts that sum to *startCount*.

    Parameters
    ----------
    n_states : int
    startCount : int
        Total number of cells/entities at t=0.
    state_names : list or 'None'

    Returns
    -------
    df : DataFrame, shape (n_states, 1), column 'd0'
    """
    prop = np.zeros(n_states)
    mag = int(np.log10(startCount))

    for i in range(n_states):
        if i == 0:
            prop[i] = round(rand.uniform(0.2, 1), mag)
        elif i < (n_states - 1):
            prop[i] = round(rand.uniform(0, 1 - sum(prop)), mag)
        else:
            prop[i] = 1 - sum(prop)

    vals = startCount * prop
    stateCount = np.round(vals)

    if state_names == 'None':
        state_names = [*range(n_states)]

    df = pd.DataFrame({'d0': stateCount}, index=state_names)
    return df


def statNam(n_states):
    """Return a list of default state names ['State 1', 'State 2', ...]."""
    return ['State ' + str(i + 1) for i in range(n_states)]


def generateAggStates(
    N0, Q_ij, n_generate, u=[1], recruit=True,
    timeCount=int(), timeMarker='d', noise=False
):
    """
    Simulate aggregate state counts forward in time using *Q_ij*.

    Parameters
    ----------
    N0 : DataFrame
        Initial counts, as returned by :func:`createN0`.
    Q_ij : ndarray, shape (k, k)
        Rate matrix.
    n_generate : int
        Number of time steps to simulate.
    u : list of float
        Time interval(s). If all equal to 1 the transition matrix is
        pre-computed once for efficiency.
    recruit : bool
        Whether to add Poisson-distributed recruits to state 0 each step.
    timeCount : int
        Starting time index.
    timeMarker : str
        Column name prefix (e.g. 'd' → 'd0', 'd1', …).
    noise : bool
        Whether to add 1 % Poisson noise to each step.

    Returns
    -------
    df : DataFrame
        Wide-format table with one column per time point.
    """
    df = N0.copy()
    firstDay = timeMarker + str(timeCount)
    all_one_interval = all(u[i] == 1 for i in range(len(u)))

    if all_one_interval:
        P_ij = transMat(Q_ij, 1)

    iter_ = 1
    while iter_ <= n_generate:
        if all_one_interval:
            today = timeMarker + str(timeCount)
            tomorrow = timeMarker + str(timeCount + u[0])
            timeCount += 1
        else:
            P_ij = transMat(Q_ij, u[iter_])
            today = timeMarker + str(timeCount)
            tomorrow = timeMarker + str(timeCount + u[iter_])
            timeCount += u[iter_]

        initial_values = df[today]
        d1 = []
        n_state = len(P_ij)
        d0 = initial_values

        for j in range(n_state):
            P_i = P_ij[:, j]
            dotProd = int(sum(d0 * P_i))
            d1.append(dotProd)

        if recruit:
            recLam = int(np.log10(sum(df[firstDay])) - 2)
            d1[0] = int(d1[0] + np.random.poisson(10 ** recLam, 1))

        df[tomorrow] = d1

        if noise:
            scale_noise = df[tomorrow] * 0.01
            df[tomorrow] += np.random.poisson(lam=scale_noise)
            df.loc[df[tomorrow] < 0, tomorrow] = 0

        iter_ += 1

    return df


def make_theta0(k, format='theta', ub=1e-1, options=None):
    """
    Generate a random starting parameter vector (or matrix) for optimisation.

    Parameters
    ----------
    k : int
        Number of states.
    format : {'theta', 'Q'}
        Return the full rate matrix or just the free-parameter vector.
    ub : float
        Upper bound for uniform sampling of off-diagonal entries.
    options : dict
        ``{'ingress': bool, 'initial': float}``

    Returns
    -------
    result : ndarray
    """
    if options is None:
        options = {'ingress': False, 'initial': 1e0}

    Q_0 = np.zeros((k, k))
    vals = np.array([rand.uniform(1e-12, ub) for _ in range(k * (k - 1) // 2)])
    Q_0[np.triu_indices(k, 1)] = vals                           # biological: no backwards
    
    # diagonal = negative row sum
    np.fill_diagonal(Q_0, -Q_0.sum(axis=1))

    if format == 'Q':
        result = Q_0
    elif format == 'theta':
        theta_0 = Q_0[np.nonzero(Q_0[:, :-1])]
        result = theta_0
        if options["ingress"] == True:
            result = np.append(theta_0, options['initial'])
    
    return result


# ---------------------------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------------------------

def formatCount(
    data, day_series, timepoint_col_name, state_col_name,
    timemarker, isotype_list, proportion_col_name=None
):
    """
    Pivot raw longitudinal data into a (states × timepoints) proportion table.

    Parameters
    ----------
    data : DataFrame
    day_series : array_like
        Ordered sequence of time point values present in *timepoint_col_name*.
    timepoint_col_name : str
    state_col_name : str
    timemarker : str
    isotype_list : list
        Ordered list of state labels.
    proportion_col_name : str or None
        If the data already contains pre-computed proportions, name the column
        here; otherwise counts are normalised automatically.

    Returns
    -------
    ISO : DataFrame, shape (len(isotype_list), len(day_series))
    """
    ISO = pd.DataFrame()

    data = data[data[state_col_name].isin(isotype_list)]

    if proportion_col_name is not None:
        ISO = data.pivot_table(
            values=proportion_col_name,
            index=state_col_name,
            columns=timepoint_col_name,
        )
        ISO = ISO.reindex(isotype_list).replace(np.nan, 0)
        return ISO

    for d in range(len(day_series)):
        day = day_series[d]
        day_name = timemarker + str(day)
        data_i = data[data[timepoint_col_name] == day]
        data_i = data_i[state_col_name].value_counts()
        total = data_i.sum()
        data_i = pd.Series.to_frame(data_i / total)
        ISO[day_name] = data_i

    ISO = ISO.reindex(isotype_list).replace(np.nan, 0)
    return ISO


def eqTin(
    df, proportion=True, giveK=False, perDonor=False,
    donor_list=None, donor_column='', time_column='',
    state_column='', state_list=None
):
    """
    Convert a count/proportion DataFrame into numpy arrays for optimisation.

    Can operate in a *per-donor* mode that returns dictionaries keyed by
    donor identifier.

    Parameters
    ----------
    df : DataFrame or ndarray
    proportion : bool
        Whether to normalise columns to sum to 1.
    giveK : bool
        Whether to also return *k* (number of states).
    perDonor : bool
        Process each donor separately.
    donor_list, donor_column, time_column, state_column, state_list
        Required when *perDonor=True*.

    Returns
    -------
    Varies by flags; see source for full details.
    """
    if donor_list is None:
        donor_list = []
    if state_list is None:
        state_list = []

    T = df.shape[1]
    k = df.shape[0]
    input_pi_hat = df.copy() if hasattr(df, 'copy') else np.array(df)
    pi_hat = np.zeros(np.array(input_pi_hat).shape)

    if perDonor:
        pi_hat_donors = {}
        T_donors = {}
        u_donors = {}
        timeCourse = {}

        for d in donor_list:
            df_donor = df.loc[df[donor_column] == d, :]
            timepoint_list = np.sort(df_donor[time_column].unique())
            timeCourse[d] = timepoint_list.shape[0]
            df_donor = formatCount(
                df_donor, timepoint_list, time_column, state_column,
                timemarker='d', isotype_list=state_list,
            )
            df_donor = df_donor.to_numpy()
            pi_hat, T, k = eqTin(df_donor, proportion=False, giveK=True)
            pi_hat_donors[d] = pi_hat
            T_donors[d] = T
            u_donors[d] = np.diff(timepoint_list)

        return pi_hat_donors, T_donors, k, u_donors

    input_pi_hat = np.array(input_pi_hat)

    if not proportion:
        M = np.delete(input_pi_hat, -1, 0)  # noqa: F841 (kept for API compat)
        return input_pi_hat, T, k

    for t in range(T):
        pi_hat_t = input_pi_hat[:, t]
        sum_t = sum(pi_hat_t)
        pi_hat_t = pi_hat_t / sum_t
        pi_hat[:, t] = pi_hat_t

    if giveK:
        return pi_hat, T, k

    return pi_hat, T


# ---------------------------------------------------------------------------
# BOUNDS AND CONSTRAINTS FOR MINIMISATION
# ---------------------------------------------------------------------------

def def_bounds(n_param, k, account_ingress=False, serial_corr=False):
    """
    Build scipy ``Bounds`` for the parameter vector.

    Diagonal entries (negative rates) get (−∞, 0]; off-diagonal entries get
    [0, +∞).

    Parameters
    ----------
    n_param : int
    k : int
        Number of states.
    account_ingress : bool
        Extend bounds to accommodate an ingress parameter in [−1, 1].
    serial_corr : bool
        Extend bounds to accommodate a serial-correlation parameter in [0, 1].

    Returns
    -------
    scipy.optimize.Bounds
    """
    q_lb = np.zeros(n_param)
    q_ub = np.zeros(n_param)
    q_diag = np.zeros(k - 1)

    a = -1 / 2
    b = (k - 2) - 3 * a
    c = 0

    for n in range(k - 1):
        q_diag[n] = a * n ** 2 + b * n + c

    for n in range(n_param):
        if n in q_diag:
            q_lb[n] = -np.inf
            q_ub[n] = 0
        else:
            q_lb[n] = 0
            q_ub[n] = np.inf

    reg_bounds = Bounds(q_lb, q_ub)

    if account_ingress:
        q_lb[-1] = -1
        q_ub[-1] = 1
        return Bounds(q_lb, q_ub)

    if serial_corr:
        q_lb[-1] = 0
        q_ub[-1] = 1
        return Bounds(q_lb, q_ub)

    return reg_bounds


def def_constraints(n_param, k):
    """
    Build a ``LinearConstraint`` enforcing row-sum ≤ 0 on Q.

    Parameters
    ----------
    n_param : int
    k : int

    Returns
    -------
    scipy.optimize.LinearConstraint
    """
    row_eq = np.zeros((k - 1, n_param))

    a = -1 / 2
    b = (k - 2) - 3 * a
    c = 0

    for n in range(k - 1):
        i = int(a * n ** 2 + b * n + c)
        j = int(a * (n + 1) ** 2 + b * (n + 1) + c)
        row_eq[n, i:j] = 1

    lin_con = LinearConstraint(row_eq, ub=np.zeros(k - 1), keep_feasible=True)
    return lin_con


# ---------------------------------------------------------------------------
# COST FUNCTIONS
# ---------------------------------------------------------------------------

def calc_cost(theta, pi_hat, T, k, u, Q_template=np.array(None), stop_region=1e5):
    """
    Sum of squared residuals between observed and model-predicted proportions.

    Parameters
    ----------
    theta : array_like
        Current parameter vector.
    pi_hat : ndarray, shape (k, T)
        Observed proportions at each time point (columns).
    T : int
        Number of time points.
    k : int
        Number of states.
    u : array_like
        Time intervals between consecutive observations, length T−1.
    Q_template : ndarray
        Template passed to :func:`theta_to_Q`.
    stop_region : float
        Return this value when numerics break down to guide the optimiser away.

    Returns
    -------
    cost : float
    """
    cost = 0.0
    Q = theta_to_Q(theta, k, Q_template)

    for l in range(1, T):
        u_l = u[l - 1]
        pi_hat_star_l = pi_hat[:-1, l]
        pi_hat_lm1 = pi_hat[:, l - 1]

        P_l = expm(Q * u_l)

        if np.isnan(P_l).any() or np.isinf(P_l).any():
            return np.array(stop_region)

        P1_l = P_l[:, :-1]
        cost_1 = pi_hat_star_l - (P1_l.T @ pi_hat_lm1)
        norm_cost_1 = np.linalg.norm(cost_1, np.inf)

        if norm_cost_1 > stop_region or np.isnan(cost_1).any():
            return np.array(stop_region)

        cost_l = cost_1.T @ cost_1

        if np.iscomplexobj(cost_l):
            return np.array(stop_region)

        cost += cost_l

    return cost


def calc_cost_donors(
    theta, pi_hat_donors, T_donors, k, u_donors,
    Q_template=np.array(None), stop_region=1e5
):
    """
    Multi-donor cost: sum of per-donor SSR, scaled by relative time coverage.

    Parameters
    ----------
    theta : array_like
    pi_hat_donors : dict  {donor_id: ndarray (k, T_d)}
    T_donors : dict       {donor_id: int}
    k : int
    u_donors : dict       {donor_id: array_like}
    Q_template : ndarray
    stop_region : float

    Returns
    -------
    cost : float
    """
    cost = 0.0
    T_max = max(T_donors.values())

    for d in pi_hat_donors:
        pi_hat = pi_hat_donors[d]
        T = T_donors[d]
        u = u_donors[d]
        scale_factor = T_max / T
        cost_donor = (
            calc_cost(theta, pi_hat, T, k, u, Q_template, stop_region)
            * scale_factor
        )
        if cost_donor >= stop_region:
            return stop_region
        cost += cost_donor

    return cost


# ---------------------------------------------------------------------------
# OPTIMISER
# ---------------------------------------------------------------------------

def single_mc_optimiser(
    i, ingressInitial=0, optimiserArgs=None, options=None,
    theta_generator=None, hessian=None
):
    """
    Run one trust-region constrained minimisation from a random starting point.

    Parameters
    ----------
    i : int
        Iteration index (returned as-is for bookkeeping).
    ingressInitial : float
        Initial value for any ingress parameter.
    optimiserArgs : dict
        Keys: ``costFunc``, ``args``, ``bounds``, ``constraints``,
        ``sampRange_ub``, ``ingressAccount``.
    options : dict
        Keys: ``xterm``, ``gterm``, ``max_iter``.
    theta_generator : callable
        Function with signature ``(k, ub, options) → theta_0``.
    hessian : callable or None

    Returns
    -------
    (i, theta_est, cost_min) or (None, inf) on failure
    """
    if optimiserArgs is None:
        raise ValueError("optimiserArgs cannot be None")
    if options is None:
        options = {'xterm': 1e-8, 'gterm': 1e-15, 'max_iter': 1e4}
    if theta_generator is None:
        raise ValueError("theta_generator must be supplied")

    func = optimiserArgs.get('costFunc')
    args = optimiserArgs.get('args')
    bounds = optimiserArgs.get('bounds')
    constraints = optimiserArgs.get('constraints')
    sampRange_ub = optimiserArgs.get('sampRange_ub')
    ingressAccount = optimiserArgs.get('ingressAccount', False)

    xtol = options.get('xterm', 1e-8)
    gtol = options.get('gterm', 1e-15)
    maxiter = options.get('max_iter', 1e4)

    theta_0 = theta_generator(
        k=args[2], ub=sampRange_ub,
        options={'ingress': ingressAccount, 'initial': ingressInitial},
    )

    try:
        result = minimize(
            func, theta_0, args=args, bounds=bounds,
            constraints=constraints, hess=hessian,
            method='trust-constr',
            options={
                'verbose': 0,
                'initial_tr_radius': 1e3,
                'initial_constr_penalty': 1e2,
                'xtol': xtol,
                'gtol': gtol,
                'maxiter': maxiter,
            },
        )
        return i, result.x, result.fun
    except Exception as e:
        print(f"Optimization failed at ingressInitial={ingressInitial}: {e}")
        return None, np.inf


def initGlobal(optimiserArgs_dict, thetaGen_function, option_dict):
    """Expose optimiser state to worker processes via module-level globals."""
    global optimiserArgs, theta_generator, options
    optimiserArgs = optimiserArgs_dict
    theta_generator = thetaGen_function
    options = option_dict


def oneCore_mc_optimiser(
    iter_samp, n_param, theta_generator=make_theta0,
    optimiserArgs=None, options=None, progress_bar=False
):
    """
    Serial multi-start optimiser: run *iter_samp* restarts, return the best.

    Parameters
    ----------
    iter_samp : int
    n_param : int
    theta_generator : callable
    optimiserArgs : dict
    options : dict
    progress_bar : bool

    Returns
    -------
    (theta_best, cost_best) : (ndarray, float)
    """
    if optimiserArgs is None:
        optimiserArgs = {
            'costFunc': None, 'args': None, 'bounds': None,
            'constraints': None, 'sampRange_ub': None, 'ingressAccount': False,
        }
    if options is None:
        options = {
            'initial': 1e0, 'hessian': None,
            'xterm': 1e-8, 'gterm': 1e-15, 'max_iter': 1e4,
        }

    optimiserArgs.setdefault('ingressAccount', False)
    options.setdefault('gterm', 1e-15)
    initGlobal(optimiserArgs, theta_generator, options)

    theta_samples = np.zeros((iter_samp, n_param))
    cost_samples = np.full(iter_samp, np.inf)

    if progress_bar and _HAS_PROGRESSBAR:
        bar = progressbar.ProgressBar(
            maxval=iter_samp,
            widgets=[
                'Optimisation progress: ',
                progressbar.Bar('=', '[', ']'), ' ',
                progressbar.Percentage(),
            ],
        )
        bar.start()

    for i, *rest in map(
        partial(
            single_mc_optimiser,
            theta_generator=theta_generator,
            optimiserArgs=optimiserArgs,
            options=options,
        ),
        range(iter_samp),
    ):
        if rest[0] is not np.inf:
            est, cost = rest[0], rest[1]
            theta_samples[i] = est
            cost_samples[i] = cost
        if progress_bar and _HAS_PROGRESSBAR:
            bar.update(i)

    if progress_bar and _HAS_PROGRESSBAR:
        bar.finish()

    min_idx = np.argmin(cost_samples)
    return theta_samples[min_idx], cost_samples[min_idx]


def parallel_mc_optimiser(
    iter_samp, n_cores, n_param, theta_generator=make_theta0,
    optimiserArgs=None, options=None, progress_bar=False,
    convergence_tol=None, pool=None,
):
    if optimiserArgs is None:
        optimiserArgs = {
            'costFunc': None, 'args': None, 'bounds': None,
            'constraints': None, 'sampRange_ub': None, 'ingressAccount': False,
        }
    if options is None:
        options = {
            'initial': 1e0, 'hessian': None,
            'xterm': 1e-8, 'gterm': 1e-15, 'max_iter': 1e4,
        }
    optimiserArgs.setdefault('ingressAccount', False)
    options.setdefault('gterm', 1e-15)

    worker = partial(
        single_mc_optimiser,
        optimiserArgs=optimiserArgs,
        options=options,
        theta_generator=theta_generator,
    )
    chunksize = max(1, ceil(iter_samp / (n_cores * 4)))

    best_cost = np.inf
    best_theta = np.zeros(n_param)

    if progress_bar and _HAS_PROGRESSBAR:
        bar = progressbar.ProgressBar(maxval=iter_samp, redirect_stdout=True)
        bar.start()

    _owns_pool = pool is None
    if _owns_pool:
        pool = Pool(processes=n_cores)

    try:
        completed = 0
        for result in pool.imap_unordered(worker, range(iter_samp), chunksize=chunksize):
            i, est, cost = result
            if i is not None and cost < best_cost:
                best_cost = cost
                best_theta = est.copy()
            completed += 1
            if progress_bar and _HAS_PROGRESSBAR:
                bar.update(completed)
            if convergence_tol is not None and best_cost < convergence_tol:
                pool.terminate()
                break
    except Exception as e:
        print(f"Error in parallel execution: {e}")
    finally:
        if _owns_pool:
            pool.close()
            pool.join()

    if progress_bar and _HAS_PROGRESSBAR:
        bar.finish()

    return best_theta, best_cost


# ---------------------------------------------------------------------------
# RESIDUAL CALCULATION
# ---------------------------------------------------------------------------

def residCalc(dataArgTuple, theta_point):
    """
    Compute per-time-interval residuals for all donors.

    Parameters
    ----------
    dataArgTuple : tuple
        ``(pi_hat_donors, T_donors, k, u_donors, Q_template)``
    theta_point : ndarray
        Point estimate of the parameter vector.

    Returns
    -------
    resid_lib : ndarray, shape (k-1, n_residuals)
        Each column is one (k-1)-dimensional residual vector.
    """
    pi_hat_donors, T_donors, k, u_donors, Q_template = dataArgTuple
    donor_list = pi_hat_donors.keys()
    Q_est = theta_to_Q(theta_point, k, Q_template)
    resid_list = []

    for d in donor_list:
        pi_hat = pi_hat_donors[d]
        T = T_donors[d]
        u = u_donors[d]
        for i in range(1, T):
            P_est = transMat(Q_est, u[i - 1])
            epsi = (
                pi_hat[:(k - 1), i]
                - np.matmul(P_est[:, :(k - 1)].T, pi_hat[:, i - 1])
            )
            resid_list.append(epsi)

    resid_lib = (
        np.column_stack(resid_list) if resid_list
        else np.empty((k - 1, 0))
    )
    return resid_lib


def calc_donorSD(pi_hat_donors, k, timepoint=0):
    """
    Standard deviation across donors at a given time point.

    Parameters
    ----------
    pi_hat_donors : dict
    k : int
    timepoint : int

    Returns
    -------
    donorSD_t : ndarray, shape (k,)
    """
    donor_no = len(pi_hat_donors)
    if donor_no == 0:
        return np.zeros(k)

    time_pi_vec = np.column_stack(
        [pi_hat_donors[d][:, timepoint] for d in pi_hat_donors]
    )
    return np.std(time_pi_vec, axis=1, ddof=1)


def residResample_perDonor(i, dataArgTuple, point_est, resid_lib, t0_donorSD):
    """
    Generate one bootstrap replicate via residual resampling.

    The starting proportions are perturbed by sampling from a truncated normal
    distribution (mean = observed, sd = cross-donor SD); subsequent time
    points are obtained by propagating the fitted model and adding a randomly
    drawn residual.

    Parameters
    ----------
    i : int
        Bootstrap index (unused internally; kept for ``imap`` compatibility).
    dataArgTuple : tuple
    point_est : ndarray
    resid_lib : ndarray, shape (k-1, n_residuals)
    t0_donorSD : ndarray, shape (k-1,)

    Returns
    -------
    prop_boot_dict : dict  {donor_id: ndarray (k, T_d)}
    """
    pi_hat_donors, T_donors, k, u_donors, Q_template = dataArgTuple
    prop_boot_dict = {
        d: np.zeros_like(pi_hat) for d, pi_hat in pi_hat_donors.items()
    }
    Q_est = theta_to_Q(point_est, k, Q_template)

    for d, pi_hat in pi_hat_donors.items():
        T = T_donors[d]
        u = u_donors[d]

        a = (0.0 - pi_hat[:(k - 1), 0]) / t0_donorSD
        b = (1.0 - pi_hat[:(k - 1), 0]) / t0_donorSD
        prop_boot_dict[d][:(k - 1), 0] = truncnorm.rvs(
            a, b, loc=pi_hat[:(k - 1), 0], scale=t0_donorSD
        )
        prop_boot_dict[d][(k - 1), 0] = (
            1 - prop_boot_dict[d][:(k - 1), 0].sum()
        )

        pi_fitted = np.zeros_like(pi_hat)
        pi_fitted[:, 0] = prop_boot_dict[d][:, 0]

        for t in range(1, T):
            P_est = transMat(Q_est, u[t - 1])
            pi_fitted[:(k - 1), t] = (
                P_est[:, :(k - 1)].T @ pi_fitted[:, t - 1]
            )
            sampRes = np.apply_along_axis(np.random.choice, 1, resid_lib)
            prop_boot_dict[d][:(k - 1), t] = (
                pi_fitted[:(k - 1), t] + sampRes
            )
            prop_boot_dict[d][(k - 1), t] = max(
                0, 1 - prop_boot_dict[d][:(k - 1), t].sum()
            )

    return prop_boot_dict


# ---------------------------------------------------------------------------
# CONFIDENCE INTERVALS
# ---------------------------------------------------------------------------

def truncated_t_ci(theta_point, std_error, dof, alpha=0.05):
    """
    Bootstrap-*t* confidence intervals with sign-aware truncation.

    Negative entries (diagonal / rate terms) are truncated to (−∞, 0];
    positive entries to [0, +∞).

    Parameters
    ----------
    theta_point : ndarray
    std_error : ndarray
    dof : int
        Degrees of freedom (typically B − 1).
    alpha : float

    Returns
    -------
    ci_matrix : ndarray, shape (n_params, 2)
        Columns are [lower, upper].
    """
    bounds = np.zeros((len(theta_point), 2))
    for i, val in enumerate(theta_point):
        if val < 0:
            bounds[i] = [-np.inf, 0]
        else:
            bounds[i] = [0, np.inf]

    ci_matrix = np.zeros((len(theta_point), 2))
    t_alpha = t.ppf(1 - alpha / 2, dof)

    for idx, point_estimate in enumerate(theta_point):
        lower_ci = point_estimate - t_alpha * std_error[idx]
        upper_ci = point_estimate + t_alpha * std_error[idx]
        ci_matrix[idx, 0] = max(bounds[idx, 0], lower_ci)
        ci_matrix[idx, 1] = min(bounds[idx, 1], upper_ci)

    return ci_matrix


def calc_BCa_CI(theta_point, theta_boot, alpha):
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence intervals.

    The acceleration *a* is estimated via the skewness approximation.

    Parameters
    ----------
    theta_point : ndarray, shape (n_params,)
    theta_boot : ndarray, shape (B, n_params)
    alpha : float

    Returns
    -------
    ci_matrix : ndarray, shape (n_params, 2)
    z0_vec : ndarray, shape (n_params,)
    a_vec : ndarray, shape (n_params,)
    """
    z0_vec = np.zeros(len(theta_point))
    a_vec = np.zeros(len(theta_point))
    ci_matrix = np.zeros((len(theta_point), 2))

    for idx, point_estimate in enumerate(theta_point):
        theta_boot_i = theta_boot[:, idx]

        z0 = np.mean(theta_boot_i < point_estimate)
        z0 = np.clip(z0, 1e-6, 1 - 1e-6)
        z0 = norm.ppf(z0)

        a = skew(theta_boot_i, axis=0, bias=False) / 6

        z0_vec[idx] = z0
        a_vec[idx] = a

        z_lo = norm.ppf(alpha / 2)
        z_hi = norm.ppf(1 - alpha / 2)

        adj_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
        adj_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

        ci_matrix[idx, 0] = np.quantile(theta_boot_i, adj_lo)
        ci_matrix[idx, 1] = np.quantile(theta_boot_i, adj_hi)

    return ci_matrix, z0_vec, a_vec


# ---------------------------------------------------------------------------
# BOOTSTRAP WRAPPERS
# ---------------------------------------------------------------------------

def bootstrap_resResamp_perDonor(
    point_estimate=None, n_bootstrapSamples=None, n_param=None,
    theta_generator=None, mcArgs=None, optimiserArgs=None,
    options=None, startIteration=0, checkpoint_filename='',
):
    """
    Full bootstrap pipeline: point estimation → residual resampling → CIs.

    Uses :func:`parallel_mc_optimiser` internally.  Results are checkpointed
    to *checkpoint_filename* after every iteration.

    Parameters
    ----------
    point_estimate : ndarray or None
        Pre-computed point estimate; if None it is calculated first.
    n_bootstrapSamples : int
    n_param : int
    theta_generator : callable
    mcArgs : dict
        Keys: ``mciter``, ``n_cores``.
    optimiserArgs : dict
    options : dict
        Must contain ``alpha_significance``.
    startIteration : int
        Resume from this bootstrap index (0-based).
    checkpoint_filename : str

    Returns
    -------
    bootstrapOutput : dict
        Keys: ``theta_est``, ``bootEstimates``, ``standardError``,
        ``confidenceInterval_studentt``, ``confidenceInterval_BCa``,
        ``BCa_biasAccel``, ``significanceLevel``.
    """
    if options is None:
        options = {'alpha_significance': 0.05}

    B = n_bootstrapSamples
    theta_boot = np.zeros((B, n_param))

    if point_estimate is None or point_estimate.size == 0:
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(
            iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'],
            n_param=n_param, theta_generator=theta_generator,
            optimiserArgs=optimiserArgs, options=options, progress_bar=True,
        )
    else:
        theta_point = point_estimate

    pi_hat_donors, T_donors, k, u_donors, Q_template = optimiserArgs['args']
    resid_dict = residCalc(optimiserArgs['args'], theta_point)
    t0_donorSD = calc_donorSD(pi_hat_donors, k, timepoint=0)[:(k - 1)]

    try:
        with open(checkpoint_filename, 'rb') as f:
            theta_boot = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("Checkpoint file not found or corrupted. Starting fresh.")

    iterator = range(startIteration, B)
    if _HAS_PROGRESSBAR:
        iterator = progressbar.progressbar(iterator, redirect_stdout=True)

    for i in iterator:
        prop_boot_donors = residResample_perDonor(
            i, optimiserArgs['args'], theta_point, resid_dict, t0_donorSD,
        )
        optimiserArgs['args'] = (
            prop_boot_donors, T_donors, k, u_donors, Q_template
        )
        theta_b, _cost = parallel_mc_optimiser(
            iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'],
            n_param=n_param, theta_generator=theta_generator,
            optimiserArgs=optimiserArgs, options=options,
        )
        theta_boot[i, :] = theta_b

        if checkpoint_filename:
            with open(checkpoint_filename, 'wb') as f:
                pickle.dump(theta_boot, f)

    print("Compiling results...")
    se_boot = np.std(theta_boot, ddof=1, axis=0)
    alpha = options.get('alpha_significance', 0.05)
    conInt_alpha = truncated_t_ci(theta_point, se_boot, B - 1, alpha)
    conIntBCa_alpha, z0_vec, a_vec = calc_BCa_CI(theta_point, theta_boot, alpha)

    return {
        'theta_est': theta_point,
        'bootEstimates': theta_boot,
        'standardError': se_boot,
        'confidenceInterval_studentt': conInt_alpha,
        'confidenceInterval_BCa': conIntBCa_alpha,
        'BCa_biasAccel': [z0_vec, a_vec],
        'significanceLevel': alpha,
    }


def bootstrap_resResamp_perDonor_oneiter(
    point_estimate=None, n_bootstrapSamples=None, n_param=None,
    theta_generator=None, mcArgs=None, optimiserArgs=None,
    options=None, startIteration=0, checkpoint_filename='',
):
    """
    Single-iteration bootstrap (for embarrassingly-parallel HPC use).

    Saves the result to *checkpoint_filename* as a CSV.

    Returns
    -------
    theta_boot : ndarray, shape (n_param,)
    """
    if options is None:
        options = {'alpha_significance': 0.05}

    if point_estimate is None or point_estimate.size == 0:
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(
            iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'],
            n_param=n_param, theta_generator=theta_generator,
            optimiserArgs=optimiserArgs, options=options, progress_bar=True,
        )
    else:
        theta_point = point_estimate

    pi_hat_donors, T_donors, k, u_donors, Q_template = optimiserArgs['args']
    resid_dict = residCalc(optimiserArgs['args'], theta_point)
    t0_donorSD = calc_donorSD(pi_hat_donors, k, timepoint=0)[:(k - 1)]

    prop_boot_donors = residResample_perDonor(
        startIteration, optimiserArgs['args'], theta_point,
        resid_dict, t0_donorSD,
    )
    optimiserArgs['args'] = (
        prop_boot_donors, T_donors, k, u_donors, Q_template
    )
    theta_b, _cost = parallel_mc_optimiser(
        iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'],
        n_param=n_param, theta_generator=theta_generator,
        optimiserArgs=optimiserArgs, options=options,
    )

    if checkpoint_filename:
        np.savetxt(checkpoint_filename, theta_b, delimiter=",")
        print("Results saved!")

    return theta_b


def parallel_bootstrap_resResamp(
    point_estimate=None, n_bootstrapSamples=None, n_param=None,
    theta_generator=None, mcArgs=None, optimiserArgs=None,
    options=None, startIteration=0, checkpoint_filename='',
):
    """
    Fully-parallel bootstrap: bootstrap iterations run in a process pool,
    each using a single-core inner optimiser.

    Parameters and return value are identical to
    :func:`bootstrap_resResamp_perDonor`.
    """
    if options is None:
        options = {'alpha_significance': 0.05}

    if point_estimate is None or (
        hasattr(point_estimate, 'any') and point_estimate.any() is None
    ):
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(
            iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'],
            n_param=n_param, theta_generator=theta_generator,
            optimiserArgs=optimiserArgs, options=options, progress_bar=True,
        )
    else:
        theta_point = point_estimate

    B = n_bootstrapSamples
    theta_boot = np.zeros((B, n_param))
    pi_hat_donors, T_donors, k, u_donors, Q_template = optimiserArgs['args']
    resid_dict = residCalc(optimiserArgs['args'], theta_point)
    t0_donorSD = calc_donorSD(pi_hat_donors, k, timepoint=0)[:(k - 1)]

    try:
        with open(checkpoint_filename, 'rb') as f:
            theta_boot = pickle.load(f)
    except FileNotFoundError:
        print("Checkpoint not found. Starting fresh.")

    if _HAS_PROGRESSBAR:
        bar = progressbar.ProgressBar(maxval=B, min_value=startIteration)
        bar.start()

    def bootstrap_iteration(i):
        prop_boot_donors = residResample_perDonor(
            i, optimiserArgs['args'], theta_point, resid_dict, t0_donorSD,
        )
        local_args = (
            prop_boot_donors, T_donors, k, u_donors, Q_template
        )
        local_optimiserArgs = dict(optimiserArgs)
        local_optimiserArgs['args'] = local_args
        theta_b, _cost = oneCore_mc_optimiser(
            iter_samp=mcArgs['mciter'], n_param=n_param,
            theta_generator=theta_generator,
            optimiserArgs=local_optimiserArgs, options=options,
        )
        return i, theta_b

    n_cores = mcArgs['n_cores']
    with Pool(n_cores) as pool:
        for i, theta_b in pool.imap_unordered(
            bootstrap_iteration, range(startIteration, B)
        ):
            theta_boot[i, :] = theta_b
            if _HAS_PROGRESSBAR:
                bar.update(i - startIteration + 1)
            if checkpoint_filename:
                with open(checkpoint_filename, 'wb') as f:
                    pickle.dump(theta_boot, f)

    if _HAS_PROGRESSBAR:
        bar.finish()

    print("Compiling results...")
    se_boot = np.std(theta_boot, ddof=1, axis=0)
    alpha = options.get('alpha_significance', 0.05)
    conInt_alpha = truncated_t_ci(theta_point, se_boot, B - 1, alpha)
    conIntBCa_alpha, z0_vec, a_vec = calc_BCa_CI(theta_point, theta_boot, alpha)

    return {
        'theta_est': theta_point,
        'bootEstimates': theta_boot,
        'standardError': se_boot,
        'confidenceInterval_studentt': conInt_alpha,
        'confidenceInterval_BCa': conIntBCa_alpha,
        'BCa_biasAccel': [z0_vec, a_vec],
        'significanceLevel': alpha,
    }
