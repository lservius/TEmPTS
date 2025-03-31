### LIBRARY ###
import pandas as pd
import numpy as np
import random as rand
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.linalg import expm
from scipy.stats import truncnorm, t, norm, skew
from math import ceil


from multiprocess import Pool
from functools import partial

import progressbar
import pickle

# suppress warning from minimise
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

### ---------------------------------------------------------------------------------------------------

## REPRODUCIBILITY
def planting(s):
    np.random.seed(s)
    rand.seed(s)


## BASELINE CALCULATIONS
def theta_to_Q(theta_u, n_states, Q_template = np.array(None)):
  
    R = len(np.nonzero(Q_template[:,:-1])[0])
    
    Q_new = np.zeros((n_states, n_states))
    
    Q_to_theta = np.nonzero(Q_template[:,:-1])
    
    for r in range(R):
        i = Q_to_theta[0][r]
        j = Q_to_theta[1][r]
        
        Q_new[i,j] = theta_u[r]
    
    for r in range(n_states):
        Q_new[r, -1] = -sum(Q_new[r,:(n_states - 1)])
            
    return(Q_new)
        


# Q to P
def transMat(Q, u):
    P_l = expm(Q*u)
    return(P_l)


## SIMULATION FUNCTIONS
# create a function to generate initial state counts
def createN0(n_states, startCount, state_names = 'None'):
    prop = np.zeros(n_states)
    mag = int(np.log10(startCount))
    
    for i in range(n_states):
        if i == 0:
            prop[i] = round(rand.uniform(.2,1),mag)
        elif i < (n_states-1):
            prop[i] = round(rand.uniform(0,1 - sum(prop)),mag)
        else:
            prop[i] = 1 - sum(prop)
    
    # setting the proportions for the initial N, want to then multiple by startCount to generate
    vals = startCount * prop
    stateCount = np.round(vals)
    
    if state_names == 'None':
        state_names = [*range(n_states)]
    df = pd.DataFrame({'d0': stateCount}, index = state_names)
    
    return df

# create the name of the states
def statNam(n_states):
    names = []
    for i in range(n_states):
        names.append('State ' + str(i + 1))
    return names

# actually gennerate the aggregate states
def generateAggStates(N0, Q_ij, n_generate, u = [1], recruit = True, timeCount = int(), timeMarker = 'd', noise = False):
    df = N0.copy()
     
    # set the first date as reference
    firstDay = timeMarker + str(timeCount)
    
    all_one_interval = all([u[i] == 1 for i in range(len(u))])
    
    if all_one_interval == True:
       P_ij = transMat(Q_ij, 1)
    
    iter = 1
    while iter <= n_generate: 
        
        if all_one_interval == True:
            # name the day
            today = timeMarker + str(timeCount)
            tomorrow = timeMarker + str(timeCount + u[0])
            timeCount += 1
            
        elif all_one_interval == False:
            # calculate and P
            P_ij = transMat(Q_ij, u[iter])

            today = timeMarker + str(timeCount)
            tomorrow = timeMarker + str(timeCount + u[iter])
            timeCount += u[iter]
            
    
        # final column
        initial_values = df[today]
        
        # next day
        d1 = []
    
        # number of states
        n_state = len(P_ij)
        
        # save out d0 values as array
        d0 = initial_values
    
        for j in range(n_state):
            
            # select related probability column
            P_i = P_ij[:,j]
            
            # dot product
            dotProd = sum( d0 * P_i )
            
            # only want integer value
            dotProd = int(dotProd)
            
            d1.append(dotProd)
            
            if recruit == True:
                # add recruits to first state
                recLam = int(np.log10(sum(df[firstDay])) - 2 )
                d1[0] = int(d1[0] + np.random.poisson(10**recLam,1))
                
        # add to data
        df[tomorrow] = d1
        
        if noise == True:
            # add 1% noise but prevent negative values
            scale_noise = df[tomorrow] * 0.01
            df[tomorrow] += np.random.poisson(lam=scale_noise)
            df.loc[df[tomorrow] < 0, tomorrow] = 0
                    
        iter += 1
        
    
    return df

# generate the parameters for simulation or estimation seeds
def make_theta0(k, format = 'theta', ub = 1e-1, options = {'ingress': False, 'initial': 1e0}):
    # create starting intensity trans. mat
    Q_0 = np.zeros((k,k))
    
    Q_0 = np.random.uniform(1e-12, ub, size=(k,k))
    Q_0 = np.triu(Q_0) # biological constraint - no moving backwards!
    
    # set diag. to neg. sum rest of row
    row_sums = Q_0.sum(axis=1)
    np.fill_diagonal(Q_0, -row_sums)
    
    if format == 'Q':
        result = Q_0
    elif format == 'theta':
        theta_0 = Q_0[np.nonzero(Q_0[:,:-1])]
        result = theta_0
        
        if options["ingress"] == True:
            income = options['initial']
            result = np.append(theta_0, income)
    
    return result

## PREPARE DATA

def formatCount(data, day_series, timepoint_col_name, state_col_name, timemarker, isotype_list, proportion_col_name=None):
    ISO = pd.DataFrame()
    
    # isolate isotypes of interest
    data[data[state_col_name].isin(isotype_list)]
    
    # if the data is already proportions
    if proportion_col_name != None:
        ISO = data.pivot_table(values=proportion_col_name, index=state_col_name, columns=timepoint_col_name)
        ISO = ISO.reindex(isotype_list)
        ISO = ISO.replace(np.nan, 0)
        return(ISO)
   
    for d in range(len(day_series)):
        # selecting the day
        day = day_series[d]
        day_name = timemarker + str(day)
        data_i = data[data[timepoint_col_name] == day]
        
        # creating the per class counts
        data_i = data_i[state_col_name].value_counts()
        
        # turn per class counts into proportions
        total = data_i.sum()
        data_i = pd.Series.to_frame(data_i / total)
        
        ISO[day_name] = data_i
   
    ISO = ISO.reindex(isotype_list)
    ISO = ISO.replace(np.nan, 0)
    
    return(ISO)

# proportion the state counts
def eqTin(df, proportion = True, giveK = False, perDonor = False, donor_list = [], donor_column = '', time_column = '', state_column = '', state_list = []):
    T = df.shape[1]
    k = df.shape[0]
    
    input_pi_hat = df.copy()
    pi_hat = np.zeros(input_pi_hat.shape)
    
    if perDonor == True:
        pi_hat_donors = {}
        T_donors = {}
        u_donors = {}
        donors = donor_list
        timeCourse = {}

        for d in donors:
    
            # donor filter
            df_donor = df.loc[df[donor_column] == d, :]
            
            # extract the sampled time points
            timepoint_list = np.sort(df_donor[time_column].unique())
            timeCourse[d] = timepoint_list.shape[0] 
            
            # transform into state count
            df_donor = formatCount(df_donor, timepoint_list, time_column, state_column, timemarker='d', isotype_list=state_list)

            # transform into nump array for calculations
            df_donor = df_donor.to_numpy()
            
            pi_hat, T, k = eqTin(df_donor, proportion=False, giveK=True)
            
            pi_hat_donors[d] = pi_hat
            T_donors[d] = T
            u_donors[d] = np.diff(timepoint_list)
        return pi_hat_donors, T_donors, k, u_donors
        
    if proportion == False:
        M = np.delete(input_pi_hat, -1,0)  
        
        return input_pi_hat, T, k
    
    for t in range(T):        
        pi_hat_t = input_pi_hat[:,t]
        
        # calculate the total
        sum_t = sum(pi_hat_t)
        
        pi_hat_t = pi_hat_t / sum_t
        
        pi_hat[:,t] = pi_hat_t
    
    if giveK == True:
        return pi_hat, T, k 
    
   
    return pi_hat, T


## BOUNDS AND CONSTRAINTS FOR MINIMISATION
# calculate bounds
def def_bounds(n_param, k, account_ingress = False, serial_corr = False):
    q_lb = np.zeros(n_param)
    q_ub = np.zeros(n_param)

    q_diag = np.zeros((k-1))

    a = -1/2
    b = (k-2) - 3*a
    c = 0
    
    for n in range((k-1)):
        q_diag[n] = a * n**2 + b * n + c
    
    for n in range(n_param):
        if n in q_diag:
            q_lb[n] = -np.inf
            q_ub[n] = 0
        else:
            q_lb[n] = 0 
            q_ub[n] = np.inf

    reg_bounds = Bounds(q_lb, q_ub)
    
    if account_ingress == True:
        # bounds for ingress
        q_lb[-1] = -1
        q_ub[-1] = 1
        
        alt_bounds = Bounds(q_lb, q_ub)
        return alt_bounds
    
    if serial_corr == True:
        # bounds for serial correlation
        q_lb[-1] = 0
        q_ub[-1] = 1
        
        alt_bounds = Bounds(q_lb, q_ub)
        return alt_bounds
    
    return reg_bounds

# calculate linear constraints
def def_constraints(n_param, k):
    row_eq = np.zeros((((k-1),n_param)))
    
    a = -1/2
    b = (k-2) - 3*a
    c = 0
    
    for n in range((k-1)):
        i = int(a * n**2 + b * n + c)
        j = int(a * (n + 1)**2 + b * (n + 1) + c)
    
        row_eq[n,i:j] = 1
    
    lin_con = LinearConstraint(row_eq, ub = np.zeros((k-1)), keep_feasible=True)
    
    return lin_con 
     
     
## COST FUNCTIONS
def calc_cost(theta, pi_hat, T, k, u, Q_template = np.array(None), stop_region = 1e5):
    cost = 0.
    
    # Q from theta
    Q = theta_to_Q(theta, k, Q_template)
        
    for l in range(1,T):
        u_l = u[l-1]
        
        # pi_hat @ t = l w\out last element
        pi_hat_star_l = pi_hat[:-1, l]
        
        # pi_hat @ t = l - 1 
        pi_hat_lm1 = pi_hat[:, (l-1)]

        # calculate P_l
        P_l = expm(Q*u_l)
        
        if np.isnan(P_l).any() or np.isinf(P_l).any():
            return np.array(stop_region)
        
        P1_l = P_l[:, :-1]
        
        cost_1 = pi_hat_star_l - (P1_l.T @ pi_hat_lm1)
        norm_cost_1 = np.linalg.norm(cost_1, np.inf)
        
        # avoid inf values, this can be set based on range of problem
        if norm_cost_1 > stop_region or np.isnan(cost_1).any():
            return np.array(stop_region)
        
        cost_l = cost_1.T @ cost_1

        if np.iscomplexobj(cost_l):  
            return np.array(stop_region)
        
        cost += cost_l

        
    return cost

# Calculate cost with donor consideration
def calc_cost_donors(theta, pi_hat_donors, T_donors, k, u_donors, Q_template = np.array(None), stop_region = 1e5):
    cost = 0.
    
    T_max = max(T_donors.values())
    
    # the values per donor
    for d in pi_hat_donors.keys():
        pi_hat = pi_hat_donors[d]
        T = T_donors[d]
        u = u_donors[d]
        
        # to re-scale to the maximum number of time points collected.
        scale_factor = T_max / T
        
        cost_donor = calc_cost(theta, pi_hat, T, k, u, Q_template, stop_region) * scale_factor
        
        if cost_donor >= stop_region:
            return stop_region
        
        cost += cost_donor
        
    return cost

def single_mc_optimiser(i, ingressInitial=0, optimiserArgs=None, options=None, theta_generator=None, hessian=None):
    
    if optimiserArgs is None:
        raise ValueError("optimiserArgs cannot be None")

    if options is None:
        options = {'xterm': 1e-8, 'gterm': 1e-15, 'max_iter': 1e4}
    
    # define variables
    func = optimiserArgs.get('costFunc')
    args = optimiserArgs.get('args')
    bounds = optimiserArgs.get('bounds')
    constraints = optimiserArgs.get('constraints')
    sampRange_ub = optimiserArgs.get('sampRange_ub')
    ingressAccount = optimiserArgs.get('ingressAccount', False)

    if theta_generator is None:
        raise ValueError("theta_generator must be supplied")

      
    xtol = options.get('xterm', 1e-8)
    gtol = options.get('gterm', 1e-15)
    maxiter = options.get('max_iter', 1e4)
       
    
    def applyFun(fn, **kwargs):
        return fn(**kwargs)

    # sample theta_0
    theta_0 = theta_generator(k=args[2], ub=sampRange_ub, options={'ingress': ingressAccount, 'initial': ingressInitial})
  # this position matches with requirement for calc_cost
    
    # minimise
    try:
        # minimise using trust-region constrained method
        result = minimize(func, theta_0, args=args, bounds=bounds, constraints=constraints, hess=hessian,
                          method='trust-constr', options={'verbose': 0, 'initial_tr_radius': 1e3,
                                                          'initial_constr_penalty': 1e2, 'xtol': xtol,
                                                          'gtol': gtol, 'maxiter': maxiter})

        cost_min = result.fun
        theta_est = result.x

    except Exception as e:
        print(f"Optimization failed at ingressInitial={ingressInitial}: {e}")
        return None, np.inf
    
    return i, theta_est, cost_min 


def oneCore_mc_optimiser(iter_samp, n_param, theta_generator = make_theta0, optimiserArgs = None, options = None, progress_bar = False):
    if optimiserArgs is None:
        optimiserArgs = {'costFunc': None, 'args': None, 'bounds': None, 'constraints': None, 'sampRange_ub': None, 'ingressAccount': False}
    if options is None:
        options = {'initial': 1e0, 'hessian': None, 'xterm': 1e-8, 'gterm': 1e-15, 'max_iter': 1e4}
    
    
    # accounting for ingress
    optimiserArgs.setdefault('ingressAccount', False)
    options.setdefault('gterm', 1e-15)
    
    initGlobal(optimiserArgs, theta_generator, options)
    
    # initialise and set max
    theta_samples = np.zeros((iter_samp, n_param))
    cost_samples = np.full(iter_samp, np.inf)
    
    # track optimisation progress
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=iter_samp,
                                      widgets=['Optimisation progress: ', progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
    
    for i, est, cost in map(partial(single_mc_optimiser, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options), range(iter_samp)):
        theta_samples[i-1] = est
        cost_samples[i-1] = cost
    
    if progress_bar:
        bar.update(i)   
    
    if progress_bar:
        bar.finish()
        
    # find best estimate
    min_idx = np.argmin(cost_samples)
    
    return theta_samples[min_idx], cost_samples[min_idx]

# PARALLELISE

# make variables and packages for optimiser global
def initGlobal(optimiserArgs_dict, thetaGen_function, option_dict):
    import random as rand
    import numpy as np
    from scipy.optimize import minimize
    
    global optimiserArgs
    optimiserArgs = optimiserArgs_dict
    
    global theta_generator
    theta_generator = thetaGen_function
    
    global options
    options = option_dict

def parallel_mc_optimiser(iter_samp, n_cores, n_param, theta_generator = make_theta0, optimiserArgs = None, options = None, progress_bar = False):
    
    # default values for optimiserArgs and options
    if optimiserArgs is None:
        optimiserArgs = {'costFunc': None, 'args': None, 'bounds': None, 
                         'constraints': None, 'sampRange_ub': None, 'ingressAccount': False}
    
    if options is None:
        options = {'initial': 1e0, 'hessian': None, 'xterm': 1e-8, 
                   'gterm': 1e-15, 'max_iter': 1e4}
        
    optimiserArgs.setdefault('ingressAccount', False)
    options.setdefault('gterm', 1e-15)
    
    # number of tasks per cpu
    chunk_size = ceil(iter_samp / n_cores)
    
    # initialiser args
    initArgs = (optimiserArgs, theta_generator, options)
    
    # iterations
    iterate = range(iter_samp)
    
    # Set initial cost minimum
    theta_samples = np.zeros((iter_samp, n_param))
    cost_samples = np.full(iter_samp, np.inf)
    
    mcArgs = {'optimiserArgs': optimiserArgs, 'options': options, 'theta_generator': theta_generator}
    
    
    if progress_bar:
        bar = progressbar.ProgressBar(maxval=iter_samp, redirect_stdout=True)
        bar.start()
        
    try:
        with Pool(processes=n_cores) as pool:
            for i, est, cost in pool.imap(partial(single_mc_optimiser, optimiserArgs=mcArgs['optimiserArgs'], options=mcArgs['options'], theta_generator=mcArgs['theta_generator']), iterate):
                if progress_bar:
                    bar.update(bar.currval + 1)

                theta_samples[i-1] = est
                cost_samples[i-1] = cost 
    
    except Exception as e:
        print(f"Error in parallel execution: {e}")
        
    if progress_bar:
        bar.finish()
    # find best estimate
    min_idx = np.argmin(cost_samples)
    
    return theta_samples[min_idx], cost_samples[min_idx]


# -------------------------------------------------------------------------------------------------------------------------
# input dictionary with proportions and donor_list and list of residuals
def residCalc(dataArgTuple, theta_point):
    pi_hat_donors, T_donors, k, u_donors, Q_template = dataArgTuple
    donor_list = pi_hat_donors.keys()
    
    Q_est = theta_to_Q(theta_point, k, Q_template)
    resid_list = []
    
    for d in donor_list:
        pi_hat = pi_hat_donors[d]
        T = T_donors[d]
        u = u_donors[d]
        
        for i in range(1,T):
            P_est = transMat(Q_est, u[i-1])
            epsi = pi_hat[:(k-1), i] - np.matmul(P_est[:, :(k-1)].T, pi_hat[:, (i-1)])
            
            resid_list.append(epsi)
    
    # stack into an array
    resid_lib = np.column_stack(resid_list) if resid_list else np.empty((k-1, 0))
    
    return resid_lib

def calc_donorSD(pi_hat_donors, k, timepoint=0):
    donor_no = len(pi_hat_donors)
    
    if donor_no == 0:
        return np.zeros(k)
    
    time_pi_vec = np.column_stack([pi_hat_donors[d][:, timepoint] for d in pi_hat_donors])

    donorSD_t = np.std(time_pi_vec, axis=1, ddof=1)
    
    return donorSD_t


def residResample_perDonor(i, dataArgTuple, point_est, resid_lib, t0_donorSD):
    pi_hat_donors, T_donors, k, u_donors, Q_template = dataArgTuple
    
    prop_boot_dict = {d: np.zeros_like(pi_hat) for d, pi_hat in pi_hat_donors.items()}
    
    Q_est = theta_to_Q(point_est, k, Q_template)
    
    for d, pi_hat in pi_hat_donors.items():
        T = T_donors[d]
        u = u_donors[d]
        
        # first time point is sampled from a truncated normal distribution with the mean as the donor value and variance as the variance over all donors
        a, b = (0. - pi_hat[:(k-1), 0]) / t0_donorSD, (1. - pi_hat[:(k-1), 0]) / t0_donorSD
        prop_boot_dict[d][:(k-1), 0] = truncnorm.rvs(a, b, loc=pi_hat[:(k-1), 0], scale=t0_donorSD)
        
        prop_boot_dict[d][(k-1), 0] = 1 - prop_boot_dict[d][:(k-1), 0].sum()
        
        pi_fitted = np.zeros_like(pi_hat)
        pi_fitted[:, 0] = prop_boot_dict[d][:, 0]
        
        for t in range(1,T):
            # fitted proportions
            P_est = transMat(Q_est,u[(t-1)])
            
            pi_fitted[:(k-1), t] = P_est[:, :(k-1)].T @ pi_fitted[:, t-1]
            
            sampRes = np.apply_along_axis(np.random.choice, 1, resid_lib)
            
            # add residuals to proportion for (k-1) states
            prop_boot_dict[d][:(k-1), t] = pi_fitted[:(k-1), t] + sampRes
            
            # make sure final state has column sum to 1
            prop_boot_dict[d][(k-1), t] = max(0, 1 - prop_boot_dict[d][:(k-1), t].sum())
             
    return prop_boot_dict


# BOOTSTRAP-t CONFIDENCE INTERVAL CALCULATION
def truncated_t_ci(theta_point, std_error, dof, alpha=0.05):
    bounds = np.zeros((len(theta_point), 2))
    for i in range(len(theta_point)):
        if theta_point[i] < 0:
            bounds[i,1] = 0
            bounds[i,0] = -np.inf
        else:
            bounds[i,1] = np.inf
            bounds[i,0] = 0
            
    ci_matrix = np.zeros((len(theta_point), 2))
    
    for idx, point_estimate in enumerate(theta_point):
         # Standard t-distribution percentiles
        t_alpha = t.ppf(1 - alpha / 2, dof)  # Upper critical value
        lower_ci = point_estimate - t_alpha * std_error[idx]
        upper_ci = point_estimate + t_alpha * std_error[idx]

        # Truncate the confidence interval
        ci_matrix[idx, 0] = max(bounds[idx,0], lower_ci)  # Apply lower truncation
        ci_matrix[idx, 1] = min(bounds[idx,1], upper_ci)  # Apply upper truncation

    return ci_matrix
             

# BCa CONFIDENCE INTERVAL CALCULATION
def calc_BCa_CI(theta_point, theta_boot, alpha):
    z0_vec = np.zeros(len(theta_point))
    a_vec = np.zeros(len(theta_point))

    ci_matrix = np.zeros((len(theta_point), 2))
    for idx, point_estimate in enumerate(theta_point):
            
        # Extract bootstrap samples for the current point estimate
        theta_boot_i = theta_boot[:, idx]
        point_estimate_value = point_estimate
        
        # Bias correction (z0)
        z0 = np.mean(theta_boot_i < point_estimate_value)
        
        # prevent inf. values
        if z0 == 1:
            z0 = .999999    
        if z0 == 0:
            z0 = .000001    
        z0 = norm.ppf(z0)
        
        # Acceleration (a) using skewness approximation
        a = skew(theta_boot_i, axis=0, bias=False) / 6
        
        # Save out values to sense check
        z0_vec[idx] = z0
        a_vec[idx] = a
        
        # Adjusted percentiles
        z_alpha_lower = norm.ppf(alpha / 2)
        z_alpha_upper = norm.ppf(1 - alpha / 2)
        adj_alpha_lower = norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        adj_alpha_upper = norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
        
        # Confidence intervals
        ci_matrix[idx, 0] = np.quantile(theta_boot_i, adj_alpha_lower)
        ci_matrix[idx, 1] = np.quantile(theta_boot_i, adj_alpha_upper)

    return ci_matrix, z0_vec, a_vec       
        
## NON-PARAMETRIC: RESAMPLING RESIDUALS
def bootstrap_resResamp_perDonor(point_estimate = None, n_bootstrapSamples = None, n_param = None, theta_generator = None, mcArgs = None, optimiserArgs = None, options={'alpha_significance': 0.05}, startIteration = 0, checkpoint_filename = ''):
    # initialise
    B = n_bootstrapSamples
    theta_boot = np.zeros((B,n_param))
    se_boot = np.zeros(B)   
    
    # first estimation
    if point_estimate is None or point_estimate.size == 0:
        ## POINT ESTIMATION
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options, progress_bar = True)
        
    else:
        theta_point = point_estimate
    
    # get donor list and values
    pi_hat_donors, T_donors, k, u_donors, Q_template = optimiserArgs['args']
    resid_dict = residCalc(optimiserArgs['args'], theta_point)

    # calculate donorSD for resid resampling starting proportion
    t0_donorSD = calc_donorSD(pi_hat_donors, k, timepoint=0)[:(k-1)]
    
    # load result file
    try:
        with open(checkpoint_filename, 'rb') as f:
            theta_boot = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("Checkpoint file not found or corrupted. Starting fresh.")


    # track bootstrap progress
    
    for i in progressbar.progressbar(range(startIteration, B), redirect_stdout=True):
        ## RESAMPLING
        prop_boot_donors = residResample_perDonor(i, optimiserArgs['args'], point_estimate, resid_dict, t0_donorSD)

        # new values for boot. sample      
        optimiserArgs['args'] = (prop_boot_donors, T_donors, k, u_donors, Q_template)
                
        ## ESTIMATE
        theta_b, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options)

        # save out value
        theta_boot[i,:] = theta_b
        
        # checkpoint
        with open(checkpoint_filename, 'wb') as f:
            pickle.dump(theta_boot,f)
        
    
    print("Compiling results...")
    
    ## STANDARD ERROR     
    se_boot = np.std(theta_boot, ddof = 1, axis = 0)

    ## CONFIDENCE INTERVAL
    alpha = options.get('alpha_significance', 0.05)
    
    conInt_alpha = truncated_t_ci(theta_point, se_boot, B - 1, alpha)
    
    ### BCa CI estimation
    conIntBCa_alpha, z0_vec, a_vec = calc_BCa_CI(theta_point, theta_boot, alpha)

    # also output theta_boot to test whether the mean of boot approaches the mean of the samplw
    bootstrapOutput = {'theta_est': theta_point,'bootEstimates': theta_boot, 'standardError': se_boot, 'confidenceInterval_studentt': conInt_alpha, 'confidenceInterval_BCa': conIntBCa_alpha, 'BCa_biasAccel':[z0_vec, a_vec], 'significanceLevel': alpha}
    return bootstrapOutput


def bootstrap_resResamp_perDonor_oneiter(point_estimate = None, n_bootstrapSamples = None, n_param = None, theta_generator = None, mcArgs = None, optimiserArgs = None, options={'alpha_significance': 0.05}, startIteration = 0, checkpoint_filename = ''):
    # first estimation
    if point_estimate is None or point_estimate.size == 0:
        ## POINT ESTIMATION
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options, progress_bar = True)
        
    else:
        theta_point = point_estimate
    
    # get donor list and values
    pi_hat_donors, T_donors, k, u_donors, Q_template = optimiserArgs['args']
    resid_dict = residCalc(optimiserArgs['args'], theta_point)

    # calculate donorSD for resid resampling starting proportion
    t0_donorSD = calc_donorSD(pi_hat_donors, k, timepoint=0)[:(k-1)]
    

    # track bootstrap progress
    
    ## RESAMPLING
    prop_boot_donors = residResample_perDonor(startIteration, optimiserArgs['args'], point_estimate, resid_dict, t0_donorSD)

    # new values for boot. sample      
    optimiserArgs['args'] = (prop_boot_donors, T_donors, k, u_donors, Q_template)
                
    ## ESTIMATE
    theta_b, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options)

    # save out value
    theta_boot = theta_b
        
    # save out
    np.savetxt(checkpoint_filename, theta_boot, delimiter=",")
    
    print("Results saved!")
    
    return theta_boot

##### PARALLELISE BOOTSTRAP #####

# residual resampling single iter function
def func_resampRes_est(i, point_estimate, resid_dict, t0_donorSD, optimiserArgs, mcArgs, n_param, theta_generator, options):
    # get donor list and values
    donor_list = optimiserArgs['args'][0].keys()
    T_donors = optimiserArgs['args'][1]
    k = optimiserArgs['args'][2]
    u_donors = optimiserArgs['args'][3]
    Q_template = optimiserArgs['args'][4]
    
    ## RESAMPLING
    prop_boot_donors = residResample_perDonor(i, optimiserArgs['args'], point_estimate, resid_dict, donor_list, t0_donorSD)

    # new values for boot. sample      
    optimiserArgs['args'] = (prop_boot_donors, T_donors, k, u_donors, Q_template)
                
    ## ESTIMATE
    theta_b, _cost = oneCore_mc_optimiser(iter_samp=mcArgs['mciter'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options)
    
        
    return theta_b


## NON-PARAMETRIC: RESAMPLING RESIDUALS
def parallel_bootstrap_resResamp(point_estimate = None, n_bootstrapSamples = None, n_param = None, theta_generator = None, mcArgs = None, optimiserArgs = None, options={'alpha_significance': 0.05}, startIteration = 0, checkpoint_filename = ''):
    # first estimation
    if point_estimate is None or point_estimate.any() is None:
        ## POINT ESTIMATION
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options, progress_bar = True)   
    else:
        theta_point = point_estimate
        
    # initialise
    B = n_bootstrapSamples
    theta_boot = np.zeros((B,n_param))
    se_boot = np.zeros(B)   
    
    # get donor list and values
    pi_hat_donors, T_donors, k, u_donors, Q_template = optimiserArgs['args']
    resid_dict = residCalc(optimiserArgs['args'], theta_point)
    
    # calculate donorSD for resid resampling starting proportion
    t0_donorSD = calc_donorSD(pi_hat_donors, k, timepoint=0)[:(k-1)]
    
    # load result file
    try:
        with open(checkpoint_filename, 'rb') as f:
            theta_boot = pickle.load(f)
    except FileNotFoundError:
        print("Checkpoint not found. Starting fresh.")
    
    # track bootstrap progress23589492
    bar = progressbar.ProgressBar(maxval=B, min_value=startIteration)
    bar.start()
    
    # parallel BS
    def bootstrap_iteration(i):
        # resample donor proportions
        prop_boot_donors = residResample_perDonor(i, optimiserArgs['args'], theta_point, resid_dict, t0_donorSD)
        
        optimiserArgs['args'] = (prop_boot_donors, T_donors, k, u_donors, Q_template)
        
        ## ESTIMATE
        theta_b, _cost = oneCore_mc_optimiser(iter_samp=mcArgs['mciter'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options)
        
        return i, theta_b    
    
    # configure process pool
    n_cores = mcArgs['n_cores']
    with Pool(n_cores) as pool:
        # execute tasks in order
        for i, theta_b in pool.imap_unordered(bootstrap_iteration, range(startIteration, B)):
            theta_boot[i, :] = theta_b
            bar.update(i - startIteration + 1)
            
            with open(checkpoint_filename, 'wb') as f:
                    pickle.dump(theta_boot, f)

            # save checkpoint every 10 iterations
            #if i % 10 == 0:
            #    with open(checkpoint_filename, 'wb') as f:
            #        pickle.dump(theta_boot, f)

    bar.finish() 
    
    print("Compiling results...")
    
    ## STANDARD ERROR     
    se_boot = np.std(theta_boot, ddof = 1, axis = 0) 

    ## CONFIDENCE INTERVAL
    if options.get('alpha.significance') == None:
        alpha = 0.05
    else:
        alpha = options['alpha_significance']
    
    conInt_alpha = truncated_t_ci(theta_point, se_boot, B - 1, alpha)
    
     ### BCa CI estimation
    conIntBCa_alpha, z0_vec, a_vec = calc_BCa_CI(theta_point, theta_boot, alpha)

    # also output theta_boot to test whether the mean of boot approaches the mean of the samplw
    bootstrapOutput = {'theta_est': theta_point,'bootEstimates': theta_boot, 'standardError': se_boot, 'confidenceInterval_studentt': conInt_alpha, 'confidenceInterval_BCa': conIntBCa_alpha, 'BCa_biasAccel':[z0_vec, a_vec], 'significanceLevel': alpha}
    
    return bootstrapOutput