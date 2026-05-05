### LIBRARY ###
import pandas as pd
import numpy as np
import random as rand
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.linalg import expm

from multiprocess import Pool
from functools import partial

import progressbar

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
    
    for i in range(k):    
        for j in range(k):
            # biological constraint - no moving backwards!
            if i > j:
                Q_0[i,j] = 0
                
            elif i != j:
                Q_0[i,j] = rand.uniform(1e-12,ub)
              
        # sum our row
        sum_q_ij = sum(Q_0[i,])
    
        # assign ii based on the constraint that the row must equal 0
        Q_0[i,i] = -sum_q_ij

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
# Calculate minimisation of S1
def calc_cost(theta, pi_hat, T, k, u, Q_template = np.array(None), stop_region = 1e5):
    cost = np.zeros(1, dtype = 'float128')
    
    for l in range(1,T):
        u_l = u[l-1]
        
        # pi_hat @ t = l w\out last element
        pi_hat_star_l = pi_hat[:-1, l]
        
        # pi_hat @ t = l - 1 
        pi_hat_lm1 = pi_hat[:, (l-1)]
        
        # Q from thetaoverflow encountered in matmul
        Q = theta_to_Q(theta, k, Q_template)
        
        # calculate P_l
        P_l = expm(Q*u_l)
        P1_l = np.delete(P_l, -1, 1)
        
        
        cost_1 = (pi_hat_star_l - P1_l.transpose() @ pi_hat_lm1)
        
        # avoid inf values, this can be set based on range of problem
        if np.linalg.norm(cost_1,np.inf) > stop_region:
            cost = stop_region
            break
        elif np.isnan(cost_1).any() == True:
            cost = stop_region
            break
        
        cost_l = cost_1.transpose() @ cost_1

        if np.iscomplexobj(cost_l):
            cost = stop_region
            break
        
        cost += cost_l

        
    return cost

# Calculate minimisation of S1 with donor consideration
def calc_cost_donors(theta, pi_hat_donors, T_donors, k, u_donors, Q_template = np.array(None), stop_region = 1e5):
    cost = np.zeros(1, dtype = 'float128')
    
    T_max = max(T_donors.values())
    
    donors = list(pi_hat_donors.keys())
    number_donors = len(donors)
    
    # the M,N,T values are dictionaries over the donors
    for d in range(number_donors):
        pi_hat = pi_hat_donors[donors[d]]
        T = T_donors[donors[d]]
        u = u_donors[donors[d]]
        
        cost_donor = calc_cost(theta, pi_hat, T, k, u, Q_template, stop_region)
        
        # re-scale to the maximum number of time points collected.
        cost_donor = cost_donor / T * T_max
        
        cost += cost_donor
        
    return cost

def single_mc_optimiser(ingressInitial=0, hessian = None, xtol=1e-15, gtol = 1e-15, maxiter=1e4):
    
    # define variables from global set in parallel_mc_optimiser
    func = optimiserArgs['costFunc'] 
    args = optimiserArgs['args'] 
    bounds = optimiserArgs['bounds']
    constraints = optimiserArgs['constraints']
    sampRange_ub = optimiserArgs['sampRange_ub']
    ingressAccount = optimiserArgs['ingressAccount']
    
    thetaFun = theta_generator
    
    xtol = options['xterm']
    gtol = options['gterm']
    maxiter = options['max_iter']
       
    
    def applyFun(fn, **kwargs):
        return fn(**kwargs)

    # sample theta_0
    theta_0 = applyFun(thetaFun, k = args[2], ub = sampRange_ub, options = {'ingress': ingressAccount, 'initial': ingressInitial})  # this position matches with requirement for calc_S1
    
    # minimise
    cost = minimize(func, theta_0, args = args, bounds=bounds, constraints=constraints, hess=hessian, method='trust-constr', options={'verbose': 0, 'initial_tr_radius': 1e3, 'initial_constr_penalty': 1e2, 'xtol': xtol,'gtol': gtol, 'maxiter': maxiter} )
    
    cost_min = cost.fun
    theta_est = cost.x
    
    return theta_est, cost_min 


def oneCore_mc_optimiser(iter_samp, n_param, theta_generator = make_theta0, optimiserArgs = {'costFunc': None, 'args': None, 'bounds': None, 'constraints': None, 'sampRange_ub': None, 'ingressAccount': False}, options = {'initial': 1e0, 'hessian': None, 'xterm': 1e-8, 'gterm': 1e-15, 'max_iter': 1e4}, progress_bar = False):

    # set seed
    planting(456)
    
    # accounting for ingress
    if optimiserArgs.get('ingressAccount') == None:
        optimiserArgs['ingressAccount'] = False
        
    # gtol termination criteria
    if options.get('gterm') == None:
        options['gterm'] = 1e-15

    initGlobal(optimiserArgs, theta_generator, options)
    
    # set max
    costFunc_min = np.inf
    
    # track optimisation progress
    if progress_bar==True:
        i=0
        bar = progressbar.ProgressBar(maxval=iter_samp, \
        widgets=['Optimisation progress: ',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    # execute tasks in order
    for est, cost in map(single_mc_optimiser, range(iter_samp)):
        if progress_bar==True:
            i+=1
            bar.update(i)

        if cost < costFunc_min:
            theta_min = est
            costFunc_min = cost
    
    if progress_bar==True:
        bar.finish()         
    
    return theta_min, costFunc_min

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

def parallel_mc_optimiser(iter_samp, n_cores, n_param, theta_generator = make_theta0, optimiserArgs = {'costFunc': None, 'args': None, 'bounds': None, 'constraints': None, 'sampRange_ub': None, 'ingressAccount': False}, options = {'initial': 1e0, 'hessian': None, 'xterm': 1e-8, 'gterm': 1e-15, 'max_iter': 1e4}, progress_bar = False):

    # set seed
    planting(456)
    
    # accounting for ingress
    if optimiserArgs.get('ingressAccount') == None:
        optimiserArgs['ingressAccount'] = False
        
    # gtol termination criteria
    if options.get('gterm') == None:
        options['gterm'] = 1e-15
    
    # number of tasks per cpu
    chunk_size = -(-iter_samp//n_cores)
    
    # initialiser args
    initArgs = (optimiserArgs, theta_generator, options)
    
    # iterations
    iterate = range(iter_samp)
    
    # set max
    costFunc_min = np.inf

    # track optimisation progress
    if progress_bar==True:
        i=0
        bar = progressbar.ProgressBar(maxval=iter_samp, \
        widgets=['Optimisation progress: ',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    # create a configure process pool
    with Pool(processes = n_cores, initializer = initGlobal, initargs = initArgs) as pool:
        # execute tasks in order
        for est, cost in pool.imap_unordered(single_mc_optimiser, iterate):
            if progress_bar==True:
                i+=1
                bar.update(i)

            if cost < costFunc_min:
                theta_min = est
                costFunc_min = cost
        pool.close()
        
    if progress_bar==True:
        bar.finish()         
    
    return theta_min, costFunc_min



# -------------------------------------------------------------------------------------------------------------------------
# input dictionary with proportions and donor_list and output new dictionary
def residCalc_perDonor(dataArgTuple, theta_point, donor_list):
    pi_hat_donors = dataArgTuple[0]
    T_donors = dataArgTuple[1]
    k = dataArgTuple[2]
    u_donors = dataArgTuple[3]
    Q_template = dataArgTuple[4]

    for d in donor_list:
        pi_hat = pi_hat_donors[d]
        T = T_donors[d]
        u = u_donors[d]
        
        Q_est = theta_to_Q(theta_point, k, Q_template)
        resid_dict = {d: np.zeros((T-1,(k-1))) for d in donor_list}
        
        for i in range(1,T):
            P_est = transMat(Q_est, u[i-1])
            epsi = pi_hat[:,i] - np.matmul(P_est.transpose(),pi_hat[:,(i-1)])
            resid_dict[d][(i-1),:] = epsi[:-1]
            
    return resid_dict
        
def residResample_perDonor(dataArgTuple, resid_dict, donor_list):
    pi_hat_donors = dataArgTuple[0]
    T_donors = dataArgTuple[1]
    k = dataArgTuple[2]
    
    residSampling = lambda d, state: np.random.choice(resid_dict[d][:,state], replace=True)
    prop_boot_dict = {d: np.zeros(pi_hat_donors[d].shape) for d in donor_list}

    for d in donor_list:
        prop_boot_dict[d][:,0] = pi_hat[:,0]
        pi_hat = pi_hat_donors[d]
        T = T_donors[d]
        for t in range(1,T):
            sampRes = list( map( residSampling, d, range(k-1)) )
            
            # add residuals to proportion for (k-1) states
            prop_boot_dict[d][:(k-1),t] = pi_hat[:(k-1),t] + sampRes
            
            # make sure final state has column sum to 1
            prop_boot_dict[d][(k-1),t] = 1 - prop_boot_dict[:(k-1),t].sum()
             
    return prop_boot_dict
        
        
## NON-PARAMETRIC: RESAMPLING RESIDUALS
def bootstrap_resResamp_perDonor(point_estimate = None, time_interval_all = None, n_bootstrapSamples = None, n_param = None, theta_generator = None, mcArgs = None, optimiserArgs = None, options={'alpha_significance': 0.05}, Q_template = np.array(None)):
    # initialise
    B = n_bootstrapSamples
    theta_boot = np.zeros((B,n_param))
    prop_boot = np.zeros(data.shape)
    se_boot = np.zeros(B)   
    
    # first estimation
    if point_estimate.any() == None:
        ## POINT ESTIMATION
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options, progress_bar = True)
        
    else:
        theta_point = point_estimate
    
    # get donor list
    donor_list = optimiserArgs['args'][0].keys()
    resid_dict = residCalc_perDonor(optimiserArgs['args'], theta_point, donor_list)

    # track bootstrap progress
    bar = progressbar.ProgressBar(maxval=B, \
    widgets=['Bootstrap progress:',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for i in range(B):
        ## RESAMPLING
        prop_boot_donors = residResample_perDonor(optimiserArgs['args'], resid_dict, donor_list)

        # new values for boot. sample      
        optimiserArgs['args'] = (prop_boot_donors, T_donors, k, u_donors, Q_template)
                
        ## ESTIMATE
        theta_b, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options)

        # save out value
        theta_boot[i,:] = theta_b
        
        bar.update(i+1)
    
    bar.finish()
    print("Compiling results...")
    
    ## STANDARD ERROR
    theta_star = theta_boot.sum(axis=0) / B
     
    se_boot = np.sqrt( np.sum( ( theta_boot - theta_star ) ** 2 , axis=0)  * (B - 1)**(-1)  )

    ## CONFIDENCE INTERVAL
    if options.get('alpha.significance') == None:
        alpha = 0.05
    else:
        alpha = options['alpha_significance']
    conInt_alpha = np.quantile(theta_boot,[alpha/2,1-alpha/2], axis=0)
    
    # also output theta_boot to test whether the mean of boot approaches the mean of the samplw
    bootstrapOutput = {'theta_est': theta_point, 'bootEstimates': theta_boot, 'standardError': se_boot, 'confidenceInterval': conInt_alpha, 'significanceLevel': alpha}
    return bootstrapOutput

##### PARALLELISE BOOTSTRAP #####

# residual resampling single iter function
def func_resampRes_est(n_iter,dataArgs, time_interval_all, residSampling, prop_boot, optimiserArgs, mcArgs, n_param,theta_generator, options):
    pi_hat, T, k = dataArgs
    u = time_interval_all
    
    ## RESAMPLING
    for t in range(1,T):
        sampRes = list( map( residSampling, range(k-1)) )
            
        # add residuals to proportion for (k-1) states
        prop_boot[:(k-1),t] = pi_hat[:(k-1),t] + sampRes
            
        # make sure final state has column sum to 1
        prop_boot[(k-1),t] = 1 - prop_boot[:(k-1),t].sum()

    # new values for boot. sample      
    pi_hat, T, k = eqTin(prop_boot, proportion=False)
    optimiserArgs['args'] = (pi_hat, T, k, u)
                
    ## ESTIMATE
    theta_b, _cost = oneCore_mc_optimiser(iter_samp=mcArgs['mciter'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options)
        
    return theta_b


## NON-PARAMETRIC: RESAMPLING RESIDUALS
def parallel_bootstrap_resResamp(data, point_estimate = None, time_interval_all = None, n_bootstrapSamples = None, n_param = None, theta_generator = None, mcArgs = None, optimiserArgs = None, options={'alpha_significance': 0.05}, Q_template = np.array(None), perDonorBS_Fun = None):
    
    # number of bootstrap samples
    B = n_bootstrapSamples
    
    # create array to save bootstrap estimates and bootstrap proportion
    theta_boot = np.zeros((B,n_param))
    prop_boot = np.zeros(data.shape)
    
    # create array for standard error calculation
    se_boot = np.zeros(B)   
        
    # first estimation
    if point_estimate.any() == None:
        ## POINT ESTIMATION
        print("Point estimate calculation ...")
        theta_point, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options, progress_bar = True)
        
    else:
        theta_point = point_estimate
    
    # set first time point as data from real dataset
    pi_hat, T, k = eqTin(data, proportion=False)
    u = time_interval_all
    prop_boot[:,0] = pi_hat[:,0]
    
    # residuals
    Q_est = theta_to_Q(theta_point, k, Q_template)
    resid = np.zeros((T-1,(k-1)))
    for i in range(1,T):
        P_est = transMat(Q_est, u[i-1])
        epsi = pi_hat[:,i] - np.matmul(P_est.transpose(),pi_hat[:,(i-1)])
        resid[(i-1),:] = epsi[:-1]
        
    residSampling = lambda state: np.random.choice(resid[:,state], replace=True)
    
    if perDonorBS_Fun != None:
        optimiserArgs['costFunc'] = perDonorBS_Fun
    
    # track bootstrap progress
    i = 0
    j = 0
    bar = progressbar.ProgressBar(maxval=B, \
    widgets=['Bootstrap progress:',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    bootArgs = {'dataArgs': (pi_hat, T, k), 'time_interval_all': time_interval_all, 'residSampling': residSampling, 'prop_boot': prop_boot, 'optimiserArgs': optimiserArgs, 'mcArgs': mcArgs, 'n_param': n_param,'theta_generator': theta_generator, 'options': options}
    
    # configure process pool
    n_cores = mcArgs['n_cores']
    with Pool(n_cores) as pool:
        # execute tasks in order
        for theta_b in pool.imap(partial(func_resampRes_est, dataArgs=bootArgs['dataArgs'], time_interval_all=bootArgs['time_interval_all'], residSampling=bootArgs['residSampling'], prop_boot=bootArgs['prop_boot'], optimiserArgs=bootArgs['optimiserArgs'], mcArgs=bootArgs['mcArgs'], n_param=bootArgs['n_param'],theta_generator=bootArgs['theta_generator'], options=bootArgs['options']), range(B)):
            theta_boot[j,:] = theta_b
            j+=1
            
            i+=1
            bar.update(i)
    

    bar.finish() 
    
    print("Compiling results...")
    
    ## STANDARD ERROR
    theta_star = theta_boot.sum(axis=0) / B
     
    se_boot = np.sqrt( np.sum( ( theta_boot - theta_star ) ** 2 , axis=0)  * (B - 1)**(-1)  )

    ## CONFIDENCE INTERVAL
    if options.get('alpha.significance') == None:
        alpha = 0.05
    else:
        alpha = options['alpha_significance']
    conInt_alpha = np.quantile(theta_boot,[alpha/2,1-alpha/2], axis=0)
    
    # also output theta_boot to test whether the mean of boot approaches the mean of the samplw
    bootstrapOutput = {'theta_est': theta_point, 'cost_sol': cost_point,'bootEstimates': theta_boot, 'standardError': se_boot, 'confidenceInterval': conInt_alpha, 'significanceLevel': alpha}
    return bootstrapOutput

