### LIBRARY ###
import pandas as pd
import numpy as np
import random as rand
from scipy import inf
from scipy.optimize import minimize, Bounds, LinearConstraint

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
    
    def check_argument():
        if Q_template.all() == None:
            return make_theta0(n_states, 'Q')
        else:
            return Q_template

    Q_template = check_argument()
    
    R = theta_u.shape[0]
    
    Q_new = np.zeros((n_states, n_states))
    
    Q_to_theta = np.nonzero(Q_template[:,:-1])
    
    for r in range(R):
        i = Q_to_theta[0][r]
        j = Q_to_theta[1][r]
        
        Q_new[i,j] = theta_u[r]
    
    for r in range(n_states):
        Q_new[r, -1] = -sum(Q_new[r,:(n_states - 1)])
    
    return(Q_new)
        
# eigendecomposition of trans. inten. mat.
def eigendecomp(Q_0):
    # get eigenvector and values
    d, A = np.linalg.eig(Q_0)
    
    # diagonalise eigenvalues
    D = np.diag(d)
    
    # calc. inverse of A
    A_inv = np.linalg.inv(A)
    
    return(A,D,d,A_inv)

# Q to P
def transMat(Q, u):
    A,D,d,A_inv = eigendecomp(Q)
    exp_dt = np.exp(d*u)
    exp_D = np.diag(exp_dt)
    
    # @ is short-hand for np.matmul
    P_l = A @ exp_D @ A_inv
    
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
    
    # sampSize and noise calc.
    totalDay1 = N0.sum()
    scale_noise = totalDay1 * 0.01
    
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
            # add 1% noise
            df[tomorrow] += int(np.random.normal(loc=0, scale=scale_noise))
        
        iter += 1
        
    
    return df

# generate the parameters for simulation or estimation seeds
def make_theta0(k, format = 'theta', ub = 2e-1, options = {'ingress': False, 'initial': 1e0}):
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
def eqTin(df, proportion = True, giveK = False):
    T = df.shape[1]
    k = df.shape[0]
    
    input_N = df.copy()
    N = np.zeros(input_N.shape)
    
    if proportion == False:
        M = np.delete(N, -1,0)  
        
        return input_N, M, T, k
    
    for t in range(T):        
        N_t = input_N[:,t]
        
        # calculate the total
        sum_t = sum(N_t)
        
        N_t = N_t / sum_t
        
        N[:,t] = N_t

    M = np.delete(N, -1,0)  
    
    if giveK == True:
        return N, M, T, k  
        
    return N, M, T


## BOUNDS AND CONSTRAINTS FOR MINIMISATION
# calculate bounds
def def_bounds(n_param, k, account_ingress = False):
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
            q_lb[n] = -inf
            q_ub[n] = 0
        else:
            q_lb[n] = 0 
            q_ub[n] = inf

    reg_bounds = Bounds(q_lb, q_ub)
    
    if account_ingress == True:
        # bounds for ingress
        q_lb[-1] = -1
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
def Lcalc_S1(theta, M, N, T, k, u, stop_region = 1e5):
    S1 = np.zeros(1, dtype = 'float128')
    
    N_hat = np.zeros((k,T))
    N_hat[:,0] = N[:, 0].copy()
    
    for l in range(1,T):
        u_l = u[l-1]
        
        M_l = M[:, l]
        
        # Q from theta
        Q = theta_to_Q(theta, k)
        
        # calculate P_l
        A,D,d,A_inv = eigendecomp(Q)
        exp_dt = np.exp(d*u_l)
        exp_D = np.diag(exp_dt)
        
        # @ is short-hand for np.matmul
        P_l = A @ exp_D @ A_inv
        P1_l = np.delete(P_l, -1, 1)
        
        # estimate proportion for timepoint l based on previous estimation
        M_hat_l = P1_l.transpose() @ N_hat[:,(l-1)]
        
        # calculate residuals
        S1_1 = (M_l - M_hat_l)
        
        # avoid inf values, this can be set based on range of problem
        if np.linalg.norm(S1_1,np.inf) > stop_region:
            S1 = stop_region
            break
        elif np.isnan(S1_1).any() == True:
            S1 = stop_region
            break
        
         
        S1_i = S1_1.transpose() @ S1_1

        if np.iscomplexobj(S1_i):
            S1 = stop_region
            break
        
        S1 += S1_i
        
        # save to N_hat for next run
        N_hat_l = np.append(M_hat_l, 1-M_hat_l.sum())
        N_hat[:,l] = N_hat_l
        
    return S1

# Calculate minimisation of S1
def calc_S1(theta, M, N, T, k, u, Q_template = np.array(None), stop_region = 1e5):
    S1 = np.zeros(1, dtype = 'float128')
    
    for l in range(1,T):
        u_l = u[l-1]
        
        N_l = N[:, l]
        
        # N_{l-1}
        N_lm1 =N[:, (l-1)]
        
        # Q from thetaoverflow encountered in matmul
        Q = theta_to_Q(theta, k, Q_template)
        
        # calculate P_l
        A,D,d,A_inv = eigendecomp(Q)
        exp_dt = np.exp(d*u_l)
        exp_D = np.diag(exp_dt)
        
        # @ is short-hand for np.matmul
        P_l = A @ exp_D @ A_inv
        #P1_l = np.delete(P_l, -1, 1)
        
        S1_1 = (N_l - P_l.transpose() @ N_lm1)
        
        # avoid inf values, this can be set based on range of problem
        if np.linalg.norm(S1_1,np.inf) > stop_region:
            S1 = stop_region
            break
        elif np.isnan(S1_1).any() == True:
            S1 = stop_region
            break
        
        S1_i = S1_1.transpose() @ S1_1

        if np.iscomplexobj(S1_i):
            S1 = stop_region
            break
        
        S1 += S1_i

        
    return S1

# Calculate minimisation of S1 with donor consideration
def calc_S1_donors(theta, M_donors, N_donors, T_donors, k, u_donors, Q_template = np.array(None), stop_region = 1e5):
    S1 = np.zeros(1, dtype = 'float128')
    
    donors = list(M_donors.keys())
    number_donors = len(donors)
    
    # the M,N,T values are dictionaries over the donors
    for d in range(number_donors):
        M = M_donors[donors[d]]
        N = N_donors[donors[d]]
        T = T_donors[donors[d]]
        u = u_donors[donors[d]]
        
        S1_donor = calc_S1(theta, M, N, T, k, u, Q_template, stop_region)
        
        # average so that it is not weighted by the number of time points sampled for each donor.
        
        S1_donor = S1_donor / T
        
        S1 += S1_donor
        
    return S1

# Calculate minimisation of S1
def calc_S2(theta, M, N, T, k, u, Q_template = np.array(None), stop_region = 1e5):
    S2 = np.zeros(1, dtype = 'float128')
    
    for l in range(1,T):
        u_l = u[l-1]

        M_l = M[:, l]
        
        # N_{l-1}
        N_lm1 =N[:, (l-1)]
        
        # Q from thetaoverflow encountered in matmul
        Q = theta_to_Q(theta, k, Q_template)
        
        # calculate P_l
        A,D,d,A_inv = eigendecomp(Q)
        exp_dt = np.exp(d*u_l)
        exp_D = np.diag(exp_dt)
        
        # @ is short-hand for np.matmul
        P_l = A @ exp_D @ A_inv
        P1_l = np.delete(P_l, -1, 1)
        
        S2_1 = (M_l - P1_l.transpose() @ N_lm1)
        
        # covariance of M_l and N_lm1
        covar_1l = np.diag(P1_l.transpose() @ N_lm1) - P1_l.transpose() @ np.diag(N_lm1) @ P1_l
        covar_inv = np.linalg.inv(covar_1l)
        
        # avoid inf values, this can be set based on range of problem
        if np.linalg.norm(S2_1,np.inf) > stop_region:
            S2 = stop_region
            break
        elif np.isnan(S2_1).any() == True:
            S2 = stop_region
            break
        
        S2_i = S2_1 @ covar_inv @ S2_1

        if np.iscomplexobj(S2_i):
            S2 = stop_region
            break
        
        S2 += S2_i

    return S2

# Calculate minimisation of S1 with additional parameter handing ingress
def calc_S1_b0(param, M, N, T, k, u, Q_template = np.array(None), stop_region = 1e5):
    theta = param[:-1]
    lam = param[-1]

    # set variables
    S1 = np.zeros(1, dtype = 'float128')
    mu_r = np.zeros(k)
    
    N_hat = np.zeros((k,T))
    N_hat[:,0] = N[:, 0].copy()
    
    
    for l in range(1,T):
        u_l = u[l-1]
        
        N_l = N[:, l]
        
        # Q from theta
        Q = theta_to_Q(theta, k, Q_template)
        
        # calculate P_l
        A,D,d,A_inv = eigendecomp(Q)
        exp_dt = np.exp(d*u_l)
        exp_D = np.diag(exp_dt)
        
        # @ is short-hand for np.matmul
        P_l = A @ exp_D @ A_inv

        # estimate proportion for timepoint l based on previous estimation
        N_hat_l = P_l.transpose() @ N_hat[:,(l-1)]

        # weighted ingress so it decreases as time progresses
        mu_r[0] = lam * (1-np.exp(-l))
        
        S1_1 = (N_l - N_hat_l - mu_r )
        
        # avoid inf values, this can be set based on range of problem
        if np.linalg.norm(S1_1,np.inf) > stop_region:
            S1 = stop_region
            break
        elif np.isnan(S1_1).any() == True:
            S1 = stop_region
            break
        
        S1_i = S1_1.transpose() @ S1_1
        
        if np.iscomplexobj(S1_i):
            S1 = stop_region
            break
        
        S1 += S1_i
        
        # save to N_hat for next run
        N_hat[:,l] = N_hat_l
        
    return S1

# cost function with model with baseline for all states
def calc_S1_baseline(param, M, N, T, k, u, Q_template = np.array(None), stop_region = 1e5):
    theta = param[:-k]
    
    # baseline proportions
    mu_r = param[-k:]

    # set variables
    S1 = np.zeros(1, dtype = 'float128')
    
    N_hat = np.zeros((k,T))
    N_hat[:,0] = N[:, 0].copy()
    
    
    for l in range(1,T):
        u_l = u[l-1]
        
        N_l = N[:, l]
        
        # Q from theta
        Q = theta_to_Q(theta, k, Q_template)
        
        # calculate P_l
        A,D,d,A_inv = eigendecomp(Q)
        exp_dt = np.exp(d*u_l)
        exp_D = np.diag(exp_dt)
        
        # @ is short-hand for np.matmul
        P_l = A @ exp_D @ A_inv

        # estimate proportion for timepoint l based on previous estimation
        N_hat_l = P_l.transpose() @ N_hat[:,(l-1)]
        
        S1_1 = (N_l - N_hat_l - mu_r )
        
        # avoid inf values, this can be set based on range of problem
        if np.linalg.norm(S1_1,np.inf) > stop_region:
            S1 = stop_region
            break
        elif np.isnan(S1_1).any() == True:
            S1 = stop_region
            break
        
        S1_i = S1_1.transpose() @ S1_1
        
        if np.iscomplexobj(S1_i):
            S1 = stop_region
            break
        
        S1 += S1_i
        
        # save to N_hat for next run
        N_hat[:,l] = N_hat_l
        
    return S1

# OPTIMISER W\ RANDOM START SEED
def mc_optimiser(func, args, bounds, constraints, sampN, sampRange_ub=1e0, options = {'ingress': False, 'initial': 1e0}, hessian = None, xtol=1e-8, maxiter=1e-4):
    samp = 0
    
    # set S1 initial value
    S1_min = inf
    
    while samp < sampN:
        # sample theta_0
        theta_0 = make_theta0(args[3], ub = sampRange_ub, options = options)  # this position matches with requirement for calc_S1
        
        # minimise
        S1 = minimize(func, theta_0, args = args, bounds=bounds, constraints=constraints, hess=hessian, method='trust-constr', options={'verbose': 0, 'initial_tr_radius': 1e2, 'initial_constr_penalty': 1e2,'xtol': xtol,'gtol': 1e-12, 'maxiter': maxiter} )
        
        if S1.fun < S1_min:
            theta_est = S1.x
            S1_min = S1.fun
        
        samp += 1
    print("Cost function: ", S1_min)
    print("theta_est is: ", theta_est)
    
    return theta_est, S1_min

def single_mc_optimiser(sampN,func, args, bounds, constraints, sampRange_ub, thetaFun, ingressAccount=False, ingressInitial=1e0, hessian = None, xtol=1e-15, gtol = 1e-15, maxiter=1e-4):
    
    def applyFun(fn, **kwargs):
        return fn(**kwargs)
    
    # sample theta_0
    theta_0 = applyFun(thetaFun, k = args[3], ub = sampRange_ub, options = {'ingress': ingressAccount, 'initial': ingressInitial})  # this position matches with requirement for calc_S1

    # minimise
    S1 = minimize(func, theta_0, args = args, bounds=bounds, constraints=constraints, hess=hessian, method='trust-constr', options={'verbose': 0, 'initial_tr_radius': 1e2, 'initial_constr_penalty': 1e2,'xtol': xtol,'gtol': gtol, 'maxiter': maxiter} )
    
    S1_min = S1.fun
    theta_est = S1.x
    
    return theta_est, S1_min 

# PARALLELISE

# make variables and packages for optimiser global
def initGlobal(cost_function,gen_theta, func_optimise):
    import random as rand
    import numpy as np
    from scipy.optimize import minimize
    global cost
    cost = cost_function
    global genTheta
    genTheta = gen_theta
    
    global minimize
    minimize = func_optimise

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
    chunk_size = int(iter_samp/n_cores)
    
    # set max
    costFunc_min = inf
    
    # track optimisation progress
    if progress_bar==True:
        i=0
        bar = progressbar.ProgressBar(maxval=iter_samp, \
        widgets=['Optimisation progress: ',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    # create a configure process pool
    with Pool(n_cores) as pool:
        # execute tasks in order
        for est, cost in pool.imap(partial(single_mc_optimiser, func=optimiserArgs['costFunc'], args=optimiserArgs['args'], bounds=optimiserArgs['bounds'], constraints=optimiserArgs['constraints'], sampRange_ub=optimiserArgs['sampRange_ub'], thetaFun=theta_generator, xtol=options['xterm'], gtol=options['gterm'], maxiter=options['max_iter'], ingressAccount=optimiserArgs['ingressAccount']), range(iter_samp), chunksize=chunk_size):
            if progress_bar==True:
                i+=1
                bar.update(i)
            
            if cost < costFunc_min:
                theta_min = est
                costFunc_min = cost
    
    if progress_bar==True:
        bar.finish()         
    
    return theta_min, costFunc_min



# -------------------------------------------------------------------------------------------------------------------------


#### BOOTSTRAP FUNCTIONS  ####

## NON-PARAMETRIC: RESAMPLING CASES
def bootstrap_caseResamp(repertoire, n_bootstrapSamples, timeParam, n_param, theta_generator, stateParam, isotype_list, mcArgs, optimiserArgs, options={'alpha_significance': 0.05}):
    # separate out time and state arguments
    dayColumnName = timeParam['dayColumnName']
    timemarker = timeParam['timemarker']
    u = timeParam['timeinterval']
    
    state_col_name = stateParam['state_col_name']
    k = stateParam['numStates']
    
    # number of bootstrap samples
    B = n_bootstrapSamples
    
    # sample from the repertoire per day 
    timespan = np.sort(repertoire[dayColumnName].unique())

    # create array to save bootstrap estimates
    theta_boot = np.zeros((B,n_param))
    
    # create array for standard error calculation
    se_boot = np.zeros(B)

    # track bootstrap progress
    bar = progressbar.ProgressBar(maxval=B, \
    widgets=['Bootstrap progress:',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for i in range(B):
        ## RESAMPLING
        # empty dataframe for new sampled df
        sim_boot = pd.DataFrame({'State': [], 'Day': []})

        for j in timespan:
            day_rep = repertoire[repertoire[dayColumnName] == j]
    
            # size of sample
            n = day_rep.shape[0]
    
            sim_boot = sim_boot._append(day_rep.sample(n, replace=True, random_state=0, ignore_index=True))

        boot_df = formatCount(sim_boot, timespan, dayColumnName, state_col_name, timemarker, isotype_list)        
        boot = boot_df.to_numpy()
        
        # new values for boot. sample      
        N, M, T, k = eqTin(boot, proportion=False)
        optimiserArgs['args'] = (M, N, T, k, u)
                
        ## ESTIMATE
        theta_b, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options)
        
        # save out value
        theta_boot[i,:] = theta_b
        
        bar.update(i+1)
    
    bar.finish()    
    print("Compiling results...")
    
    ## STANDARD ERROR
    theta_star = theta_boot.sum(axis=0) / B
     
    se_boot = np.sqrt( np.sum( ( theta_boot - theta_star ) ** 2 , axis=0) * (B - 1)**(-1) )
    
    ## CONFIDENCE INTERVAL
    if options.get('alpha.significance') == None:
        alpha = 0.05
    else:
        alpha = options['alpha_significance']
    conInt_alpha = np.quantile(theta_boot,[alpha/2,1-alpha/2], axis=0)
    
    # also output theta_boot to test whether the mean of boot approaches the mean of the samplw
    bootstrapOutput = {'bootEstimates': theta_boot, 'standardError': se_boot, 'confidenceInterval': conInt_alpha, 'significanceLevel': alpha}
    
    return bootstrapOutput


## NON-PARAMETRIC: RESAMPLING RESIDUALS
def bootstrap_resResamp(data, time_interval, n_bootstrapSamples, n_param, theta_generator, mcArgs, optimiserArgs, options={'alpha_significance': 0.05}, Q_template = np.array(None)):
    # number of bootstrap samples
    B = n_bootstrapSamples
    
    # create array to save bootstrap estimates and bootstrap proportion
    theta_boot = np.zeros((B,n_param))
    prop_boot = np.zeros(data.shape)
    
    # create array for standard error calculation
    se_boot = np.zeros(B)   
    
    ## POINT ESTIMATION
    print("Point estimate calculation ...")
    
    # set variables for estimation
    N, M, T, k = eqTin(data, giveK=True)
    u = time_interval
    
    # set first time point as data from real dataset
    prop_boot[:,0] = N[:,0]
    
    optimiserArgs['args'] = (M, N, T, k, u)
    
    # first estimation
    theta_point, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options, progress_bar = True)
    
    # residuals
    Q_est = theta_to_Q(theta_point, k, Q_template)
    resid = np.zeros((T-1,(k-1)))
    for i in range(1,T):
        P_est = transMat(Q_est, u[i-1])
        resid[(i-1),:] = M[:,i] - np.matmul(P_est[:,:(k-1)].transpose(),N[:,(i-1)])
        
    residSampling = lambda state: np.random.choice(resid[:,state], replace=True)
    
    # track bootstrap progress
    bar = progressbar.ProgressBar(maxval=B, \
    widgets=['Bootstrap progress:',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for i in range(B):
        ## RESAMPLING
        # empty dataframe for new sampled df
        sim_boot = pd.DataFrame({'State': [], 'Day': []})

        for t in range(1,T):
            sampRes = list( map( residSampling, range(k-1)) )
            
            # add residuals to proportion for (k-1) states
            prop_boot[:(k-1),t] = N[:(k-1),t] + sampRes
            
            # make sure final state has column sum to 1
            prop_boot[(k-1),t] = 1 - prop_boot[:(k-1),t].sum()

        # new values for boot. sample      
        N, M, T, k = eqTin(prop_boot, proportion=False)
        optimiserArgs['args'] = (M, N, T, k, u)
                
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


## PARAMETRIC: GENERATE FROM MODEL
def bootstrap_parametric(data, time_interval, n_bootstrapSamples, n_param, theta_generator, mcArgs, optimiserArgs, options={'alpha_significance': 0.05}, Q_template = np.array(None)):
    # number of bootstrap samples
    B = n_bootstrapSamples
    
    # create array to save bootstrap estimates and bootstrap proportion
    theta_boot = np.zeros((B,n_param))
    N_boot = np.zeros(data.shape)
    
    # create array for standard error calculation
    se_boot = np.zeros(B)   
    
    ## POINT ESTIMATION
    print("Point estimate calculation ...")
    
    # set variables for estimation
    N, M, T, k = eqTin(data, proportion=True, giveK=True)
    u = time_interval
    
    # new values for boot. sample      
    optimiserArgs['args'] = (M, N, T, k, u)
    
    # set first time point as data from real dataset
    N_boot[:,0] = N[:,0]
    
    # first estimation
    theta_est, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, theta_generator=theta_generator, optimiserArgs=optimiserArgs, options=options, progress_bar = True)    
    
    # save out value
    theta_boot[0,:] = theta_est
    
    # track bootstrap progress
    bar = progressbar.ProgressBar(maxval=B, \
    widgets=['Bootstrap progress:',' ',progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for i in range(1,B):
        ## GENERATE
        if optimiserArgs.get('ingress') == True:
            Q_boot = theta_to_Q(theta_boot[(i-1),:-1], k, Q_template)
        else:
            Q_boot = theta_to_Q(theta_boot[(i-1),:], k, Q_template)
        
        for t in range(1,T):
            P_boot = transMat(Q_boot, u[t-1])
            N_boot[:,t] = np.matmul(P_boot.transpose(),N_boot[:,(t-1)])
        
        # make other vars needed
        M_boot = np.delete(N, -1,0)
    
        # new values for boot. sample      
        optimiserArgs['args'] = (M_boot, N_boot, T, k, u)
                
        ## ESTIMATE
        theta_b, _cost = parallel_mc_optimiser(iter_samp=mcArgs['mciter'], n_cores=mcArgs['n_cores'], n_param=n_param, optimiserArgs=optimiserArgs, options=options)
        
        # save out value
        theta_boot[i,:] = theta_b
        
        bar.update(i+1)
        
    bar.finish()
    print("Compiling results...")
        
    ## STANDARD ERROR
    theta_star = theta_boot.sum(axis=0) / B
     
    se_boot = np.sqrt( np.sum( ( theta_boot - theta_star ) ** 2 , axis=0) * (B - 1)**(-1) )

    ## CONFIDENCE INTERVAL
    if options.get('alpha.significance') == None:
        alpha = 0.05
    else:
        alpha = options['alpha_significance']
    conInt_alpha = np.quantile(theta_boot,[alpha/2,1-alpha/2], axis=0)
    
    # also output theta_boot to test whether the mean of boot approaches the mean of the samplw
    bootstrapOutput = {'theta_est': theta_est, 'bootEstimates': theta_boot, 'standardError': se_boot, 'confidenceInterval': conInt_alpha, 'significanceLevel': alpha}
    return bootstrapOutput