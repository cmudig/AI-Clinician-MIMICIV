import numpy as np
from tqdm import tqdm
from numba import jit
from modeling.columns import *
from preprocessing.columns import C_ICUSTAYID

@jit(nopython=True)
def _q_learn_trace(Q, trace, gamma, alpha):
    tracelength = len(trace)
    return_t = trace[tracelength - 1, 0] # get last reward as return for penultimate state and action.
    
    for t in range(tracelength - 2, -1, -1):       #Step through time-steps in reverse order
        s = trace[t, 1] # get state index from trace at time t
        a = trace[t, 2] # get action index
        if a >= 0:
            Q[s, a] = (1 - alpha) * Q[s, a] + alpha * return_t # update Q.
        return_t = return_t * gamma + trace[t, 0] # return for time t-1 in terms of return and reward at t
    return Q

def off_policy_q_learning(traces, n_states, n_actions, gamma, alpha, num_traces):
    # OFF POLICY Q LEARNING

    sumQ = []
    Q = np.zeros((n_states, n_actions))
    maxavgQ = 1
    check_step = 50000
    
    stay_ids = list(traces.keys())
    
    for j in range(num_traces):
        trace = traces[stay_ids[np.random.choice(len(stay_ids))]]
        Q = _q_learn_trace(Q, trace, gamma, alpha)
        sumQ.append(Q.sum())
        
        if j > 0 and j % check_step == 0:  #check if can stop iterating (when no more improvement is seen)
            s = np.mean(sumQ[-check_step:])
            d = (s - maxavgQ) / maxavgQ
            if abs(d) < 0.001:
                break
            maxavgQ = s

    return Q, np.array(sumQ)


def offpolicy_eval_tdlearning( qldata3, physpol, gamma, num_iter ):
    # V value averaged over state population
    # hence the difference with mean(V) stored in recqvi(:,3)

    ncl = len(physpol) - 2
    bootql = np.zeros(num_iter)
    p = np.unique(qldata3[:, 7])
    prop = 5000 / len(p)  #5000 patients of the samples are used
    prop = min(prop, 0.75)  #max possible value is 0.75 (75# of the samples are used)

    ii = qldata3[:, 0] == 1
    a = qldata3[ii, 1]
    d = np.zeros(ncl)
    for i in range(ncl):
        d[i] = sum(a == i)    # intitial state disctribution

    for i in tqdm(range(num_iter), desc='ql evaluation'):

        ii = np.floor(np.random.rand(len(p)) + prop)     # select a random sample of trajectories
        j = np.isin(qldata3[:, 7], p[ii == 1])
        q = qldata3[j, :4]

        Qoff, _ = OffpolicyQlearning150816( q , gamma, 0.1, 300000)

        V = np.zeros([ncl,25])
        for k in range(ncl):
            for j in range(25):
                V[k, j] = physpol[k, j] * Qoff[k, j]


        Vs = sum(V.transpose()).transpose()
        bootql[i] = np.nansum(Vs[:ncl] * d) / sum(d)

        # Vs=nansum((physpol.*Qoff)')';
        # bootql(i)={sum(Vs(1:ncl).*d)/sum(d)};

    return bootql


def offpolicy_eval_wis( qldata3,gamma ,num_iter):
    # WIS estimator of AI policy
    # Thanks to Omer Gottesman (Harvard) for his assistance

    bootwis = np.zeros(num_iter)
    p = np.unique(qldata3[:, 7])
    prop = 25000 / len(p) #25000 patients of the samples are used
    prop = min(prop, 0.75)  #max possible value is 0.75 (75# of the samples are used)


    for jj in tqdm(range(num_iter), desc='wis evaluation'):
        
        ii = np.floor(np.random.rand(len(p)) + prop) # prop# of the samples are used
        j = np.isin(qldata3[:, 7], p[ii == 1])

        q = qldata3[j, :]

        fence_posts = np.where(q[:, 0] == 1)[0]
        num_of_trials = len(fence_posts)
        individual_trial_estimators = np.empty(num_of_trials)
        individual_trial_estimators[:] = np.NaN
        rho_array = np.empty(num_of_trials)
        rho_array[:] = np.NaN
        c = 0  #count of matching pairs pi_e + pi_b
            
        for i in range(num_of_trials - 1):
            rho = 1
            for t in range(fence_posts[i], fence_posts[i+1] - 1): #stops at -2
                rho = rho * q[t, 5] / q[t, 4]

            if rho > 0:
                c += 1
            rho_array[i] = rho


        ii = np.isinf(rho_array) | np.isnan(rho_array)  #some rhos are INF
        normalization = np.nansum(rho_array[~ii])

        for i in range(num_of_trials - 1):

            current_trial_estimator = 0
            rho = 1
            discount = 1 / gamma
            
            for t in range(fence_posts[i], fence_posts[i+1] - 1): #stops at -2 otherwise ratio zeroed
                rho = rho * q[t, 5] / q[t, 4]
                discount = discount * gamma
                current_trial_estimator = current_trial_estimator + discount * q[t + 1, 3]
                    
            individual_trial_estimators[i] =  current_trial_estimator*rho
            
        
        bootwis[jj] = np.nansum( individual_trial_estimators[~ii] ) / normalization

    individual_trial_estimators = individual_trial_estimators[~ii] / rho_array[~ii]  #for the last iteration only!

    return [ bootwis ,c,individual_trial_estimators]




def offpolicy_multiple_eval_010518( qldata3,physpol, gamma,do_ql,iter_ql,iter_wis):
    #performs all off-policy algos in one run. bootstraps to generate CIs.

    if do_ql == 1:  #to save time, do offpol q_learning or not
        bootql = offpolicy_eval_tdlearning( qldata3,physpol, gamma ,iter_ql)
    else:
        bootql = 55  #gives an approximate value

    
    bootwis,_,_ = offpolicy_eval_wis( qldata3,gamma ,iter_wis);

    return [ bootql,bootwis]

def offpolicy_eval_tdlearning_with_morta( qldata3, physpol, ptid, idx, actionbloctrain, Y90, gamma, num_iter ):

    bootql = np.array(num_iter)
    p = np.unique(qldata3[:, 7])
    prop = 5000 / len(p) #5000 patients of the samples are used
    prop = min(prop, 0.75)  #max possible value is 0.75 (75% of the samples are used)

    jprog = 1
    prog = np.empty([np.floor(len(ptid) * 1.01 * prop * num_iter), 4])
    prog[:] = np.NaN

    ii = qldata3[:, 0] == 1
    a = qldata3[ii, 1]
    d = np.zeros(750)

    for i in range(750):
        d[i] = sum(a == i)
    
    for i in range(num_iter):

        ii = np.floor(np.random.rand(len(p))+prop)     # select a random sample of trajectories
        j = np.isin(qldata3[:,7], p[ii == 1])
        q = qldata3[j, :4]

        Qoff, _ = OffpolicyQlearning150816( q , gamma, 0.1, 300000)


        V = np.zeros([750,25])
        for k in range(750):
            for j in range(25):
                V[k, j] = physpol[k, j] * Qoff[k, j]

        Vs = sum(V.T).T
        bootql[i] = np.nansum(Vs[:750] * d) / sum(d)
        jj = np.where(np.isin(ptid, p[ii == 1]))[0]

        for ii in range(len(jj)):  # record offline Q value in training set & outcome - for plot
            prog[jprog, 0] = Qoff[idx[jj[ii]], actionbloctrain[jj[ii]]]
            prog[jprog, 1] = Y90[jj[ii]]
            prog[jprog, 2] = ptid[jj[ii]]   #HERE EACH ITERATION GIVES A DIFFERENT PT_ID  //// if I just do rep*ptid it bugs and mixes up ids, for ex with id3 x rep 10 = 30 (which already exists)
            prog[jprog, 3] = i
            jprog += 1
        
    prog = np.delete(prog, np.s_[jprog:], 0)

    return [ bootql,prog ]

