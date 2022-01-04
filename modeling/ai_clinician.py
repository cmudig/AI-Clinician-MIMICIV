import numpy as np
import pandas as pd
from tqdm import tqdm
from modeling.columns import *
from preprocessing.columns import *
from sklearn.cluster import MiniBatchKMeans, KMeans
from modeling.MDPtoolbox import mdp_policy_iteration_with_Q
from modeling.offpolicy import off_policy_q_learning

def cluster_states(state_data, fit_fraction=0.25, n_cluster_init=32, n_clusters=750, random_state=None):
    """
    Produces a clustering of the given state data, where each state is
    considered independent (even from the same patient).
    
    Returns: a clustering object that can be queried using a predict() function,
        and an array of clustering indexes ranging from 0 to n_clusters.
    """
    sample = state_data[np.random.choice(len(state_data),
                                         size=int(len(state_data) * fit_fraction),
                                         replace=False)]
    clusterer = MiniBatchKMeans(n_clusters=n_clusters,
                                random_state=random_state,
                                n_init=n_cluster_init,
                                max_iter=30).fit(sample)
    return clusterer, clusterer.predict(state_data)

def fit_action_bins(input_amounts, vaso_doses, n_action_bins=5):
    """
    Groups the given fluid inputs and vasopressor doses into discrete bins based
    on percentile ranks. The resulting set of actions has n_action_bins *
    n_action_bins possible actions.
    
    Returns: an assignment of each patient step (same length as input_amounts 
        and vaso_doses) to an action number; a tuple of the median values of
        input and vaso for each bin; and a tuple of the bin boundaries for each
        bin that can be used in transform_actions.
    """
    bin_percentiles = np.linspace(0, 100, n_action_bins - 1, endpoint=False)

    # io = input actions from 0 to n_actions - 1, where 0 is no fluid and 1 is
    # the next smallest amount of fluid
    input_cutoffs = [0.0] + np.percentile(input_amounts[input_amounts > 0], bin_percentiles).tolist()
    io = np.digitize(input_amounts, input_cutoffs)
    median_inputs = [
        np.median(input_amounts[io == bin_num])
        for bin_num in range(n_action_bins)
    ]
    
    # vc = vasopressor actions, same rule as input
    vaso_cutoffs = [0.0] + np.percentile(vaso_doses[vaso_doses > 0], bin_percentiles).tolist()
    vc = np.digitize(vaso_doses, vaso_cutoffs)
    median_vaso = [
        np.median(vaso_doses[vc == bin_num])
        for bin_num in range(n_action_bins)
    ]
    
    med = np.array([io, vc])
    actions = (med[0] - 1) * n_action_bins + (med[1] - 1)
        
    return actions, (median_inputs, median_vaso), (input_cutoffs, vaso_cutoffs)
    
def transform_actions(input_amounts, vaso_doses, cutoffs):
    """
    Transforms a set of continuous fluid and vasopressor actions into discrete
    bins using the given set of cutoffs. The cutoffs are a tuple of bin
    boundaries for fluids and vasopressors, such as those produced by the last
    return value of fit_action_bins.
    """
    input_cutoffs, vaso_cutoffs = cutoffs
    return (
        len(input_cutoffs) * (np.digitize(input_amounts, input_cutoffs) - 1) +
        (np.digitize(vaso_doses, vaso_cutoffs) - 1)
    )

def build_complete_record_sequences(blocs, stay_ids, states, actions, outcomes, absorbing_states, reward_values):
    """
    Builds a dataframe of timestepped records, adding a bloc at the end of each
    record sequence for the absorbing state (discharge or death) that occurs for
    the patient.
    """
    qldata3 = []
    for i in range(len(blocs)):
        qldata3.append({
            C_BLOC: blocs[i],
            C_ICUSTAYID: stay_ids[i],
            C_STATE: states[i],
            C_ACTION: actions[i],
            C_OUTCOME: outcomes[i],
            C_REWARD: 0
        })
        if i < len(blocs) - 1 and blocs[i + 1] == 1: # end of trace for this patient (next bloc is 1 for the next patient)
            qldata3.append({
                C_BLOC: blocs[i] + 1,
                C_ICUSTAYID: stay_ids[i],
                C_STATE: absorbing_states[int(outcomes[i])],
                C_ACTION: -1,
                C_OUTCOME: int(outcomes[i]),
                C_REWARD: reward_values[int(outcomes[i])]
            })
    return pd.DataFrame(qldata3)

def compute_transition_counts(qldata3, n_states, n_actions, transition_threshold=None):
    """
    qldata3: Dataframe of blocs, states, actions, and outcomes
    n_states: Number of states (including both clustered states and absorbing states)
    n_actions: Number of actions
    transition_threshold: If not None, the number of occurrences of an (S', S, A)
        transition required to include in the transition matrix
        
    Returns: Transition counts matrix T(S', S, A).
    """
    transitionr = np.zeros([n_states, n_states, n_actions])  #this is T(S',S,A)
    
    for _, group in qldata3.groupby(C_ICUSTAYID):
        group_states = group[C_STATE].values
        group_actions = group[C_ACTION].values
        transitionr[group_states[1:], group_states[:-1], group_actions[:-1]] += 1

    # Zero out rare transitions
    if transition_threshold is not None:
        transition_sums = np.sum(transitionr, axis=0)
        print("Zeroing out {}/{} transitions".format(
            (transition_sums <= transition_threshold).sum(),
            transitionr.shape[1] * transitionr.shape[2]
        ))
        transitionr = np.where(transition_sums <= transition_threshold,
                               0, transitionr)

    return transitionr
    
def compute_optimal_policy(qldata3, n_states, n_actions, absorbing_states, reward_val=100, transition_threshold=5, gamma=0.99):
    """
    Creates a treatment policy for the given data. Given a set of S states and
    A actions, the policy is defined by an (S, A) matrix providing the long-term
    (finite horizon?) reward for taking each action a from each state s.
    
    Returns: The Q matrix (S, A) providing the value of taking each action from
        each state; the physician's policy as an (S, A) probability matrix;
        a transition matrix T(S', S, A); the reward matrix R(S, A).
    """
    
    # Average transition counts over S, A
    transitionr = compute_transition_counts(qldata3,
                                             n_states,
                                             n_actions,
                                             transition_threshold=transition_threshold)
    action_counts = np.sum(transitionr, axis=0)
    transitionr = np.nan_to_num(np.where(transitionr > 0, transitionr / action_counts, 0))
    physpol = np.nan_to_num(action_counts / action_counts.sum(axis=1, keepdims=True))

    print("Create reward matrix R(S, A)")
    # CF sutton& barto bottom 1998 page 106 - compute R[S,A] from R[S'SA] and T[S'SA]
    transition_rewards = np.zeros((n_states, n_states, n_actions))
    transition_rewards[absorbing_states[0], :, :] = reward_val
    transition_rewards[absorbing_states[1], :, :] = -reward_val
    
    R = (transitionr * transition_rewards).sum(axis=0)

    print("Policy iteration")

    _, _, _, Q = mdp_policy_iteration_with_Q(
        np.swapaxes(transitionr, 0, 1),
        R,
        gamma,
        np.ones(n_states)
    )
    # optimal_actions = Qon.argmax(axis=1)  #deterministic 

    # print(optimal_actions.shape)
    # print(optimal_actions)
    
    return Q, physpol, transitionr, R

def build_record_sequences_with_policies(qldata3, predicted_actions, physpol, n_cluster_states, soften_factor=0.01):
    """
    Creates a copy of the given qldata3 containing softened physician and model
    actions, as well as the optimal predicted action for the state.
    """
    qldata3 = qldata3.copy()
    
    n_states, n_actions = physpol.shape
    soft_physpol = physpol.copy() # behavior policy = clinicians'

    for i in range(n_cluster_states):
        ii = soft_physpol[i,:] == 0
        z = soften_factor / ii.sum()
        nz = soften_factor / (~ii).sum()
        soft_physpol[i, ii] = z
        soft_physpol[i, ~ii] = soft_physpol[i,~ii] - nz
    
    # "optimal" policy = target policy = evaluation policy
    soft_modelpol = np.ones([n_states, n_actions]) * soften_factor / (n_actions - 1)
    soft_modelpol[np.arange(n_states), predicted_actions] = 1 - soften_factor

    qldata3[C_SOFTENED_PHYSICIAN_PROBABILITY] = soft_physpol[
        qldata3[C_STATE],
        qldata3[C_ACTION]
    ]
    qldata3[C_SOFTENED_MODEL_PROBABILITY] = soft_modelpol[
        qldata3[C_STATE],
        qldata3[C_ACTION]
    ]
    qldata3[C_OPTIMAL_ACTION] = predicted_actions[qldata3[C_STATE]]
    return qldata3


### Evaluation

def offpolicy_eval_tdlearning(qldata3, physpol, gamma, num_iter, n_cluster_states, alpha=0.1, num_traces=300000):
    """
    Performs off-policy evaluation on the physician policy.
    """
    # V value averaged over state population
    # hence the difference with mean(V) stored in recqvi(:,3)

    n_states, n_actions = physpol.shape
    bootql = np.zeros(num_iter)
    a = qldata3.loc[qldata3[C_BLOC] == 1, C_STATE].values
    initial_state_dist = np.array([(a == i).sum() for i in range(n_cluster_states)])

    # Changed the implementation to sample from all patients
    traces = {
        stay_id: trace[[C_REWARD, C_STATE, C_ACTION]].values
        for stay_id, trace in qldata3.groupby(C_ICUSTAYID)
    }
    
    for i in tqdm(range(num_iter), desc='TD evaluation'):
        Qoff, _ = off_policy_q_learning(
            traces,
            n_states, n_actions,
            gamma, alpha, num_traces)

        # Value is sum of Q values over each action
        V = (physpol * Qoff).sum(axis=1)
        bootql[i] = np.nansum(V[:n_cluster_states] * initial_state_dist) / initial_state_dist.sum()

    return bootql


def offpolicy_eval_wis( qldata3,gamma ,num_iter):
    """
    Performs off-policy evaluation on the predicted policy using weighted
    importance sampling (WIS).
    """
    # WIS estimator of AI policy
    # Thanks to Omer Gottesman (Harvard) for his assistance

    bootwis = np.zeros(num_iter)
    p = qldata3[C_ICUSTAYID].unique()
    num_patients = min(25000, int(len(p) * 0.75))  # 25000 patients or 75% of samples, whichever is less

    traces = {
        stay_id: trace[[C_SOFTENED_PHYSICIAN_PROBABILITY,
                        C_SOFTENED_MODEL_PROBABILITY,
                        C_REWARD]].values
        for stay_id, trace in qldata3.groupby(C_ICUSTAYID)
    }
    
    for jj in tqdm(range(num_iter), desc='WIS estimation'):
        # Sample the population
        sample = [traces[stay_id] for stay_id in np.random.choice(p, size=num_patients, replace=False)]
        
        # Compute the cumulative importance ratio rho for each trajectory
        def compute_rho(trajectory):
            physician_probs = trajectory[:-1, 0]
            model_probs = trajectory[:-1, 1]
            return np.prod(model_probs / physician_probs)
        
        rho_array = np.array([compute_rho(trace) for trace in sample])
        num_nonzero_rhos = (rho_array > 0).sum()
        invalid_rhos = np.isinf(rho_array) | np.isnan(rho_array)  #some rhos are INF

        normalization = np.nansum(rho_array)

        # Compute the individual trial estimators V_WIS, which are the discounted
        # rewards over each trajectory weighted by rho
        def compute_trial_estimator(trajectory):
            rewards = trajectory[1:, 2]
            discounts = gamma ** np.arange(-1, len(rewards) - 1)
            return np.sum(discounts * rewards)
            
        individual_trial_estimators = np.array([compute_trial_estimator(trace) for trace in sample]) * rho_array

        # Normalize by rhos
        bootwis[jj] = np.nansum(individual_trial_estimators) / normalization

    # Return the individual information for the last sampling run
    individual_trial_estimators = individual_trial_estimators[~invalid_rhos] / rho_array[~invalid_rhos]

    return bootwis, num_nonzero_rhos, individual_trial_estimators

def evaluate_policy(qldata3, predicted_actions, physpol, n_cluster_states, soften_factor=0.01, gamma=0.99, num_iter_ql=6, num_iter_wis=750):
    """
    Evaluates a policy using two off-policy evaluation methods on the given set
    of patient trajectories.
    
    Parameters:
        qldata3: A dataframe containing complete record sequences of each
            patient
        predicted_actions: A vector of length S (number of states) containing
            the actions predicted by the policy to be evaluated.
        physpol: The actual physician policy, expressed as a matrix (S, A) of
            action probabilities given each state.
            
    Returns: Bootstrapped CIs for TD-learning and WIS.
    """

    qldata3 = build_record_sequences_with_policies(
        qldata3,
        predicted_actions,
        physpol,
        n_cluster_states,
        soften_factor=soften_factor
    )

    bootql = offpolicy_eval_tdlearning(qldata3, physpol, gamma, num_iter_ql, n_cluster_states)
    bootwis, _, _ = offpolicy_eval_wis(qldata3, gamma, num_iter_wis)

    return bootql, bootwis