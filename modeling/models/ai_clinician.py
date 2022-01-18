import numpy as np
import pandas as pd
from tqdm import tqdm
from modeling.columns import *
from preprocessing.columns import *
from sklearn.cluster import MiniBatchKMeans, KMeans
from modeling.models.MDPtoolbox import mdp_policy_iteration_with_Q
from modeling.models.offpolicy import off_policy_q_learning
from modeling.models.common import build_record_sequences_with_policies, offpolicy_eval_tdlearning, offpolicy_eval_wis

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

def compute_optimal_policy(physpol, transitionr, R, gamma=0.99):
    """
    Creates a treatment policy for the given data. Given a set of S states and
    A actions, the policy is defined by an (S, A) matrix providing the long-term
    (finite horizon?) reward for taking each action a from each state s.
    
    Returns: The Q matrix (S, A) providing the value of taking each action from
        each state.
    """    
    n_states = physpol.shape[0]

    _, _, _, Q = mdp_policy_iteration_with_Q(
        np.swapaxes(transitionr, 0, 1),
        R,
        gamma,
        np.ones(n_states)
    )
    
    return Q

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