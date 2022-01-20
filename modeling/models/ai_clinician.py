import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from modeling.columns import *
from preprocessing.columns import *
from sklearn.cluster import MiniBatchKMeans, KMeans
from modeling.models.MDPtoolbox import mdp_policy_iteration_with_Q
from modeling.models.offpolicy import off_policy_q_learning
from modeling.models.common import build_complete_record_sequences, compute_physician_policy
from modeling.models.base_ import BaseModel

class AIClinicianModel(BaseModel):
    def __init__(self,
                 n_cluster_states=750,
                 n_actions=25,
                 cluster_fit_fraction=0.25,
                 n_cluster_init=32,
                 gamma=0.99,
                 reward_val=100,
                 transition_threshold=5,
                 soften_factor=0.01,
                 random_state=None,
                 metadata=None):
        super(AIClinicianModel, self).__init__()
        self.n_cluster_states = n_cluster_states
        self.n_actions = n_actions
        self.cluster_fit_fraction = cluster_fit_fraction
        self.n_cluster_init = n_cluster_init
        self.gamma = gamma
        self.random_state = random_state
        self.reward_val = reward_val
        self.transition_threshold = transition_threshold
        self.soften_factor = soften_factor
        self.metadata = metadata
        self.n_states = self.n_cluster_states + 2
        self.absorbing_states = [self.n_cluster_states + 1, self.n_cluster_states] # absorbing state numbers
        self.rewards = [self.reward_val, -self.reward_val]
        
        self.clusterer = None
        self.Q = None
        self.physician_policy = None
        self.transitionr = None
        self.R = None
    
    def _cluster_states(self, state_data):
        """
        Produces a clustering of the given state data, where each state is
        considered independent (even from the same patient).
        
        Returns: a clustering object that can be queried using a predict() function,
            and an array of clustering indexes ranging from 0 to n_clusters.
        """
        sample = state_data[np.random.choice(len(state_data),
                                            size=int(len(state_data) * self.cluster_fit_fraction),
                                            replace=False)]
        clusterer = MiniBatchKMeans(n_clusters=self.n_cluster_states,
                                    random_state=self.random_state,
                                    n_init=self.n_cluster_init,
                                    max_iter=30).fit(sample)
        return clusterer, clusterer.predict(state_data)
    
    def train(self, X_train, actions_train, metadata_train, X_val=None, actions_val=None, metadata_val=None):
        print("Clustering")
        self.clusterer, states_train = self._cluster_states(X_train)
        
        # Create qldata3
        qldata3 = build_complete_record_sequences(
            metadata_train,
            states_train,
            actions_train,
            self.absorbing_states,
            self.rewards
        )
        
        ####### BUILD MODEL ########
        physpol, transitionr, R = compute_physician_policy(
            qldata3,
            self.n_states,
            self.n_actions,
            self.absorbing_states,
            reward_val=self.reward_val,
            transition_threshold=self.transition_threshold
        )
        
        print("Policy iteration")
        self.Q = mdp_policy_iteration_with_Q(
            np.swapaxes(transitionr, 0, 1),
            R,
            self.gamma,
            np.ones(self.n_states)
        )[-1]
        self.R = R
        self.physician_policy = physpol
        self.transitionr = transitionr
    
    def compute_states(self, X):
        return self.clusterer.predict(X)
        
    def compute_Q(self, X=None, states=None, actions=None):
        assert X is not None or states is not None, "At least one of states or X must not be None"
        if states is None:
            states = self.compute_states(X)

        if actions is not None:
            assert len(actions) == len(states)
            return self.Q[states, actions]
        return self.Q[states]
    
    def compute_V(self, X):
        raise NotImplementedError

    def compute_probabilities(self, X=None, states=None, actions=None):
        assert X is not None or states is not None, "At least one of states or X must not be None"
        Q_vals = self.compute_Q(X=X, states=states)
        optimal_actions = np.argmax(Q_vals, axis=1).reshape(-1, 1)
        probs = np.where(np.arange(self.n_actions)[np.newaxis,:] == optimal_actions,
                         1 - self.soften_factor,
                         self.soften_factor / (self.n_actions - 1))
        if actions is not None:
            return probs[np.arange(len(actions)), actions]
        return probs
    
    def compute_physician_probabilities(self, X=None, states=None, actions=None):
        """
        Returns the probabilities for each state and action according to the
        physician policy, which is learned using the same state clustering model
        as the AI Clinician.
        """
        assert X is not None or states is not None, "At least one of states or X must not be None"
        soft_physpol = self.physician_policy.copy() # behavior policy = clinicians'

        for i in range(self.n_cluster_states):
            ii = soft_physpol[i,:] == 0
            z = self.soften_factor / ii.sum()
            coef = soft_physpol[i,~ii].sum()
            soft_physpol[i, ii] = z
            soft_physpol[i, ~ii] = soft_physpol[i,~ii] * (1 - self.soften_factor / coef)

        if states is None:
            states = self.compute_states(X)
        probs = soft_physpol[states]
        if actions is not None:
            return probs[np.arange(len(actions)), actions]
        return probs
        
    def save(self, filepath, metadata=None):
        """
        Saves the model as a pickle to the given filepath.
        """
        model_data = {
            'model_type': 'AIClinicianModel',
            'Qon': self.Q,
            'physician_policy': self.physician_policy,
            'T': self.transitionr,
            'R': self.R,
            'clusterer': self.clusterer,
            'params': {
                'n_cluster_states': self.n_cluster_states,
                'n_actions': self.n_actions,
                'cluster_fit_fraction': self.cluster_fit_fraction,
                'n_cluster_init': self.n_cluster_init,
                'gamma': self.gamma,
                'random_state': self.random_state,
                'reward_val': self.reward_val,
                'transition_threshold': self.transition_threshold,
                'soften_factor': self.soften_factor
            }
        }
        if metadata is not None:
            model_data['metadata'] = metadata 
        with open(filepath, 'wb') as file:
            pickle.dump(model_data, file)
    
    @classmethod
    def load(cls, filepath):
        """
        Loads a model from a pickle file at the given filepath.
        """
        with open(filepath, 'rb') as file:
            model_data = pickle.load(file)
        assert model_data['model_type'] == 'AIClinicianModel', 'Invalid model type for AIClinicianModel'
        model = cls(**model_data['params'])
        model.metadata = model_data.get('metadata', None)
        model.clusterer = model_data['clusterer']
        model.physician_policy = model_data['physician_policy']
        model.Q = model_data['Qon']
        model.transitionr = model_data['T']
        model.R = model_data['R']
        return model


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