import numpy as np

class BaseModel:
    """
    A base interface for a model that predicts optimal treatment strategies
    given patient trajectory data.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def train(self, X_train, actions_train, metadata_train, X_val=None, actions_val=None, metadata_val=None):
        """
        Fits the model to compute optimal treatments.
        
        Parameters:
        - X_train: A matrix of states, where each row corresponds to a patient's
            state at a timestep.
        - actions_train: A vector of action values of the same length as X_train.
        - metadata_train: A dataframe of metadata, including bloc numbers 
            (indexes of timesteps), ICU stay IDs, and binary outcomes.
        - X_val (optional): Same as X_train, but for validation data.
        - actions_val (optional): Same as actions_train, but for validation 
            data.
        - metadata_val (optional): Same as metadata_train, but for validation 
            data.
        """
        raise NotImplementedError
    
    def compute_states(self, X):
        """
        Determines a representation of the states described in X.
        """
        raise NotImplementedError
        
    def compute_Q(self, X=None, states=None, actions=None):
        """
        Computes Q values for every action taken in each state described in X,
        or the state representations provided in states. If actions are provided,
        compute the Q values only for those actions.
        """
        raise NotImplementedError
    
    def compute_V(self, X):
        """
        Computes V, the value of each state described in X.
        """
        raise NotImplementedError
    
    def compute_probabilities(self, X=None, states=None, actions=None):
        """
        Computes the probability of taking each of the given actions in each of
        the states, which are either defined in X or states.
        """
        raise NotImplementedError
        
    def save(self, filepath):
        """
        Saves the model as a pickle to the given filepath.
        """
        raise NotImplementedError
    
    @classmethod
    def load(cls, filepath):
        """
        Loads a model from a pickle file at the given filepath.
        """