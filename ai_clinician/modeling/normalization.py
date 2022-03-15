import numpy as np
import pandas as pd
import tqdm
import pickle
from ai_clinician.preprocessing.utils import load_csv
from ai_clinician.preprocessing.columns import *
from ai_clinician.modeling.columns import *
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tqdm.tqdm.pandas()

class DataNormalization:
    """
    Handles all normalization of MIMIC data.
    """
    
    def __init__(self, training_data, scaler=None):
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            scores_to_norm = np.hstack([training_data[NORM_COLUMNS].astype(np.float64).values,
                                        self._clip_and_log_transform(training_data[LOG_NORM_COLUMNS])])
            self.scaler.fit(scores_to_norm)

    def _preprocess_normalized_data(self, MIMICzs):
        """Performs ad-hoc normalization on the normalized variables."""
        
        MIMICzs[pd.isna(MIMICzs)] = 0
        MIMICzs[C_MAX_DOSE_VASO] = np.log(MIMICzs[C_MAX_DOSE_VASO] + 6)   # MAX DOSE NORAD 
        MIMICzs[C_INPUT_STEP] = 2 * MIMICzs[C_INPUT_STEP]   # increase weight of this variable
        return MIMICzs

    def _clip_and_log_transform(self, data, log_gamma=0.1):
        """Performs a log transform log(gamma + x), and clips x values less than zero to zero."""
        return np.log(log_gamma + np.clip(data, 0, None))

    def transform(self, data):
        no_norm_scores = data[AS_IS_COLUMNS].astype(np.float64).values - 0.5
        scores_to_norm = np.hstack([data[NORM_COLUMNS].astype(np.float64).values,
                                    self._clip_and_log_transform(data[LOG_NORM_COLUMNS])])
        normed = self.scaler.transform(scores_to_norm)
        
        MIMICzs = pd.DataFrame(np.hstack([no_norm_scores, normed]), columns=ALL_FEATURE_COLUMNS)
        return self._preprocess_normalized_data(MIMICzs)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump({
                'type': 'DataNormalization',
                'scaler': self.scaler
            }, file)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        assert obj['type'] == 'DataNormalization'
        return DataNormalization(None, obj['scaler'])