import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import matplotlib
import gc
from importlib import reload

import torch
import datetime as dt
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch.nn.init as weight_init
from torch.optim import Adam
import scipy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from ai_clinician.modeling.models.komorowski_model import *
from ai_clinician.modeling.models.dqn import *
from ai_clinician.modeling.models.common import *
from ai_clinician.modeling.normalization import *
from ai_clinician.modeling.columns import C_OUTCOME
from ai_clinician.preprocessing.utils import load_csv, load_intermediate_or_raw_csv
from ai_clinician.preprocessing.columns import *
from ai_clinician.modeling.models.dynamics.datasets import DynamicsDataset

AS_IS_COLUMNS = [C_MECHVENT, C_POSITIVE_CULTURE, C_ON_ANTIBIOTIC]
NORM_COLUMNS = [C_WEIGHT, C_GCS, 
                C_HR, C_SYSBP, C_MEANBP, C_DIABP, C_RR, 
                C_TEMP_C, C_FIO2_1, C_PAPSYS, C_PAPMEAN, C_PAPDIA, 
                C_CI, C_IONISED_CA, C_CO2_MEQL, C_TOTAL_PROTEIN,  # there's no SVR data so excluding C_SVR
                C_ALBUMIN, C_TROPONIN, C_CRP, C_ACT,
                C_POTASSIUM, C_SODIUM, C_CHLORIDE, C_GLUCOSE, C_MAGNESIUM, C_CALCIUM,
                C_HB, C_HT, C_RBC_COUNT, C_WBC_COUNT, C_PLATELETS_COUNT, C_PTT, 
                C_PT, C_ARTERIAL_PH, C_PAO2, C_PACO2, C_ARTERIAL_BE, C_HCO3, 
                C_ETCO2, C_SVO2, C_ARTERIAL_LACTATE, C_SOFA, C_SIRS, C_CUMULATED_BALANCE]
LOG_NORM_COLUMNS = [C_SPO2, C_BUN, C_CREATININE, C_SGOT, C_SGPT, 
                    C_TOTAL_BILI, C_DIRECT_BILI, C_INR, C_INPUT_TOTAL, "norm_input_step", C_MAX_DOSE_VASO, 
                    C_OUTPUT_TOTAL, "norm_output_step", C_SHOCK_INDEX, C_PAO2_FIO2]

ALL_FEATURE_COLUMNS = AS_IS_COLUMNS + NORM_COLUMNS + LOG_NORM_COLUMNS

DEMOG_NORM_COLUMNS = [C_AGE, C_ELIXHAUSER, C_HEIGHT]
COMORBIDITIES = [x for x in RAW_DATA_COLUMNS["comorbidities"] if x not in (C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID)]
DEMOGRAPHICS_AS_IS_COLUMNS = [C_GENDER, C_RE_ADMISSION] + COMORBIDITIES

class DataNormalization:
    """
    Handles all normalization of MIMIC and eICU state and demographics data.
    """
    
    def __init__(self, training_data, scaler=None, as_is_columns=AS_IS_COLUMNS, norm_columns=NORM_COLUMNS, log_norm_columns=LOG_NORM_COLUMNS):
        self.as_is_columns = as_is_columns
        self.norm_columns = norm_columns
        self.log_norm_columns = log_norm_columns
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
            scores_to_norm = np.hstack([training_data[self.norm_columns].values.astype(np.float64),
                                        self._clip_and_log_transform(training_data[self.log_norm_columns].fillna(np.nan).values.astype(np.float64))])
            self.scaler.fit(scores_to_norm)

    def _preprocess_normalized_data(self, MIMICzs):
        """Performs ad-hoc normalization on the normalized variables."""
        
        # MIMICzs[pd.isna(MIMICzs)] = 0
        # MIMICzs[C_MAX_DOSE_VASO] = np.log(MIMICzs[C_MAX_DOSE_VASO] + 6)   # MAX DOSE NORAD 
        # MIMICzs[C_INPUT_STEP] = 2 * MIMICzs[C_INPUT_STEP]   # increase weight of this variable
        return MIMICzs

    def _clip_and_log_transform(self, data, log_gamma=0.1):
        """Performs a log transform log(gamma + x), and clips x values less than zero to zero."""
        return np.log(log_gamma + np.clip(data, 0, None))
    
    def _inverse_log_transform(self, data, log_gamma=0.1):
        """Performs the inverse of the _clip_and_log_transform function (without the clipping)."""
        return np.exp(data) - log_gamma

    def transform(self, data):
        no_norm_scores = data[self.as_is_columns].astype(np.float64).values - 0.5
        scores_to_norm = np.hstack([data[self.norm_columns].values.astype(np.float64),
                                    self._clip_and_log_transform(data[self.log_norm_columns].fillna(np.nan).values.astype(np.float64))])
        normed = self.scaler.transform(scores_to_norm)
        
        MIMICzs = pd.DataFrame(np.hstack([no_norm_scores, normed]), columns=self.as_is_columns + self.norm_columns + self.log_norm_columns)
        return self._preprocess_normalized_data(MIMICzs)
    
    def inverse_transform(self, data):
        no_norm_scores = data[:,:len(self.as_is_columns)] + 0.5
        unnormed = self.scaler.inverse_transform(data[:,len(self.as_is_columns):])
        unnormed[:,len(self.norm_columns):] = self._inverse_log_transform(unnormed[:,len(self.norm_columns):])
        return pd.DataFrame(np.hstack([no_norm_scores, unnormed]), columns=self.as_is_columns + self.norm_columns + self.log_norm_columns)
        
REWARD_VAL = 10

class DynamicsDataNormalizer:
    def __init__(self, obs, demog, actions, _loaded_data=None):
        super().__init__()
        if _loaded_data is not None:
            self.obs_norm = DataNormalization(None, scaler=_loaded_data['obs_scaler'])
            self.demog_norm = DataNormalization(None,
                                                scaler=_loaded_data['demog_scaler'], 
                                                as_is_columns=DEMOGRAPHICS_AS_IS_COLUMNS,
                                                norm_columns=DEMOG_NORM_COLUMNS,
                                                log_norm_columns=[])
            self.mean_action = _loaded_data['mean_action']
            self.std_action = _loaded_data['std_action']
        else:
            self.obs_norm = DataNormalization(obs)
            self.demog_norm = DataNormalization(demog, 
                                            as_is_columns=DEMOGRAPHICS_AS_IS_COLUMNS,
                                            norm_columns=DEMOG_NORM_COLUMNS,
                                            log_norm_columns=[])
            self.transform_action(actions, fit=True)
            
    def transform_state(self, obs, demog):
        obs_train = self.obs_norm.transform(obs).values
        demog_train = self.demog_norm.transform(demog).values
        return obs_train, demog_train
    
    def inverse_transform_obs(self, obs):
        return self.obs_norm.inverse_transform(obs)
    
    def transform_action(self, actions, fit=False):
        actions_train = np.log(1 + actions)

        if fit:
            self.mean_action = actions_train[np.isnan(actions_train).sum(axis=1) == 0].mean(axis=0)
            self.std_action = actions_train[np.isnan(actions_train).sum(axis=1) == 0].std(axis=0)
        
        norm_actions = (actions_train - self.mean_action) / self.std_action
        # Zero the nans since these are just at the end of each trajectory
        norm_actions = np.where(np.isnan(norm_actions), 0, norm_actions) 
        return norm_actions
    
    def inverse_transform_action(self, norm_actions):
        reverse_z_transform = norm_actions * self.std_action + self.mean_action
        reverse_log = np.exp(reverse_z_transform) - 1
        return reverse_log
        
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump({
                'type': 'DynamicsDataNormalizer',
                'obs_scaler': self.obs_norm.scaler,
                'demog_scaler': self.demog_norm.scaler,
                'mean_action': self.mean_action,
                'std_action': self.std_action
            }, file)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        assert obj['type'] == 'DynamicsDataNormalizer'
        return DynamicsDataNormalizer(None, None, None, _loaded_data=obj)
    
def calculate_rewards(df):
    # Calculate the rewards using SOFA (same as Nanayakkara et al)
    def calc_sofa_rewards(g):
        deltas = g.values[1:] - g.values[:-1]
        return np.concatenate([-0.5 * deltas - 0.1 * np.logical_and(deltas == 0, g.values[:-1] > 0), np.zeros(1)])
    
    # Add rewards based on whether the patient was discharged alive or dead
    def calc_outcome_rewards(g):
        is_death = g.max() > 0
        return np.concatenate([np.zeros(len(g) - 1), np.array([REWARD_VAL * (-1 if is_death else 1)])])
    
    rewards = (df[C_SOFA].groupby(df[C_ICUSTAYID]).transform(calc_sofa_rewards) + 
               df[C_DIED_IN_HOSP].groupby(df[C_ICUSTAYID]).transform(calc_outcome_rewards)).values
    return rewards

def next_state_value(g):
    """
    Groupby aggregation function that returns the value from the next row for
    each row, and pd.NA for the last item.
    """
    return np.concatenate([g.iloc[1:].values, np.array([pd.NA])])

def pad_collate(batch):
    """
    Helper function for DataLoader that takes an iterable of tuples containing
    variable-length tensors, and combines them together into a tuple of stacked
    tensors padded to the maximum length. Also adds a tensor to the end of the
    tuple containing the lengths of each sequence.
    """
    arrays_to_pad = list(zip(*batch))
    x_lens = [len(x) for x in arrays_to_pad[0]]

    padded_arrays = [pad_sequence(xx, batch_first=True, padding_value=0) for xx in arrays_to_pad]
    return (*padded_arrays, torch.LongTensor(x_lens))

def prepare_dataset(dataset, comorb, train_test_split=None, test_size=0.15, val_size=0.15):
    """
    Args:
        dataset: a dataframe containing all the state and most of the demographics
            columns (except for comorbidities) for each patient.
        comorb: a dataframe containing elixhauser comorbidities for each patient.
        train_test_split: If None, a train/val/test split will be generated. If
            not None, should be a tuple of (train_ids, val_ids, test_ids).
        test_size: Fraction of the entire dataset that should be used for the  
            test split (if train_test_split is None).
        val_size: Fraction of the entire dataset that should be used for the
            val split (if train_test_split is None).
    """
    # Make miscellaneous adjustments, remove patients with no recorded weight and who have only one timestep
    print("Cleaning dataset")
    dataset = dataset[~pd.isna(dataset[C_WEIGHT]) & 
                    (dataset[C_WEIGHT].groupby(dataset[C_ICUSTAYID]).transform("min") > 50) & 
                    (dataset[C_AGE].groupby(dataset[C_ICUSTAYID]).transform("min") > 18)]
    dataset.loc[np.isinf(dataset[C_PAO2_FIO2]), C_PAO2_FIO2] = pd.NA
    dataset = dataset[dataset[C_BLOC].groupby(dataset[C_ICUSTAYID]).transform("count") > 1]
    dataset = dataset.copy()
    
    dataset[C_MAX_DOSE_VASO] = dataset[C_MAX_DOSE_VASO].clip(upper=10)
    
    # Normalize input and output by weight
    dataset["norm_input_step"] = dataset[C_INPUT_STEP] / np.where(dataset[C_WEIGHT] == 0, 1, dataset[C_WEIGHT])
    dataset["norm_output_step"] = dataset[C_OUTPUT_STEP] / np.where(dataset[C_WEIGHT] == 0, 1, dataset[C_WEIGHT])

    # Generate train/val/test split
    if train_test_split is not None:
        train_ids, val_ids, test_ids = train_test_split
        print("Using provided train/val/test split")
    else:
        print("Generating new train/val/test split")
        icuuniqueids = dataset[C_ICUSTAYID].unique()
        train_ids, test_ids = train_test_split(icuuniqueids, test_size=test_size)
        train_ids, val_ids = train_test_split(train_ids, test_size=val_size * len(icuuniqueids) / len(train_ids))
    print("Train test split sizes:", len(train_ids), len(val_ids), len(test_ids))
    
    train_dataset = dataset[dataset[C_ICUSTAYID].isin(train_ids)]
    val_dataset = dataset[dataset[C_ICUSTAYID].isin(val_ids)]
    test_dataset = dataset[dataset[C_ICUSTAYID].isin(test_ids)]

    # Create demog table
    print("Creating demog table")
    cleaned_demog = pd.merge(dataset[[C_ICUSTAYID, C_GENDER, C_AGE, C_ELIXHAUSER, C_HEIGHT, C_RE_ADMISSION]],
                             comorb,
                            how='left', on=C_ICUSTAYID)
    cleaned_demog.loc[pd.isna(cleaned_demog[C_HEIGHT]), C_HEIGHT] = np.nanmean(cleaned_demog[C_HEIGHT])
    cleaned_demog[pd.isna(cleaned_demog[COMORBIDITIES])] = 0
    train_demog = cleaned_demog[cleaned_demog[C_ICUSTAYID].isin(train_ids)]
    val_demog = cleaned_demog[cleaned_demog[C_ICUSTAYID].isin(val_ids)]
    test_demog = cleaned_demog[cleaned_demog[C_ICUSTAYID].isin(test_ids)]

    print("Calculating rewards")
    rewards_train = calculate_rewards(train_dataset)
    rewards_val = calculate_rewards(val_dataset)
    rewards_test = calculate_rewards(test_dataset)

    # Get shifted fluid actions to see what action was taken AFTER each timestep
    print("Calculating actions")
    raw_actions = pd.DataFrame({
        'fluid': (dataset["norm_input_step"]
                  .groupby(dataset[C_ICUSTAYID])
                  .transform(next_state_value)), 
        'vaso': (dataset[C_MAX_DOSE_VASO]
                 .groupby(dataset[C_ICUSTAYID])
                 .transform(next_state_value))
    }).replace(pd.NA, np.nan).astype(float)
    train_actions = raw_actions[dataset[C_ICUSTAYID].isin(train_ids)].values
    val_actions = raw_actions[dataset[C_ICUSTAYID].isin(val_ids)].values
    test_actions = raw_actions[dataset[C_ICUSTAYID].isin(test_ids)].values

    print("Fitting normalization parameters")
    normer = DynamicsDataNormalizer(train_dataset, train_demog, train_actions)

    obs_train, demog_train = normer.transform_state(train_dataset, train_demog)
    replacement_values = np.nanmedian(obs_train, axis=0)

    train_dataset = DynamicsDataset(dataset[dataset[C_ICUSTAYID].isin(train_ids)][C_ICUSTAYID].values,
                                    obs_train,
                                    demog_train,
                                    normer.transform_action(train_actions),
                                    rewards_train,
                                    replacement_values)
    val_dataset = DynamicsDataset(dataset[dataset[C_ICUSTAYID].isin(val_ids)][C_ICUSTAYID].values,
                                  *normer.transform_state(val_dataset, val_demog),
                                  normer.transform_action(val_actions),
                                  rewards_val,
                                  replacement_values)
    test_dataset = DynamicsDataset(dataset[dataset[C_ICUSTAYID].isin(test_ids)][C_ICUSTAYID].values,
                                   *normer.transform_state(test_dataset, test_demog),
                                   normer.transform_action(test_actions),
                                   rewards_test,
                                   replacement_values)
    
    return dataset, cleaned_demog, train_dataset, val_dataset, test_dataset, train_test_split, normer