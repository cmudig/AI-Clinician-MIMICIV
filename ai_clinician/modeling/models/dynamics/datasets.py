import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch.nn.init as weight_init
from torch.optim import Adam

class DynamicsDataset(torch.utils.data.Dataset):
    """
    A dataset that creates sequence-level items containing states, demographics,
    actions, missingness, rewards, and returns.
    """
    def __init__(self, 
                 stay_ids, 
                 observations, 
                 demographics,
                 actions, 
                 rewards, 
                 replacement_values=None,
                 obs_transform=None,
                 demog_transform=None, 
                 gamma=0.95,
                 mask_prob=0.0):
        """
        stay_ids, observations, demographics, actions, and outcomes should all be the same length.
        gamma = discount factor for value function.
        mask_prob = probability of zeroing any value when returned.
        next_step_delta = if True, return the feature-wise difference between the next step and the previous one.
            if False, return the actual next step value
        obs_transform: if not None, a function that should take a matrix of
            observations and return a transformed version as well as a missingness
            mask.
        demog_transform: Same as obs_transform but for demographics.
        """
        assert len(stay_ids) == len(observations) == len(demographics) == len(actions) == len(rewards)
        self.observations = observations
        self.demographics = demographics
        self.actions = actions
        self.stay_ids = stay_ids
        self.rewards = rewards
        self.gamma = gamma
        self.mask_prob = mask_prob
        self.replacement_values = replacement_values
        self.obs_transform = obs_transform
        self.demog_transform = demog_transform
        
        self.stay_id_pos = []
        last_stay_id = None
        for i, stay_id in enumerate(self.stay_ids):
            if last_stay_id != stay_id:
                if self.stay_id_pos:
                    self.stay_id_pos[-1] = (self.stay_id_pos[-1][0], i)
                    assert i - 1 > self.stay_id_pos[-1][0], last_stay_id
                self.stay_id_pos.append((i, 0))
                last_stay_id = stay_id
        self.stay_id_pos[-1] = (self.stay_id_pos[-1][0], len(self.stay_ids))
        
    def __len__(self):
        return len(self.stay_id_pos)

    def __getitem__(self, index):
        """
        Returns:
            observation sequence (N, L, S)
            demographics (N, L, D)
            actions (N, L, A)
            missingness (N, L, S) - 1 if value was missing and imputed with median value, otherwise 0
            rewards (N, L) - rewards at each time step
            values (N, L) - discounted returns at each time step
        """
        trajectory_indexes = np.arange(*self.stay_id_pos[index])
        assert len(trajectory_indexes) > 0
        observations = self.observations[trajectory_indexes]
        demographics = self.demographics[trajectory_indexes]
        actions = self.actions[trajectory_indexes]
        rewards = self.rewards[trajectory_indexes]
        
        if self.obs_transform is not None:
            observations, obs_missing_mask = self.obs_transform(observations)
        else:
            # Replace NaNs with median
            obs_missing_mask = pd.isna(observations)
            if self.replacement_values is not None:
                observations = np.where(obs_missing_mask, self.replacement_values, observations)
        
        if self.demog_transform is not None:
            demographics, _ = self.demog_transform(demographics)
            
        # Mask if needed
        input_obs = observations
        if self.mask_prob > 0.0:
            # Randomly replace observation values with the median
            should_mask = np.logical_and(np.random.uniform(0.0, 1.0, size=input_obs.shape) < self.mask_prob,
                                         1 - obs_missing_mask)
            input_obs = np.where(should_mask, self.replacement_values, input_obs)
               
        # Calculate discounted rewards 
        outcome_baseline = 0.0
        discounted_rewards = np.zeros((len(trajectory_indexes), 1))
        for i in reversed(range(len(trajectory_indexes))):
            outcome_baseline = self.gamma * outcome_baseline + rewards[i]
            discounted_rewards[i] = outcome_baseline
        
        return (
            torch.from_numpy(input_obs).float(), 
            torch.from_numpy(demographics).float(),
            torch.from_numpy(actions).float(),
            torch.from_numpy(obs_missing_mask), 
            torch.from_numpy(rewards.reshape(-1, 1)).float(),
            torch.from_numpy(discounted_rewards).float()
        )

    def bootstrap(self, n_trajectories=None):
        """
        Creates a new DynamicsDataset that is a bootstrapped sample of this
        dataset, with optional new number of trajectories n_trajectories.
        """
        resampled_indexes = np.random.choice(len(self.stay_id_pos), size=n_trajectories or len(self.stay_id_pos), replace=True)
        sampled_stay_ids = []
        sampled_obs = []
        sampled_dem = []
        sampled_actions = []
        sampled_outcomes = []
        for stay_id_idx in resampled_indexes:
            trajectory_indexes = np.arange(*self.stay_id_pos[stay_id_idx])
            sampled_stay_ids.append(self.stay_ids[trajectory_indexes])
            sampled_obs.append(self.observations[trajectory_indexes])
            sampled_dem.append(self.demographics[trajectory_indexes])
            sampled_actions.append(self.actions[trajectory_indexes])
            sampled_outcomes.append(self.rewards[trajectory_indexes])
            
        return DynamicsDataset(
            np.concatenate(sampled_stay_ids),
            np.vstack(sampled_obs),
            np.vstack(sampled_dem),
            np.vstack(sampled_actions),
            np.concatenate(sampled_outcomes),
            replacement_values=self.replacement_values,
            obs_transform=self.obs_transform,
            demog_transform=self.demog_transform,
            gamma=self.gamma,
            mask_prob=self.mask_prob
        )