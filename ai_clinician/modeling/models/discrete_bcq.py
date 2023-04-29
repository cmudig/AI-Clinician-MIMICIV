"""
Adapted from https://github.com/MLforHealth/rl_representations.

The classes and methods in this file are derived or pulled directly from https://github.com/sfujim/BCQ/tree/master/discrete_BCQ
which is a discrete implementation of BCQ by Scott Fujimoto, et al. and featured in the following 2019 DRL NeurIPS workshop paper:
@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti, Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

import argparse
import copy
import importlib
import json
import os
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import class_weight

# Simple full-connected supervised network for Behavior Cloning of batch data
class FC_BC(nn.Module):
    def __init__(self, state_dim, action_dim, num_nodes=64):
        super(FC_BC, self).__init__()
        self.l1 = nn.Linear(state_dim, num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.l2 = nn.Linear(num_nodes, num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.l3 = nn.Linear(num_nodes, num_nodes)
        self.bn3 = nn.BatchNorm1d(num_nodes)
        self.l4 = nn.Linear(num_nodes, action_dim)
        self.init_weights()
        
    def init_weights(self):
        self.l1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.l1.weight.data)
        self.l2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.l2.weight.data)
        self.l3.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.l3.weight.data)
        self.l4.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.l4.weight.data, gain=2.0)
    
    def forward(self, state):
        out = F.leaky_relu(self.l1(state))
        out = self.bn1(out)
        out = F.leaky_relu(self.l2(out))
        out = self.bn2(out)
        out = F.leaky_relu(self.l3(out))
        out = self.bn3(out)
        return self.l4(out)

# Simple fully-connected Q-network for the policy
class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions, num_nodes=128):
        super(FC_Q, self).__init__()
        # This model learns the Q values
        self.q1 = nn.Linear(state_dim, num_nodes)
        self.q2 = nn.Linear(num_nodes, num_nodes)
        self.q3 = nn.Linear(num_nodes, num_actions)

        # This is the model that learns which actions are acceptable (behavior cloning essentially)
        self.i1 = nn.Linear(state_dim, num_nodes)
        self.i2 = nn.Linear(num_nodes, num_nodes)
        self.i3 = nn.Linear(num_nodes, num_actions)        
        
        self.init_weights()

    def init_weights(self):
        self.q1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.q1.weight.data)
        self.q2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.q2.weight.data)
        self.q3.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.q3.weight.data)
        self.i1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.i1.weight.data)
        self.i2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.i2.weight.data)
        self.i3.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.i3.weight.data)


    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i


class BehaviorCloning(object):
    def __init__(self, input_dim, num_actions, num_nodes=256, learning_rate=1e-3, weight_decay=0.1, optimizer_type='adam', device='cpu'):
        '''Implement a fully-connected network that produces a supervised prediction of the actions
        preserved in the collected batch of data following observations of patient health.
        INPUTS:
        input_dim: int, the dimension of an input array. Default: 33
        num_actions: int, the number of actions available to choose from. Default: 25
        num_nodes: int, the number of nodes
        '''

        self.device = device
        self.state_shape = input_dim
        self.num_actions = num_actions
        self.lr = learning_rate

        # Initialize the network
        self.model = FC_BC(input_dim, num_actions, num_nodes).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        if optimizer_type == 'adam':        
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)

        self.iterations = 0

    def change_loss_function(self, actions, num_actions=25):
        class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.arange(num_actions),y=actions)
        class_weights=torch.tensor(class_weights,dtype=torch.float).to(self.device)
        print(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights) #,reduction='mean')
        self.loss_func = criterion

    def train_epoch(self, train_dataloader):
        '''Sample batches of data from training dataloader, predict actions using the network,
        Update the parameters of the network using CrossEntropyLoss.'''

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Loop through the training data
        bar = tqdm.tqdm(train_dataloader, ncols=80, miniters=100)
        for state, action in bar:
            state = state.to(self.device)
            action = action.to(self.device)

            # Predict the action with the network
            pred_actions = self.model(state)

            # Compute loss
            loss = self.loss_func(pred_actions, action.flatten())

            # Optimize the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if num_batches % 100 == 0:
                bar.set_description(f"train loss: {total_loss / num_batches:.6f}")

        self.iterations += 1

        return total_loss / num_batches

    def evaluate_model(self, val_dataloader):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            correct_count = 0
            total_count = 0
            bar = tqdm.tqdm(enumerate(val_dataloader), ncols=80, miniters=100)
            for i, (state, action) in bar:
                state = state.to(self.device)
                action = action.to(self.device)

                # Predict the action with the network
                pred_actions = self.model(state)

                # Compute loss
                loss = self.loss_func(pred_actions, action)
                total_loss += loss.item()
                correct_count += (pred_actions.argmax(1) == action).sum().item()
                total_count += state.shape[0]
                if i % 100 == 0:
                    bar.set_description(f"val loss: {total_loss / (i + 1):.6f} Acc: {correct_count / total_count:.4f}")
        return total_loss / (i + 1), correct_count / total_count

    def predict(self, states):
        ds = TensorDataset(states)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        self.model.eval()
        actions = []
        with torch.no_grad():
            bar = tqdm.tqdm(enumerate(loader), ncols=80, total=len(loader), miniters=100)
            for i, (state,) in bar:
                state = state.to(self.device)

                # Predict the action with the network
                pred_actions = self.model(state) # / output_scaling_factor
                actions.append(pred_actions.cpu().numpy())
                
        return np.concatenate(actions)  
    
class DiscreteBCQ(object):
    def __init__(
        self, 
        num_actions,
        state_dim,
        device,
        BCQ_threshold=0.3,
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=1e3,
        tau=0.005
    ):
    
        self.device = device

        # Determine network type
        self.Q = FC_Q(state_dim, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # # Decay for eps
        # self.initial_eps = initial_eps
        # self.end_eps = end_eps
        # self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        # self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold

        # Number of training iterations
        self.iterations = 0

    # NOTE: This function is only usable when doing online evaluation with a simulator.
    # NOTE: This is why the function is commented out along with all epsilon params, we're not using them
    # def select_action(self, state, eval=False):
    #     # Select action according to policy with probability (1-eps)
    #     # otherwise, select random action
    #     if np.random.uniform(0,1) > self.eval_eps:
    #         with torch.no_grad():
    #             state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
    #             q, imt, i = self.Q(state)
    #             imt = imt.exp()
    #             imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
    #             # Use large negative number to mask actions from argmax
    #             return int((imt * q + (1. - imt) * -1e8).argmax(1))
    #     else:
    #         return np.random.randint(self.num_actions)

    def train(self, replay_buffer):
        self.Q.train()
        
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample()

        self.Q_optimizer.zero_grad()

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.Q_target(next_state)
            target_Q = 10*reward + not_done * self.discount * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))

        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()
        total_loss = Q_loss.item()

        # Optimize the Q
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
  
        return total_loss / state.shape[0]


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def compute_constrained_Q(self, s):
        q, imt, i = self.Q(s)
        imt = imt.exp()
        imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
        # Use large negative number to mask actions from argmax
        return (imt * q + (1. - imt) * -1e8)
        
    def predict(self, states):
        ds = TensorDataset(states)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        self.Q.eval()
        actions = []
        with torch.no_grad():
            bar = tqdm.tqdm(enumerate(loader), ncols=80, total=len(loader), miniters=100)
            for i, (state,) in bar:
                state = state.to(self.device)

                # Predict the action with the network
                pred_actions = self.compute_constrained_Q(state) # / output_scaling_factor
                actions.append(pred_actions.cpu().numpy())
                
        return np.concatenate(actions)  
    

def train_dBCQ(replay_buffer, num_actions, state_dim, device, parameters, behav_pol, pol_eval_dataloader, is_demog):
    """
    This is directly from Killian et al.'s code and is not currently used in
    the AI-Clinician-MIMICIV code. The dBCQ policies can be trained directly
    using their repository.
    """
    # For saving files
    pol_eval_file = parameters['pol_eval_file']
    pol_file = parameters['policy_file']
    buffer_dir = parameters['buffer_dir']

    # Initialize and load policy
    policy = DiscreteBCQ(
        num_actions,
        state_dim,
        device,
        parameters["BCQ_threshold"],
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"]
    )

    # Load replay buffer
    replay_buffer.load(buffer_dir, bootstrap=True)

    evaluations = []
    episode_num = 0
    done = True
    training_iters = 0

    while training_iters < parameters["max_timesteps"]:

        for _ in range(int(parameters["eval_freq"])):
            loss = policy.train(replay_buffer)

        evaluations.append(eval_policy(policy, behav_pol, pol_eval_dataloader, parameters["discount"], is_demog, device))  # TODO Run weighted importance sampling with learned policy and behavior policy
        np.save(pol_eval_file, evaluations)
        torch.save({'policy_Q_function':policy.Q.state_dict(), 'policy_Q_target':policy.Q_target.state_dict()}, pol_file)

        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters} Loss: {loss}")


def eval_policy(policy, behav_policy, pol_dataloader, discount, is_demog, device):

    wis_est = []
    wis_returns = 0
    wis_weighting = 0

    # Loop through the dataloader (representations, observations, actions, demographics, rewards)
    for representations, obs_state, actions, demog, rewards in pol_dataloader:
        representations = representations.to(device)
        obs_state = obs_state.to(device)
        actions = actions.to(device)
        demog = demog.to(device)

        cur_obs, cur_actions = obs_state[:,:-2,:], actions[:,:-1,:].argmax(dim=-1)
        cur_demog, cur_rewards = demog[:,:-2,:], rewards[:,:-2]

        # Mask out the data corresponding to the padded observations
        mask = (cur_obs==0).all(dim=2)

        # Compute the discounted rewards for each trajectory in the minibatch
        discount_array = torch.Tensor(discount**np.arange(cur_rewards.shape[1]))[None,:]
        discounted_rewards = (discount_array * cur_rewards).sum(dim=-1).squeeze()

        # Evaluate the probabilities of the observed action according to the trained policy and the behavior policy
        with torch.no_grad():
            if is_demog:  # Gather the probability from the observed behavior policy
                p_obs = F.softmax(behav_policy(torch.cat((cur_obs.flatten(end_dim=1), cur_demog.flatten(end_dim=1)), dim=-1)), dim=-1).gather(1, cur_actions.flatten()[:,None]).reshape(cur_obs.shape[:2])
            else:
                p_obs = F.softmax(behav_policy(cur_obs.flatten(end_dim=1)), dim=-1).gather(1, cur_actions.flatten()[:,None]).reshape(cur_obs.shape[:2])
            
            q_val, _, _ = policy.Q(representations)  # Compute the Q values of the dBCQ policy
            p_new = F.softmax(q_val, dim=-1).gather(2, cur_actions[:,:,None]).squeeze()  # Gather the probabilities from the trained policy

        # Check for whether there are any zero probabilities in p_obs and replace with small probability since behav_pol may mispredict actual actions...
        if not (p_obs > 0).all(): 
            p_obs[p_obs==0] = 0.1

        # Eliminate spurious probabilities due to padded observations after trajectories have concluded 
        # We do this by forcing the probabilities for these observations to be 1 so they don't affect the product
        p_obs[mask] = 1.
        p_new[mask] = 1.

        cum_ir = torch.clamp((p_new / p_obs).prod(axis=1), 1e-30, 1e4)

        # wis_idx = (cum_ir > 0)  # TODO check that wis_idx isn't empty (all zero)
        # if wis_idx.sum() == 0:
        #     import pdb; pdb.set_trace()

        # wis = (cum_ir / cum_ir.mean()).cpu() * discounted_rewards  # TODO check that there aren't any nans
        # wis_est.extend(wis.cpu().numpy())
        wis_rewards = cum_ir.cpu() * discounted_rewards
        wis_returns +=  wis_rewards.sum().item()
        wis_weighting += cum_ir.cpu().sum().item()

    wis_eval = (wis_returns / wis_weighting) 
    print("---------------------------------------")
    print(f"Evaluation over the test set: {wis_eval:.3f}")
    print("---------------------------------------")
    return wis_eval

class DiscreteReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device, encoded_state=False, obs_state_dim=50):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.encoded_state = encoded_state

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        if encoded_state:
            self.obs_state = np.zeros((self.max_size, obs_state_dim))
            self.next_obs_state = np.zeros((self.max_size, obs_state_dim))


    def add(self, state, action, next_state, reward, done, obs_state=None, next_obs_state=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        if self.encoded_state:
            self.obs_state[self.ptr] = obs_state
            self.next_obs_state[self.ptr] = next_obs_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self, batch_size=None):
        ind = np.random.randint(0, self.crt_size, size=batch_size or self.batch_size)
        if not self.encoded_state:
            return(
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
            )
        else:
            return(
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device),
                torch.FloatTensor(self.obs_state[ind]).to(self.device),
                torch.FloatTensor(self.next_obs_state[ind]).to(self.device)
            )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)
        if self.encoded_state:
            np.save(f"{save_folder}_obs_state.npy", self.obs_state[:self.crt_size])
            np.save(f"{save_folder}_next_obs_state.npy", self.next_obs_state[:self.crt_size])

    def load(self, save_folder, size=-1, bootstrap=False):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]
        if self.encoded_state:
            self.obs_state[:self.crt_size] = np.load(f"{save_folder}_obs_state.npy")[:self.crt_size]
            self.next_obs_state[:self.crt_size] = np.load(f"{save_folder}_next_obs_state.npy")[:self.crt_size]

        if bootstrap:
            # Get the indicies of the above arrays that are non-zero
            nonzero_ind = (self.reward !=0)[:,0]
            num_nonzero = sum(nonzero_ind)
            self.state[self.crt_size:(self.crt_size+num_nonzero)] = self.state[nonzero_ind]
            self.action[self.crt_size:(self.crt_size+num_nonzero)] = self.action[nonzero_ind]
            self.next_state[self.crt_size:(self.crt_size+num_nonzero)] = self.next_state[nonzero_ind]
            self.reward[self.crt_size:(self.crt_size+num_nonzero)] = self.reward[nonzero_ind]
            self.not_done[self.crt_size:(self.crt_size+num_nonzero)] = self.not_done[nonzero_ind]
            if self.encoded_state:
                self.obs_state[self.crt_size:(self.crt_size+num_nonzero)] = self.obs_state[nonzero_ind]
                self.next_obs_state[self.crt_size:(self.crt_size+num_nonzero)] = self.next_obs_state[nonzero_ind]
            
            self.crt_size += num_nonzero

            neg_ind = (self.reward < 0)[:,0]
            num_neg = sum(neg_ind)
            self.state[self.crt_size:(self.crt_size+num_neg)] = self.state[neg_ind]
            self.action[self.crt_size:(self.crt_size+num_neg)] = self.action[neg_ind]
            self.next_state[self.crt_size:(self.crt_size+num_neg)] = self.next_state[neg_ind]
            self.reward[self.crt_size:(self.crt_size+num_neg)] = self.reward[neg_ind]
            self.not_done[self.crt_size:(self.crt_size+num_neg)] = self.not_done[neg_ind]
            if self.encoded_state:
                self.obs_state[self.crt_size:(self.crt_size+num_neg)] = self.obs_state[neg_ind]
                self.next_obs_state[self.crt_size:(self.crt_size+num_neg)] = self.next_obs_state[neg_ind]

            self.crt_size += num_neg

        print(f"Replay Buffer loaded with {self.crt_size} elements.")