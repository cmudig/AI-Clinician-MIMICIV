import numpy as np
import tqdm
import pickle

from preprocessing.columns import *
from modeling.columns import *
from modeling.models.base_ import BaseModel
from modeling.models.torch_modules import SparseAutoencoder, DuelingDQN, PrioritizedBuffer, embed_autoencoder, test_autoencoder, train_autoencoder

import torch
from torch.utils.data import TensorDataset, DataLoader
        
class PERAgent:

    def __init__(self, obs_space_dim, action_space_dim, learning_rate=3e-4, gamma=0.99, tau=0.01, lambda_=0.1, max_reward=15, buffer_size=10000, batch_size=64):
        self.obs_space_dim = obs_space_dim
        self.action_space_dim = action_space_dim
        self.gamma = gamma
        self.tau = tau
        self.lambda_ = lambda_
        self.max_reward = max_reward
        self.batch_idx = 0
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        
        # exp_per_sampling = np.ceil(batch_size * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)
        self.replay_buffer = PrioritizedBuffer(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.batch_size = batch_size
	
        self.model = DuelingDQN(obs_space_dim, 128, action_space_dim).to(self.device)
        self.target_model = DuelingDQN(obs_space_dim, 128, action_space_dim).to(self.device)
        
        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)
          
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.MSE_loss = torch.nn.MSELoss()

    def get_action(self, state, eps=0.0):
        self.model.eval()
        state = torch.FloatTensor(state).float().to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy(), axis=1)
        
        if np.random.rand() < eps:
            return np.random.choice(self.action_space_dim)
          
        return action

    def experience(self, states, actions, terminal_state, reward):
        for i, (state, action) in enumerate(zip(states[:-1], actions[:-1])):
            self.replay_buffer.push(state, action, 0, states[i + 1], 0)
        # Now, add an experience taking the last state to the terminal state,
        # with the given reward. 
        self.replay_buffer.push(states[-1], actions[-1], reward, terminal_state, 1)
        # Maybe also add a final experience from the terminal
        # state to the same state with a random action and 0 reward

    def _compute_TDerror(self):
        transitions, idxs, IS_weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = transitions

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        IS_weights = torch.FloatTensor(IS_weights).to(self.device)

        curr_Q = self.model(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        with torch.no_grad():
            next_Q = torch.clamp(self.target_model(next_states), -self.max_reward, self.max_reward)
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        diffs = curr_Q - expected_Q
        losses = torch.pow(diffs, 2) * IS_weights
        
        # Add a term penalizing Q values greater than max reward
        q_mag_loss = self.lambda_ * torch.clamp(torch.abs(curr_Q) - self.max_reward, min=0)

        return losses, q_mag_loss, torch.abs(diffs), idxs

    def learn(self):
        self.batch_idx += 1
        self.model.train()
        
        losses, q_mag_loss, td_errors, idxs = self._compute_TDerror()

        # update model
        loss = (losses + q_mag_loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        for idx, td_error in zip(idxs, td_errors.cpu().detach().numpy()):
            self.replay_buffer.update_priority(idx, td_error)
            
        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            
        return loss.detach().numpy(), q_mag_loss.mean().detach().numpy()
    
    def state_dict(self):
        """
        Generates a state dictionary that can be used to reload the PERAgent
        later.
        """
        return {
            "obs_space_dim": self.obs_space_dim,
            "action_space_dim": self.action_space_dim,
            "gamma": self.gamma,
            "tau": self.tau,
            "lambda_": self.lambda_,
            "max_reward": self.max_reward,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "model": self.model.state_dict()
        }
        
    @staticmethod
    def from_state_dict(data):
        agent = PERAgent(
            data["obs_space_dim"],
            data["action_space_dim"],
            learning_rate=data["learning_rate"],
            gamma=data["gamma"],
            tau=data["tau"],
            lambda_=data["lambda_"],
            max_reward=data["max_reward"],
            buffer_size=data["buffer_size"],
            batch_size=data["batch_size"]
        )
        agent.model.load_state_dict(data["model"])
        agent.target_model.load_state_dict(data["model"])
        return agent

class DuelingDQNModel(BaseModel):
    """
    Subclass of BaseModel that implements a double dueling deep Q-network with
    prioritized experience replay, following Raghu et al.
    """
    def __init__(self, 
                 state_dim=256,
                 n_actions=25,
                 hidden_dim=128,
                 gamma=0.99,
                 lambda_=0.1,
                 tau=0.01,
                 sparsity_factor=0.5, # beta
                 reward_val=5,
                 soften_factor=0.01,
                 n_state_epochs=40,
                 n_agent_epochs=2,
                 batch_size=64,
                 learning_rate=3e-4,
                 experience_buffer_size=100000,
                 experience_update_interval=10,
                 random_state=None,
                 metadata=None):
        super(DuelingDQNModel, self).__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lambda_ = lambda_
        self.tau = tau
        self.sparsity_factor = sparsity_factor
        self.reward_val = reward_val
        self.soften_factor = soften_factor
        self.random_state = random_state
        self.n_state_epochs = n_state_epochs
        self.n_agent_epochs = n_agent_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.experience_buffer_size = experience_buffer_size
        self.experience_update_interval = experience_update_interval
        self.metadata = metadata
        
        self.input_dim = None
        self.autoencoder = None
        self.agent = None
        
    def _build_autoencoder(self, train_states, val_states):
        """
        Initializes and trains the sparse autoencoder to learn an embedded
        representation of the patient state.
        """
        self.autoencoder = SparseAutoencoder(
            self.input_dim,
            latent_size=self.state_dim
        )

        train_state_loader = DataLoader(train_states, batch_size=self.batch_size, shuffle=True)
        val_state_loader = DataLoader(val_states, batch_size=self.batch_size, shuffle=True)

        last_loss = 1e9
        for epoch in range(self.n_state_epochs):
            model_dict = {"model": self.autoencoder}
            self.autoencoder.train()
            train_autoencoder(model_dict, train_state_loader, None, self.sparsity_factor)
            self.autoencoder.eval()
            new_loss, _ = test_autoencoder(model_dict, val_state_loader, None, self.sparsity_factor)
            if new_loss["model"] > last_loss:
                print("Early stopping")
                break
            last_loss = new_loss["model"]
        
    def _build_agent(self, train_embs, actions_train, metadata_train, val_embs, actions_val):
        """
        Initializes and trains the RL agent using a deep Q network.
        """
        self.agent = PERAgent(self.state_dim + 2,
                              self.n_actions,
                              learning_rate=self.learning_rate,
                              buffer_size=self.experience_buffer_size,
                              lambda_=self.lambda_,
                              tau=self.tau,
                              max_reward=self.reward_val)

        # For the agent, we will add two state dimensions to indicate whether 
        # the patient was discharged or died
        val_emb_dataset = TensorDataset(torch.from_numpy(np.hstack([
            val_embs,
            np.zeros((val_embs.shape[0], 2))
        ])).float())
        val_emb_loader = DataLoader(val_emb_dataset, batch_size=self.batch_size, shuffle=False)

        agree_percentage = 0.0

        stay_ids_train = metadata_train[C_ICUSTAYID].values
        outcomes_train = metadata_train[C_OUTCOME].values
        for epoch in range(self.n_agent_epochs):
            unique_stay_ids = np.random.permutation(np.unique(stay_ids_train))
            pbar = tqdm.tqdm(unique_stay_ids)
            total_loss = 0.0
            total_mag_loss = 0.0
            total_examples = 0
            trajectories_added = 0
            total_trajectories = 0

            for stay_id in pbar:
                mask = stay_ids_train == stay_id
                first_idx = np.argmax(mask)
                outcome = outcomes_train[first_idx]
                assert all(o == outcomes_train[first_idx] for o in outcomes_train[mask])
                self.agent.experience(np.hstack([
                    train_embs[mask],
                    np.zeros((mask.sum(), 2))
                ]), actions_train[mask], np.concatenate([
                    np.zeros(train_embs.shape[1]),
                    [outcome == 0, outcome == 1]
                ]), self.reward_val * (-1 if outcome else 1))
                trajectories_added += 1
                total_trajectories += 1

                if total_trajectories % 1000 == 0:
                    # Calculate percentage agreement between learned policy and actual val actions
                    chosen_actions = []
                    with torch.no_grad():
                        for (val_states,) in val_emb_loader:
                            chosen_actions.append(self.agent.get_action(val_states))
                    chosen_actions = np.concatenate(chosen_actions)
                    agree_percentage = (chosen_actions == actions_val).mean() * 100

                if len(self.agent.replay_buffer) >= self.batch_size and trajectories_added >= self.experience_update_interval:
                    trajectories_added = 0
                    
                    loss, mag_loss = self.agent.learn()
                    total_loss += loss
                    total_mag_loss += mag_loss
                    total_examples += self.batch_size
                    pbar.set_description("Epoch {} Loss {:.4f} Mag loss: {:.4f} Agree {:.2f}%".format(epoch + 1, total_loss / total_examples, total_mag_loss / total_examples, agree_percentage))
                
    def train(self, X_train, actions_train, metadata_train, X_val=None, actions_val=None, metadata_val=None):
        # First train the autoencoder
        self.input_dim = X_train.shape[1]
        train_inputs = TensorDataset(torch.from_numpy(X_train).float())
        val_inputs = TensorDataset(torch.from_numpy(X_val).float())
        self._build_autoencoder(train_inputs, val_inputs)
        
        train_embs = embed_autoencoder(self.autoencoder, DataLoader(train_inputs, batch_size=self.batch_size, shuffle=False))
        val_embs = embed_autoencoder(self.autoencoder, DataLoader(val_inputs, batch_size=self.batch_size, shuffle=False))
        self._build_agent(train_embs, actions_train, metadata_train, val_embs, actions_val)

    def compute_states(self, X):
        assert self.autoencoder is not None, "DQN autoencoder is not trained"
        inputs = TensorDataset(torch.from_numpy(X).float())
        loader = DataLoader(inputs, batch_size=self.batch_size, shuffle=False)
        return embed_autoencoder(self.autoencoder, loader)
    
    def compute_Q(self, X=None, states=None, actions=None):
        assert X is not None or states is not None, "At least one of states or X must not be None"
        if states is None:
            states = self.compute_states(X)
        assert self.agent is not None, "RL agent is not trained"
        
        emb_dataset = TensorDataset(torch.from_numpy(np.hstack([
            states,
            np.zeros((states.shape[0], 2))
        ])))
        emb_loader = DataLoader(emb_dataset, batch_size=self.batch_size, shuffle=False)
        val_Q = []
        with torch.no_grad():
            for (val_states,) in tqdm.tqdm(emb_loader):
                val_Q.append(self.agent.model(val_states.float().to(self.agent.device)).detach().numpy())
        val_Q = np.clip(np.concatenate(val_Q), -self.reward_val, self.reward_val)
        if actions is not None:
            return val_Q[np.arange(len(val_Q)), actions]
        return val_Q
        
    def compute_V(self, X):
        val_Q = self.compute_Q(X=X)
        val_V = val_Q.mean(axis=1)
        return val_V
    
    def compute_probabilities(self, X=None, states=None, actions=None):
        val_Q = self.compute_Q(X=X, states=states)
        val_model_probabilities = (np.exp(val_Q) / np.sum(np.exp(val_Q), axis=1, keepdims=True))
        if actions is not None:
            return val_model_probabilities[np.arange(len(actions)), actions]
        return val_model_probabilities
    
    def save(self, filepath, metadata=None):
        """
        Saves the model as a pickle to the given filepath.
        """
        model_data = {
            'model_type': 'DuelingDQNModel',
            'autoencoder': self.autoencoder.state_dict(),
            'agent': self.agent.state_dict(),
            'input_dim': self.input_dim,
            'params': {
                 "state_dim": self.state_dim,
                 "n_actions": self.n_actions,
                 "hidden_dim": self.hidden_dim,
                 "gamma": self.gamma,
                 "lambda_": self.lambda_,
                 "tau": self.tau,
                 "sparsity_factor": self.sparsity_factor,
                 "reward_val": self.reward_val,
                 "soften_factor": self.soften_factor,
                 "n_state_epochs": self.n_state_epochs,
                 "n_agent_epochs": self.n_agent_epochs,
                 "batch_size": self.batch_size,
                 "learning_rate": self.learning_rate,
                 "experience_buffer_size": self.experience_buffer_size,
                 "experience_update_interval": self.experience_update_interval,
                 "random_state": self.random_state,
            }
        }
        if metadata is not None:
            model_data['metadata'] = metadata 
        torch.save(model_data, filepath)
    
    @classmethod
    def load(cls, filepath):
        model_data = torch.load(filepath)
        assert model_data['model_type'] == 'DuelingDQNModel', 'Invalid model type for DuelingDQNModel'
        model = cls(**model_data['params'])
        model.metadata = model_data.get('metadata', None)
        model.input_dim = model_data['input_dim']
        model.autoencoder = SparseAutoencoder(model.input_dim, latent_size=model.state_dim)
        model.autoencoder.load_state_dict(model_data['autoencoder'])
        model.agent = PERAgent.from_state_dict(model_data['agent'])
        return model