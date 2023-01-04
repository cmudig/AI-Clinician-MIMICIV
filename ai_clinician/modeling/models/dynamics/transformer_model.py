import numpy as np
import os
import pandas as pd
import tqdm
import matplotlib
import gc

import torch
import datetime as dt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch.nn.init as weight_init
from torch.optim import Adam

class TransformerLatentSpaceModel(nn.Module):
    def __init__(self, embed_dim, nhead, nlayers, dropout):
        super().__init__()
        self.encoder_layers = []
        self.norm_layers = []
        for i in range(nlayers):
            l = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
            norm = nn.LayerNorm(embed_dim)
            self.encoder_layers.append(l)
            self.norm_layers.append(norm)
            self.add_module(f"attention_{i}", l)
            self.add_module(f"norm_{i}", norm)
        # Just don't use an encoder, pass the raw data directly into the transformer
        # self.encoder = nn.Linear( # nn.Embedding(ntoken, d_model)
        self.embed_dim = embed_dim
        self.init_weights()
        
    def init_weights(self):
        pass
        #initrange = 0.1
        #for l in self.encoder_layers:
        #    l.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, embed_dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, seqlen, embed_dim]
        """
        src = src.permute(1, 0, 2) # .reshape((-1, seq_len, src.size(2)))
        # src = self.pos_encoder(src)
        for l, norm in zip(self.encoder_layers, self.norm_layers):
            transformed, _ = l(src, src, src, attn_mask=src_mask[:src.size(0),:src.size(0)])
            src = norm(transformed + src)
        final_embed = src.permute(1, 0, 2)
        return final_embed
    
class TransformerDynamicsModel(nn.Module):
    """
    A torch module that embeds a sequence of patient states, actions, and
    demographics, then contextualizes them using multi-headed self attention
    blocks.
    """

    def __init__(self, state_dim, demog_dim, action_dim, embed_dim, nhead,
                 nlayers, dropout = 0.5, device='cpu'):
        super().__init__()
        self.model_type = 'Transformer'
        self.state_dim = state_dim
        self.demog_dim = demog_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.demog_embedding = nn.Linear(demog_dim, embed_dim)
        self.state_demog = nn.Linear(embed_dim * 2, embed_dim)
        self.action_embedding = nn.Linear(action_dim, embed_dim) # Incorporate the actions at a later layer so we have a state-only embedding
        self.state_action = nn.Linear(embed_dim * 2, embed_dim)
        self.state_transformer = TransformerLatentSpaceModel(embed_dim, nhead, nlayers, dropout)
        self.state_action_transformer = TransformerLatentSpaceModel(embed_dim, nhead, nlayers, dropout)
        self.device = device

        self.init_weights()

    def init_weights(self):
        self.action_embedding.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.action_embedding.weight.data)  # uniform_(-initrange, initrange)
        self.state_embedding.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.state_embedding.weight.data)  # uniform_(-initrange, initrange)
        self.demog_embedding.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.demog_embedding.weight.data)  # uniform_(-initrange, initrange)
        self.state_demog.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.state_demog.weight.data)  # uniform_(-initrange, initrange)
        self.state_action.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.state_action.weight.data)  # uniform_(-initrange, initrange)
        self.state_transformer.init_weights()
        self.state_action_transformer.init_weights()

    def encode(self, state, demog):
        """
        state: (N, L, S) where S is the state_dim
        demog: (N, L, D) where D is the demog_dim
        """
        state_embed = F.leaky_relu(self.state_embedding(state))
        demog_embed = F.leaky_relu(self.demog_embedding(demog))
        combined = torch.cat((state_embed, demog_embed), 2)
        return self.state_demog(combined)
    
    def forward(self, state, demog, action, src_mask):
        """
        Args:
            state: (N, L, S) where S is the state_dim
            demog: (N, L, D) where D is the demog_dim
            action: (N, L, A) where A is the action_dim
            src_mask: (L, L) indicating which self-attentions should be masked

        Returns:
            state + demog embedding, state + demog + action embedding, and
            contextualized state + demog + action embedding
        """
        initial_embed = self.encode(state, demog)
        
        # Run through the FIRST transformer
        state_embed = self.state_transformer(initial_embed, src_mask)
        
        action_embed = self.action_embedding(action)
        sa_embed = self.state_action(torch.cat((initial_embed, action_embed), 2))
        
        final_embed = self.state_action_transformer(sa_embed, src_mask)
        
        return initial_embed, state_embed, final_embed

def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
### Pretext Tasks

class StatePredictionModel(nn.Module):
    """
    A model that takes a dynamics model embedding as output and predicts an
    aleatoric uncertainty-aware next state.
    """
    
    def __init__(self, 
                 state_dim, 
                 embed_dim, 
                 discrete=False,
                 predict_variance=False,
                 variance_regularizer=1.0, 
                 timestep_weight_eps=0.1, 
                 predict_delta=True, 
                 num_steps=1,
                 num_layers=1,
                 loss_fn=None, 
                 target_transform=None,
                 device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers - 1)] + [nn.Linear(embed_dim, state_dim)])
        self.discrete = discrete
        self.predict_variance = predict_variance
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif not self.discrete:
            if self.predict_variance:
                self.variance_decoder = nn.Linear(embed_dim, state_dim)
                self.variance_regularizer = variance_regularizer
                self.timestep_weight_eps = timestep_weight_eps
                self.loss_fn = None
            else:
                self.loss_fn = nn.MSELoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            
        self.predict_delta = predict_delta
        self.num_steps = num_steps
        self.device = device
        self.target_transform = target_transform
        
    def init_weights(self):
        for l in self.decoder:
            l.bias.data.zero_()
            torch.nn.init.xavier_normal_(l.weight.data) # uniform_(-initrange, initrange)
        if not self.discrete and self.predict_variance:
            self.variance_decoder.bias.data.fill_(1.0)
            torch.nn.init.xavier_normal_(self.variance_decoder.weight.data) # uniform_(-initrange, initrange)
        
    def forward(self, embedding):
        for l in self.decoder[:-1]:
            embedding = F.leaky_relu(embedding)
        output_mu = self.decoder[-1](embedding)
        if self.discrete or not self.predict_variance:
            return output_mu
        else:
            logvar = self.variance_decoder(embedding)
            return output_mu, logvar, torch.distributions.Normal(output_mu, F.softplus(logvar) ** 0.5)
    
    def compute_loss(self, in_batch, model_outputs):
        """
        Compute the NLL loss of the model's outputs compared to the next states
        found in the inputs.
        
        in_batch: tuple (state, demog, action, missing, rewards, values, seq_len)
        model_outputs: tuple containing mean, logvar, and 
            torch.distributions.Normal of shape (N, L, S) where S is state_dim
        """
        in_state, _, _, in_missing_mask, _, _, seq_lens = in_batch
        L = in_state.shape[1]
        assert L > self.num_steps

        next_state_vec = in_state[:,self.num_steps:,:].clone()
        if self.predict_delta:
            next_state_vec -= in_state[:,:in_state.shape[1] - self.num_steps,:]
        if self.num_steps > 0:
            next_state_vec = torch.cat((next_state_vec, torch.zeros(next_state_vec.shape[0], self.num_steps, next_state_vec.shape[2]).to(self.device)), 1)

        if self.target_transform is not None:
            next_state_vec = self.target_transform(next_state_vec)
            
        if self.discrete or not self.predict_variance:
            overall_loss = self.loss_fn(model_outputs, next_state_vec)
        elif not self.predict_variance:
            if self.predict_delta:
                timestep_weights = self.timestep_weight_eps + next_state_vec ** 2  # Upweight timesteps with more change
            else:
                timestep_weights = torch.ones_like(next_state_vec).to(self.device)
            overall_loss = self.loss_fn(model_outputs, next_state_vec)
            overall_loss *= timestep_weights
        else:    
            mu, logvar, distro = model_outputs
            
            if self.predict_delta:
                timestep_weights = self.timestep_weight_eps + next_state_vec ** 2  # Upweight timesteps with more change
            else:
                timestep_weights = torch.ones_like(next_state_vec).to(self.device)
            neg_log_likelihood = -distro.log_prob(next_state_vec)
            overall_loss = neg_log_likelihood + self.variance_regularizer * (F.softplus(logvar) ** 0.5)
            overall_loss *= timestep_weights
            
        loss_mask = torch.logical_and(torch.arange(L)[None, :, None].to(self.device) < seq_lens[:, None, None] - self.num_steps,
                                        ~in_missing_mask)

        loss_masked = overall_loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        return loss_masked.sum() / loss_mask.sum()

class FullyConnected2Layer(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, dropout=0.1, batch_norm=False):
        super().__init__()
        self.l1 = nn.Linear(in_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(latent_dim, out_dim)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm1 = nn.BatchNorm1d(latent_dim)
        self.init_weights()

    def init_weights(self):
        self.l1.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.l1.weight.data)  # uniform_(-initrange, initrange)
        self.l2.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.l2.weight.data)  # uniform_(-initrange, initrange)

    def forward(self, state):
        out = F.leaky_relu(self.l1(self.dropout(state)))
        if self.batch_norm:
            if len(out.shape) == 3:
                out = torch.transpose(self.norm1(torch.transpose(out, 1, 2)), 1, 2)
            else:
                out = self.norm1(out)
        return self.l2(out)
    
class ValuePredictionModel(nn.Module):
    def __init__(self, embed_dim, predict_rewards=False, hidden_dim=16, dropout=0.1, batch_norm=False, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = FullyConnected2Layer(embed_dim, hidden_dim, 1, dropout=dropout, batch_norm=batch_norm)
        self.predict_rewards = predict_rewards
        self.loss_fn = nn.MSELoss(reduction='none')
        self.device = device
        
    def forward(self, embedding):
        return self.net(embedding)
    
    def compute_loss(self, in_batch, model_outputs):
        """
        Compute the MSE loss of the return or reward compared to the model output.
        
        in_batch: tuple (state, demog, action, missing, rewards, values, seq_len)
        model_outputs: (N, L, 1)
        """
        obs, _, _, _, rewards, values, seq_lens = in_batch
        L = obs.shape[1]
        loss = self.loss_fn(model_outputs, rewards if self.predict_rewards else values).sum(2)
        
        loss_mask = torch.arange(L).to(self.device)[None, :] < seq_lens[:, None]

        loss_masked = loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        return loss_masked.sum() / loss_mask.sum()
    
class TerminationPredictionModel(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, hidden_dim=16, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = FullyConnected2Layer(embed_dim, hidden_dim, 1, dropout=dropout)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.device = device
        
    def forward(self, embedding):
        return self.net(embedding)
    
    def compute_loss(self, in_batch, model_outputs):
        """
        Compute the MSE loss of the return or reward compared to the model output.
        
        in_batch: tuple (state, demog, action, missing, rewards, values, seq_len)
        model_outputs: (N, L, 1)
        """
        obs, _, _, _, _, _, seq_lens = in_batch
        L = obs.shape[1]
        labels = (torch.arange(L).to(self.device)[None, :] == seq_lens[:, None] - 1).float()
        
        loss = self.loss_fn(model_outputs.squeeze(2), labels)
        loss_weights = (labels * (0.5 * seq_lens[:, None]) + (1 - labels) * (0.5 * seq_lens[:, None] / (seq_lens[:, None] - 1))).to(self.device)
        loss *= loss_weights
        
        loss_mask = torch.arange(L).to(self.device)[None, :] < seq_lens[:, None]

        loss_masked = loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        
        term_correct = ((torch.round(torch.sigmoid(model_outputs.squeeze(2))) == labels) & loss_mask).sum().item()
        term_total = loss_mask.sum().item()

        return loss_masked.sum() / loss_mask.sum(), term_correct, term_total
    
class ActionPredictionModel(nn.Module):
    """
    A model that predicts the action that gave rise to the transition between
    pairs of states.
    """
    def __init__(self, embed_dim, action_dim, discrete=False, num_bins_per_action=None, dropout=0.1, device='cpu', training_fraction=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = FullyConnected2Layer(embed_dim * 2, embed_dim, action_dim, dropout=dropout)
        self.discrete = discrete
        self.training_fraction = training_fraction
        if discrete:
            self.loss_fn = nn.CrossEntropyLoss()
            self.num_bins_per_action = num_bins_per_action
        else:
            self.loss_fn = nn.MSELoss()
        self.device = device
        
    def create_batch(self, embeddings, seq_lens, actions):
        """
        Creates a batch consisting of each pair of actions from the given 
        sequence of embeddings, matched to the actions that were taken.
        
        embeddings: (N, L, E)
        seq_lens: N
        actions: (N, L, A)
        
        Output:
            embeddings: (N', E * 2) where N' is sum (seq_len[i] - 1) over all i
            actions: (N', A)
        """
        transitions = []
        returned_actions = []
        idxs_to_use = np.random.choice(np.arange(embeddings.shape[0]), 
                                       size=int(np.ceil(embeddings.shape[0] * self.training_fraction)), 
                                       replace=False)
        for i in idxs_to_use:
            L = seq_lens[i].item()
            transitions.append(torch.cat((embeddings[i,:L - 1,:], embeddings[i,1:L,:]), 1))
            returned_actions.append(actions[i,:L - 1,:])
        return torch.cat(transitions, 0), torch.cat(returned_actions, 0)
        
    def forward(self, embedding):
        return self.net(embedding)
    
    def compute_loss(self, in_batch, model_outputs):
        """
        Compute the MSE loss of the return or reward compared to the model output.
        
        in_batch: tuple (transitions, actions) generated by create_batch
        model_outputs: outputs of forward
        """
        _, actions = in_batch
        if self.discrete and self.num_bins_per_action is not None:
            # Add individual components spaced at num_bins_per_action
            return sum(self.loss_fn(pred, true) for pred, true in zip(
                torch.split(model_outputs, self.num_bins_per_action, 1),
                torch.split(actions, self.num_bins_per_action, 1)))
        return self.loss_fn(model_outputs, actions)
    
    
class SubsequentBinaryPredictionModel(nn.Module):
    """
    A model that predicts whether pairs of final and initial embeddings come from
    subsequent states.
    """
    def __init__(self, embed_dim, positive_label_fraction=0.5, hidden_dim=16, training_fraction=0.1, dropout=0.1, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = FullyConnected2Layer(embed_dim * 2, hidden_dim, 1, dropout=dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.positive_label_fraction = positive_label_fraction
        self.training_fraction = training_fraction
        self.device = device
        
    def create_batch(self, final_embeddings, initial_embeddings, seq_lens):
        """
        Creates a batch consisting of each pair of actions from the given 
        sequence of embeddings, matched to the actions that were taken.
        
        final_embeddings: (N, L, E)
        initial_embeddings: (N, L, E)
        seq_lens: N
        
        Output:
            embeddings: (N', E * 2) where N' is sum (seq_len[i] - 1) over all i
            labels: (N')
        """
        final_flat = []
        initial_flat = []
        idxs_to_use = np.random.choice(np.arange(final_embeddings.shape[0]), 
                                       size=int(np.ceil(final_embeddings.shape[0] * self.training_fraction)), 
                                       replace=False)
        for i in idxs_to_use:
            L = seq_lens[i].item()
            final_flat.append(final_embeddings[i,:L - 1,:])
            initial_flat.append(initial_embeddings[i,1:L,:])
        
        final_flat = torch.cat(final_flat, 0)
        initial_flat = torch.cat(initial_flat, 0)
        left_side = np.arange(final_flat.shape[0])
        right_side = np.arange(final_flat.shape[0])

        while (left_side == right_side).mean() > 1 - self.positive_label_fraction:
            random_pair = np.random.choice(len(left_side), size=2, replace=False)
            left_side[random_pair] = left_side[np.flip(random_pair)]
            random_pair = np.random.choice(len(right_side), size=2, replace=False)
            right_side[random_pair] = right_side[np.flip(random_pair)]
            
        return final_flat[left_side], initial_flat[right_side], torch.FloatTensor(left_side == right_side).unsqueeze(1).to(self.device)
        
    def forward(self, left_side, right_side):
        return self.net(torch.cat((left_side, right_side), 1))
    
    def compute_loss(self, in_batch, model_outputs):
        """
        Compute the MSE loss of the return or reward compared to the model output.
        
        in_batch: tuple (transitions, actions) generated by create_batch
        model_outputs: outputs of forward
        """
        _, _, labels = in_batch
        return self.loss_fn(model_outputs, labels)
    
    
class MultitaskDynamicsModel:
    """
    An object which manages a transformer-based dynamics model with several
    pretext tasks.
    """
    def __init__(self,
                 obs_size,
                 dem_size,
                 embed_size=512,
                 nlayers=3,
                 nhead=4,
                 dropout=0.0,
                 value_dropout=0.1,
                 value_input_size=512,
                 action_training_fraction=0.2,
                 batch_size=32,
                 max_seq_len=160,
                 reward_batch_norm=True,
                 fc_hidden_dim=16,
                 mask_prob=0.5,
                 device='cpu',
                 replacement_values=None):
        
        super().__init__()
        self.model = TransformerDynamicsModel(
            obs_size, 
            dem_size,
            2, 
            embed_size, 
            nhead, 
            nlayers, 
            dropout, 
            device=device).to(device)

        self.current_state_model = StatePredictionModel(
            obs_size, 
            embed_size, 
            predict_delta=False, 
            num_steps=0, 
            device=device).to(device)
        self.next_state_model = StatePredictionModel(
            obs_size, 
            embed_size, 
            predict_delta=False, 
            num_steps=1, 
            device=device).to(device)
        self.value_input_size = value_input_size
        if self.value_input_size < embed_size:
            self.value_input = nn.Linear(embed_size, self.value_input_size)
        else:
            self.value_input = None
        self.reward_model = ValuePredictionModel(
            self.value_input_size, 
            predict_rewards=True, 
            device=device, 
            batch_norm=reward_batch_norm,
            hidden_dim=fc_hidden_dim,
            dropout=value_dropout).to(device)
        self.return_model = ValuePredictionModel(
            self.value_input_size, 
            predict_rewards=False, 
            device=device, 
            batch_norm=reward_batch_norm,
            hidden_dim=fc_hidden_dim,
            dropout=value_dropout).to(device)
        self.action_model = ActionPredictionModel(
            embed_size, 
            2, 
            device=device, 
            training_fraction=action_training_fraction, 
            dropout=dropout).to(device)
        self.subsequent_model = SubsequentBinaryPredictionModel(
            self.value_input_size, 
            device=device, 
            training_fraction=action_training_fraction,
            hidden_dim=fc_hidden_dim, 
            dropout=dropout).to(device)
        self.termination_model = TerminationPredictionModel(
            self.value_input_size, 
            dropout=dropout, 
            hidden_dim=fc_hidden_dim,
            device=device).to(device)
        
        self.overall_model = nn.ModuleList([
            self.model, 
            self.current_state_model, 
            self.next_state_model] + (
                [self.value_input] if self.value_input is not None else []
            ) + [
            self.reward_model, 
            self.return_model, 
            self.action_model, 
            self.subsequent_model, 
            self.termination_model
        ])
        print("Model has", sum(p.numel() for p in self.overall_model.parameters()), "parameters")

        self.device = device
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.replacement_values = replacement_values
        
        self.loss_weights = (torch.FloatTensor([1, 1, 0.2, 0.05, 1, 1, 1, 1]) * 0.1).to(device)
        
        self.consistency_offset = 1
        self.consistency_subloss = nn.MSELoss(reduction='none')

    def _consistency_loss(self, final_embed, initial_embed):
        return torch.cat((self.consistency_subloss(final_embed[:,:-self.consistency_offset],
                                                   initial_embed[:,self.consistency_offset:]), 
                         torch.zeros(final_embed.shape[0],
                                     self.consistency_offset, 
                                     final_embed.shape[2]).to(self.device)), 1)
        
    def compute_loss(self, batch, mask=True, return_elements=False, return_accuracies=False):
        obs, dem, ac, missing, rewards, discounted_rewards, in_lens = batch
        
        src_mask = generate_square_subsequent_mask(self.max_seq_len).to(self.device) # torch.ones(seq_len, seq_len).to(device)
        
        obs = obs.to(self.device)
        dem = dem.to(self.device)
        ac = ac.to(self.device)
        missing = missing.to(self.device)
        rewards = rewards.to(self.device)
        discounted_rewards = discounted_rewards.to(self.device)
        in_lens = in_lens.to(self.device)
        batch = (obs, dem, ac, missing, rewards, discounted_rewards, in_lens)

        if mask:
            assert self.replacement_values is not None
            should_mask = torch.logical_and(torch.rand(*obs.shape).to(self.device) < self.mask_prob, ~missing)
            masked_obs = torch.where(should_mask, torch.from_numpy(self.replacement_values).float().to(self.device), obs)
        else:
            masked_obs = obs

        # Run transformer
        _, initial_embed, final_embed = self.model(masked_obs, dem, ac, src_mask)
        
        # 1. Masked prediction model
        pred_curr = self.current_state_model(initial_embed)
        curr_state_loss = self.current_state_model.compute_loss(batch, pred_curr)
        
        # 2. Next-state prediction model
        pred_next = self.next_state_model(final_embed)
        next_state_loss = self.next_state_model.compute_loss(batch, pred_next)
        
        val_final_input = final_embed if self.value_input is None else F.leaky_relu(self.value_input(final_embed))
        val_initial_input = initial_embed if self.value_input is None else F.leaky_relu(self.value_input(initial_embed))
        
        # 3. Reward model
        pred_reward = self.reward_model(val_final_input)
        reward_loss = self.reward_model.compute_loss(batch, pred_reward)
        
        # 4. Return model
        pred_return = self.return_model(val_final_input)
        return_loss = self.return_model.compute_loss(batch, pred_return)

        # 5. Consistency loss
        c_loss = self._consistency_loss(final_embed, initial_embed)
        loss_mask = torch.arange(c_loss.shape[1])[None, :, None].to(self.device) < in_lens[:, None, None]
        c_loss_masked = c_loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        c_loss = c_loss_masked.sum() / loss_mask.sum()
        
        # 6. Action prediction loss
        action_batch = self.action_model.create_batch(initial_embed, in_lens, ac)
        pred_action = self.action_model(action_batch[0])
        action_loss = self.action_model.compute_loss(action_batch, pred_action)
        
        # 7. Is subsequent action loss
        subsequent_batch = self.subsequent_model.create_batch(val_final_input, val_initial_input, in_lens)
        pred_subs = self.subsequent_model(subsequent_batch[0], subsequent_batch[1])
        subs_loss = self.subsequent_model.compute_loss(subsequent_batch, pred_subs)
        subs_correct = (torch.round(torch.sigmoid(pred_subs)) == subsequent_batch[2]).sum().item()
        subs_total = subsequent_batch[2].shape[0]
        
        # 8. Termination model
        pred_term = self.termination_model(val_final_input)
        term_loss, corr, tot = self.termination_model.compute_loss(batch, pred_term)
        term_correct = corr
        term_total = tot

        itemized_loss = self.loss_weights * torch.stack((
            curr_state_loss, 
            next_state_loss, 
            reward_loss, 
            return_loss, 
            c_loss, 
            action_loss, 
            subs_loss, 
            term_loss
        ))
        
        result = itemized_loss if return_elements else itemized_loss.sum()
        if return_accuracies:
            result = (result, (subs_correct, subs_total), (term_correct, term_total))
        return result
    
    def train(self):
        self.overall_model.train()
        
    def eval(self):
        self.overall_model.eval()
        
    def state_dict(self):
        return {
            "dynamics": self.model.state_dict(),
            "current_state": self.current_state_model.state_dict(),
            "next_state": self.next_state_model.state_dict(),
            "reward": self.reward_model.state_dict(),
            "return": self.return_model.state_dict(),
            "action": self.action_model.state_dict(),
            "subsequent": self.subsequent_model.state_dict(),
            "termination": self.termination_model.state_dict()
        }