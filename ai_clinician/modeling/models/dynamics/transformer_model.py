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
    """
    Helper neural network that handles the multi-head attention (transformer)
    component of the network.
    
    Inputs:
        src: (N, L, E) - the embedded input data
        src_mask: (L, L) - indicates which positions should be allowed to attend
            to which other positions.
            
    Output: (N, L, E). The resulting embeddings are contextualized by the other
        elements in the sequence.
        
    N = batch size
    L = sequence length
    E = embedding dimension
    """
    def __init__(self, embed_dim, nhead, nlayers, dropout, positional_encoding=True):
        super().__init__()
        self.encoder_layers = []
        self.norm_layers = []
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.pos_encoder = PositionalEncoding(embed_dim, dropout)
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
        if self.positional_encoding:
            src = self.pos_encoder(src)
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
    
    Inputs:
        state: (N, L, O)
        demog: (N, L, D)
        action: (N, L, A)
        src_mask: (L, L) indicating which self-attentions should be masked

    Outputs:
        obs + demog embedding: (N, L, E)
        contextualized obs + demog embedding: (N, L, E)
        contextualized obs + demog + action embedding: (N, L, E)
        
    N = batch size
    L = sequence length
    O = dimension of the observation vector
    D = dimension of the demographics vector
    A = dimension of the action vector (usually 2)
    E = embedding dimension
    """

    def __init__(self, state_dim, demog_dim, action_dim, embed_dim, nhead,
                 nlayers, dropout = 0.5, device='cpu', positional_encoding=True):
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
        self.state_transformer = TransformerLatentSpaceModel(embed_dim, nhead, nlayers, dropout, positional_encoding=positional_encoding)
        self.state_action_transformer = TransformerLatentSpaceModel(embed_dim, nhead, nlayers, dropout, positional_encoding=False)
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
            state + demog embedding, contextualized state + demog embedding, and
            contextualized state + demog + action embedding
        """
        initial_embed = self.encode(state, demog)
        
        # Run through the FIRST transformer
        state_embed = self.state_transformer(initial_embed, src_mask)
        
        final_embed = self.unroll_from_embedding(state_embed, action, src_mask)
                
        return initial_embed, state_embed, final_embed
    
    def unroll_from_embedding(self, state_embed, action, src_mask):
        """
        Produces a state-action embedding from a state-only embedding and an action.
        """
        action_embed = self.action_embedding(action)
        sa_embed = self.state_action(torch.cat((state_embed, action_embed), 2))
        
        return self.state_action_transformer(sa_embed, src_mask)

class RNNDynamicsModel(nn.Module):
    """
    An alternative to TransformerDynamicsModel that uses an LSTM instead of
    a transformer architecture.
    """
    def __init__(self, state_dim, demog_dim, action_dim, embed_dim, num_lstm_layers=1, bidirectional=False, dropout=0.1, device='cpu'):
        super(RNNDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.demog_dim = demog_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.demog_embedding = nn.Linear(demog_dim, embed_dim)
        self.state_demog = nn.Linear(embed_dim * 2, embed_dim)
        self.state_rnn = nn.LSTM(embed_dim, embed_dim, num_layers=num_lstm_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.state_action = nn.Linear(embed_dim * 2, embed_dim)
        self.state_action_rnn = nn.LSTM(embed_dim, embed_dim, num_layers=num_lstm_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
                    
        self.device = device

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
            src_mask: unused

        Returns:
            state + demog embedding, contextualized state + demog embedding, and
            contextualized state + demog + action embedding
        """
        initial_embed = self.encode(state, demog)
        
        state_embed, _ = self.state_rnn(initial_embed, self.init_state(initial_embed.shape[0]))
        
        final_embed = self.unroll_from_embedding(state_embed, action, src_mask)
                
        return initial_embed, state_embed, final_embed
    
    def unroll_from_embedding(self, state_embed, action, src_mask):
        """
        Produces a state-action embedding from a state-only embedding and an action.
        """
        action_embed = self.action_embedding(action)
        sa_embed = self.state_action(torch.cat((state_embed, action_embed), 2))
        
        final_embed, _ = self.state_action_rnn(sa_embed, self.init_state(state_embed.shape[0]))
        return final_embed
    
    def init_state(self, batch_size):
        return (torch.zeros(self.num_lstm_layers * (2 if self.bidirectional else 1), batch_size, self.embed_dim).to(self.device), 
                torch.zeros(self.num_lstm_layers * (2 if self.bidirectional else 1), batch_size, self.embed_dim).to(self.device))
      
class LinearDynamicsModel(nn.Module):
    """
    An alternative to TransformerDynamicsModel that uses only linear layers, so
    the embeddings are not contextualized across the sequence.
    """
    def __init__(self, state_dim, demog_dim, action_dim, embed_dim, num_layers=1, dropout=0.1, device='cpu'):
        super(LinearDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.demog_dim = demog_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim

        self.state_embedding = nn.Linear(state_dim, embed_dim)
        self.demog_embedding = nn.Linear(demog_dim, embed_dim)
        self.state_demog = nn.Linear(embed_dim * 2, embed_dim)
        self.state_encoder = nn.ModuleList()
        for _ in range(num_layers):
            self.state_encoder.append(nn.Linear(embed_dim, embed_dim))

        self.action_embedding = nn.Linear(action_dim, embed_dim)
        self.state_action = nn.Linear(embed_dim * 2, embed_dim)
        self.state_action_encoder = nn.ModuleList()
        for _ in range(num_layers):
            self.state_action_encoder.append(nn.Linear(embed_dim, embed_dim))

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.device = device

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
            src_mask: unused

        Returns:
            state + demog embedding, contextualized state + demog embedding, and
            contextualized state + demog + action embedding
        """
        initial_embed = self.dropout(F.leaky_relu(self.encode(state, demog)))
        
        state_embed = initial_embed
        for layer in self.state_encoder:
            state_embed = self.dropout(F.leaky_relu(layer(state_embed)))
                    
        final_embed = self.unroll_from_embedding(state_embed, action, src_mask)
                
        return initial_embed, state_embed, final_embed
    
    def unroll_from_embedding(self, state_embed, action, src_mask):
        """
        Produces a state-action embedding from a state-only embedding and an action.
        """
        action_embed = self.action_embedding(action)
        sa_embed = self.state_action(torch.cat((state_embed, action_embed), 2))
        
        final_embed = sa_embed
        for layer in self.state_action_encoder:
            final_embed = self.dropout(F.leaky_relu(layer(final_embed)))
        return final_embed    
      

def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 5000, weight=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.weight = weight

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.weight * self.pe[:x.size(0)]
        return self.dropout(x)
    
### Pretext Tasks

class StatePredictionModel(nn.Module):
    """
    A model that takes a dynamics model's embedding output and predicts an
    aleatoric uncertainty-aware next state.
    
    Input: embedding (N, L, E) representing the contextualized embeddings
        for each timestep
        
    Output:
        If predict_variance is True: returns a mean next state prediction 
        (N, L, O), an estimate of the log-variance of the prediction (N, L, O),
        and a torch Distribution object that can be used to sample from the
        distribution defined by this mean and variance.
        If predict_variance is False: returns the mean next state prediction
        (N, L, O)
        
    N = batch size
    L = sequence length
    O = dimension of the observation vector
    E = embedding dimension
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
        self.timestep_weight_eps = timestep_weight_eps
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif not self.discrete:
            if self.predict_variance:
                self.variance_decoder = nn.Linear(embed_dim, state_dim)
                self.variance_regularizer = variance_regularizer
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
            return output_mu, logvar, torch.distributions.Normal(output_mu, F.softplus(logvar)) # ** 0.5)
    
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

        if self.predict_delta:
            timestep_weights = self.timestep_weight_eps + next_state_vec ** 2  # Upweight timesteps with more change
        else:
            timestep_weights = torch.ones_like(next_state_vec).to(self.device) # torch.clamp(self.timestep_weight_eps + next_state_vec ** 2, max=10.0)

        if self.target_transform is not None:
            next_state_vec = self.target_transform(next_state_vec)
            
        if self.discrete or not self.predict_variance:
            overall_loss = self.loss_fn(model_outputs, next_state_vec) * timestep_weights
        else:    
            mu, logvar, distro = model_outputs
            
            neg_log_likelihood = -distro.log_prob(next_state_vec) # + torch.abs(next_state_vec - in_state) * torch.clamp(distro.log_prob(in_state), min=-5, max=5)
            overall_loss = neg_log_likelihood + self.variance_regularizer * torch.log(F.softplus(logvar)) # ** 0.5)
            overall_loss *= timestep_weights
            
        loss_mask = torch.logical_and(torch.arange(L)[None, :, None].to(self.device) < seq_lens[:, None, None] - self.num_steps,
                                        ~in_missing_mask)

        loss_masked = overall_loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        return loss_masked.sum() / loss_mask.sum()

class FullyConnected2Layer(nn.Module):
    """
    Helper module that defines a simple 2-layer fully connected network with
    optional dropout and batch normalization.
    """
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
    """
    A model that takes a dynamics model's embedding output and predicts a single
    value for each timestep, either the reward for the current step or the
    discounted sum of rewards till the end of the trajectory.
    
    Input: embedding (N, L, E) representing the contextualized embeddings
        for each timestep
        
    Output:
        If predict_variance is True: returns a mean value prediction 
        (N, L, 1), an estimate of the log-variance of the prediction (N, L, 1),
        and a torch Distribution object that can be used to sample from the
        distribution defined by this mean and variance.
        If predict_variance is False: returns the mean next state prediction
        (N, L, 1)
        
    N = batch size
    L = sequence length
    E = embedding dimension
    """
    def __init__(self, embed_dim, predict_rewards=False, hidden_dim=16, dropout=0.1, batch_norm=False, predict_variance=False, variance_regularizer=10.0, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = FullyConnected2Layer(embed_dim, hidden_dim, 2 if predict_variance else 1, dropout=dropout, batch_norm=batch_norm)
        self.predict_rewards = predict_rewards
        self.predict_variance = predict_variance
        self.variance_regularizer = variance_regularizer
        self.loss_fn = nn.MSELoss(reduction='none')
        self.device = device
        
    def forward(self, embedding):
        out = self.net(embedding)
        if self.predict_variance:
            mu = embedding[:,:,0].unsqueeze(2)
            logvar = embedding[:,:,1].unsqueeze(2)
            return mu, logvar, torch.distributions.Normal(mu, F.softplus(logvar)) # ** 0.5)
        return out
    
    def compute_loss(self, in_batch, model_outputs):
        """
        Compute the MSE loss of the return or reward compared to the model output.
        
        in_batch: tuple (state, demog, action, missing, rewards, values, seq_len)
        model_outputs: (N, L, 1)
        """
        obs, _, _, _, rewards, values, seq_lens = in_batch
        L = obs.shape[1]
        if self.predict_variance:
            _, logvar, distro = model_outputs            
            neg_log_likelihood = -distro.log_prob(rewards if self.predict_rewards else values)
            loss = (neg_log_likelihood + self.variance_regularizer * torch.log(F.softplus(logvar))).squeeze(2) # ** 0.5)
        else:
            loss = self.loss_fn(model_outputs, rewards if self.predict_rewards else values).sum(2)
        
        loss_mask = torch.arange(L).to(self.device)[None, :] < seq_lens[:, None]

        loss_masked = loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        return loss_masked.sum() / loss_mask.sum()
    
class TerminationPredictionModel(nn.Module):
    """
    A model that takes a dynamics model's embedding output and predicts whether
    the trajectory will end at each timestep.
    
    Input: embedding (N, L, E) representing the contextualized embeddings
        for each timestep
        
    Output: termination logit probabilities (N, L, 1)
        
    N = batch size
    L = sequence length
    E = embedding dimension
    """
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
        loss_weights = (labels * (0.5 * seq_lens[:, None]) + (1 - labels) * (0.5 * seq_lens[:, None] / (torch.clamp(seq_lens[:, None] - 1, min=1)))).to(self.device)
        loss *= loss_weights
        
        loss_mask = torch.arange(L).to(self.device)[None, :] < seq_lens[:, None]

        loss_masked = loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        
        term_correct = ((torch.round(torch.sigmoid(model_outputs.squeeze(2))) == labels) & loss_mask).sum().item()
        term_total = loss_mask.sum().item()

        return loss_masked.sum() / loss_mask.sum(), term_correct, term_total
    
class ActionPredictionModel(nn.Module):
    """
    A model that predicts the action that gave rise to the transition between
    pairs of states. This can be used as an auxiliary task to help the transformer
    learn a more robust embedding space.
    
    Input: embedding (N', L, 2 * E) representing the contextualized embeddings
        for each timestep
        
    Output: mean action prediction (N, L, A)
        
    N' = batch size (use create_batch to create a batch of the appropriate size)
    L = sequence length
    E = embedding dimension
    A = action dimension
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
            if L == 0: continue
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
            if L == 0: continue
            final_flat.append(final_embeddings[i,:L - 1,:])
            initial_flat.append(initial_embeddings[i,1:L,:])
        
        final_flat = torch.cat(final_flat, 0)
        initial_flat = torch.cat(initial_flat, 0)
        left_side = np.arange(final_flat.shape[0])
        right_side = np.arange(final_flat.shape[0])

        while (left_side == right_side).mean() > 1 - self.positive_label_fraction:
            # Randomly flip indexes on the left and right, but use a normal distribution
            # so that more of the flips will be between states that are part of the
            # same trajectory. This should encourage the model to learn the difference
            # between different states for the same patient
            random_pair = [np.random.choice(len(left_side))] # np.random.choice(len(left_side), size=2, replace=False)
            offset = np.random.normal()
            random_pair.append(int(np.clip(random_pair[0] - max(1, abs(np.round(offset))) * np.sign(offset), 0, len(left_side) - 1)))
            left_side[random_pair] = left_side[np.flip(random_pair)]
            random_pair = [np.random.choice(len(right_side))] # np.random.choice(len(right_side), size=2, replace=False)
            offset = np.random.normal()
            random_pair.append(int(np.clip(random_pair[0] - max(1, abs(np.round(offset))) * np.sign(offset), 0, len(right_side) - 1)))
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
                 dynamics_architecture='transformer',
                 positional_encoding=True,
                 dropout=0.0,
                 num_decoder_layers=3,
                 value_dropout=0.1,
                 value_input_size=512,
                 action_training_fraction=0.2,
                 batch_size=32,
                 max_seq_len=160,
                 boolean_mask_as_input=False,
                 reward_batch_norm=True,
                 predict_variance=False,
                 variance_regularizer=0.0,
                 fc_hidden_dim=16,
                 mask_prob=0.5,
                 input_noise_scale=None,
                 num_unrolling_steps=1,
                 device='cpu',
                 replacement_values=None,
                 loss_weights=None):
        """
        dynamics_architecture: Can be "transformer" (TransformerDynamicsModel),
            "rnn" (RNNDynamicsModel), or "linear" (LinearDynamicsModel)
        loss_weights: Can be a tensor of 8 values corresponding to the following
            losses (where MSE is used if predict_variance is False, and log
            likelihood is used if true):
            1. Current state prediction loss (MSE or log likelihood)
            2. Next state prediction loss (MSE or log likelihood)
            3. Timestep reward loss (MSE or log likelihood)
            4. Discounted sum of rewards loss (MSE or log likelihood)
            5. Consistency loss (MSE between initial and final embeddings)
            6. Action prediction loss (MSE)
            7. Is subsequent action loss (BCE)
            8. Termination prediction loss (BCE)
            If a weight is zero, that loss will not be computed.
        """
        
        super().__init__()
        self.dynamics_architecture = dynamics_architecture
        if dynamics_architecture == "transformer":
            self.model = TransformerDynamicsModel(
                obs_size * 2 if boolean_mask_as_input else obs_size, 
                dem_size,
                2, 
                embed_size, 
                nhead, 
                nlayers, 
                dropout, 
                device=device,
                positional_encoding=positional_encoding).to(device)
        elif dynamics_architecture == "rnn":
            self.model = RNNDynamicsModel(
                obs_size * 2 if boolean_mask_as_input else obs_size, 
                dem_size,
                2, 
                embed_size,
                num_lstm_layers=nlayers,
                dropout=dropout,
                device=device
            ).to(device)
        elif dynamics_architecture == "linear":
            self.model = LinearDynamicsModel(
                obs_size * 2 if boolean_mask_as_input else obs_size, 
                dem_size,
                2, 
                embed_size,
                num_layers=nlayers,
                dropout=dropout,
                device=device
            ).to(device)
        self.boolean_mask_as_input = boolean_mask_as_input

        self.current_state_model = StatePredictionModel(
            obs_size, 
            embed_size, 
            predict_delta=False, 
            predict_variance=predict_variance,
            variance_regularizer=variance_regularizer,
            num_steps=0, 
            num_layers=num_decoder_layers,
            device=device).to(device)
        self.next_state_model = StatePredictionModel(
            obs_size, 
            embed_size, 
            predict_delta=False, 
            predict_variance=predict_variance,
            variance_regularizer=variance_regularizer,
            num_steps=1, 
            num_layers=num_decoder_layers,
            device=device).to(device)
        self.value_input_size = value_input_size
        if self.value_input_size < embed_size:
            self.value_input = nn.Linear(embed_size, self.value_input_size).to(device)
        else:
            self.value_input = None
        self.reward_model = ValuePredictionModel(
            self.value_input_size, 
            predict_rewards=True, 
            device=device, 
            batch_norm=reward_batch_norm,
            hidden_dim=fc_hidden_dim,
            predict_variance=predict_variance,
            variance_regularizer=variance_regularizer,
            dropout=value_dropout).to(device)
        self.return_model = ValuePredictionModel(
            self.value_input_size, 
            predict_rewards=False, 
            device=device, 
            batch_norm=reward_batch_norm,
            hidden_dim=fc_hidden_dim,
            predict_variance=predict_variance,
            variance_regularizer=variance_regularizer,
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
        self.input_noise_scale = input_noise_scale
        
        lw = loss_weights if loss_weights is not None else torch.FloatTensor([1, 1, 0.2, 0.05, 1, 1, 1, 1])
        self.loss_weights = (lw / lw.sum()).to(device)
        
        self.consistency_offset = 1
        self.consistency_subloss = nn.MSELoss(reduction='none')
        self.num_unrolling_steps = num_unrolling_steps

    def _consistency_loss(self, final_embed, initial_embed):
        return torch.cat((self.consistency_subloss(final_embed[:,:-self.consistency_offset],
                                                   initial_embed[:,self.consistency_offset:]), 
                         torch.zeros(final_embed.shape[0],
                                     self.consistency_offset, 
                                     final_embed.shape[2]).to(self.device)), 1)
        
    def compute_loss(self, batch, corrupt_inputs=True, return_elements=False, return_accuracies=False):
        """
        Runs the dynamics model and returns the weighted total loss over all
        tasks.
        
        Args:
            batch: A tuple from a DataLoader wrapped around a DynamicsDataset, 
                including (observations, demographics, actions, missingness 
                flags, rewards, discounted rewards, sequence lengths).
            corrupt_inputs: If True, add Gaussian noise to the inputs scaled
                by the input_noise_scale property.
            return_elements: If True, return the itemized, weighted length-8 
                vector loss instead of the sum.
            return_accuracies: If True, return a tuple (loss, subs_accuracy,
                term_accuracy) where subs_accuracy is a tuple (correct, total)
                indicating the accuracy of the subsequent state prediction task,
                and term_accuracy is a similar tuple indicating the accuracy of
                the termination prediction task.
        """
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

        masked_obs = obs
        should_mask = torch.zeros_like(obs)
        if corrupt_inputs:
            if self.input_noise_scale is not None:
                masked_obs = masked_obs + (torch.randn(masked_obs.shape) * self.input_noise_scale).to(self.device)
            if self.mask_prob > 0.0:
                assert self.replacement_values is not None
                should_mask = torch.logical_or(torch.rand(*obs.shape).to(self.device) < self.mask_prob, missing)
                masked_obs = torch.where(should_mask, torch.from_numpy(self.replacement_values).float().to(self.device), obs)
                should_mask = should_mask.float()

        if self.num_unrolling_steps > 1:
            # Run transformer multiple times and use the previous outputs as the next inputs.
            assert not self.boolean_mask_as_input
            
            itemized_loss = torch.zeros_like(self.loss_weights).to(self.device)
            initial_embed = None
            for step in range(self.num_unrolling_steps):
                if initial_embed is None:
                    _, initial_embed, final_embed = self.model(masked_obs, dem, ac, src_mask)
                else:
                    final_embed = self.model.unroll_from_embedding(initial_embed, ac[:,step:,:], src_mask)
                unroll_batch = (obs[:,step:,:], 
                                dem[:,step:,:], 
                                ac[:,step:,:], 
                                missing[:,step:,:], 
                                rewards[:,step:], 
                                discounted_rewards[:,step:], 
                                in_lens - step)
                
                step_loss, subs_acc, term_acc = self._compute_losses(unroll_batch, initial_embed, final_embed)
                itemized_loss += step_loss / len(self.num_unrolling_steps)
                
                # Shorten the input by one off the end, so that the model doesn't
                # try to evaluate on steps that it doesn't have knowledge about
                initial_embed = final_embed[:,:-1,:]
        else:
            # Just run transformer once
            embed_in = torch.cat((masked_obs, should_mask), 2) if self.boolean_mask_as_input else masked_obs
            _, initial_embed, final_embed = self.model(embed_in, dem, ac, src_mask)
            
            itemized_loss, subs_acc, term_acc = self._compute_losses(batch, initial_embed, final_embed)
        
        result = itemized_loss if return_elements else itemized_loss.sum()
        if return_accuracies:
            result = (result, subs_acc, term_acc)
        return result
    
    def _compute_losses(self, batch, initial_embed, final_embed):
        # 1. Masked prediction model
        if abs(self.loss_weights[0].item()) > 0.0001:
            pred_curr = self.current_state_model(initial_embed)
            curr_state_loss = self.current_state_model.compute_loss(batch, pred_curr)
        else:
            curr_state_loss = torch.tensor(0.0).to(self.device)
        
        # 2. Next-state prediction model
        if abs(self.loss_weights[1].item()) > 0.0001:
            pred_next = self.next_state_model(final_embed)
            next_state_loss = self.next_state_model.compute_loss(batch, pred_next)
        else:
            next_state_loss = torch.tensor(0.0).to(self.device)
        
        val_final_input = final_embed if self.value_input is None else F.leaky_relu(self.value_input(final_embed))
        val_initial_input = initial_embed if self.value_input is None else F.leaky_relu(self.value_input(initial_embed))
        
        # 3. Reward model
        if abs(self.loss_weights[2].item()) > 0.0001:
            pred_reward = self.reward_model(val_final_input)
            reward_loss = self.reward_model.compute_loss(batch, pred_reward)
        else:
            reward_loss = torch.tensor(0.0).to(self.device)

        # 4. Return model
        if abs(self.loss_weights[3].item()) > 0.0001:
            pred_return = self.return_model(val_final_input)
            return_loss = self.return_model.compute_loss(batch, pred_return)
        else:
            return_loss = torch.tensor(0.0).to(self.device)

        # 5. Consistency loss
        if abs(self.loss_weights[4].item()) > 0.0001:
            c_loss = self._consistency_loss(final_embed, initial_embed)
            loss_mask = torch.arange(c_loss.shape[1])[None, :, None].to(self.device) < batch[-1][:, None, None]
            c_loss_masked = c_loss.where(loss_mask, torch.tensor(0.0).to(self.device))
            c_loss = c_loss_masked.sum() / loss_mask.sum()
        else:
            c_loss = torch.tensor(0.0).to(self.device)
        
        # 6. Action prediction loss
        if abs(self.loss_weights[5].item()) > 0.0001:
            action_batch = self.action_model.create_batch(initial_embed, batch[-1], batch[2])
            pred_action = self.action_model(action_batch[0])
            action_loss = self.action_model.compute_loss(action_batch, pred_action)
        else:
            action_loss = torch.tensor(0.0).to(self.device)
        
        # 7. Is subsequent action loss
        if abs(self.loss_weights[6].item()) > 0.0001:
            subsequent_batch = self.subsequent_model.create_batch(val_final_input, val_initial_input, batch[-1])
            pred_subs = self.subsequent_model(subsequent_batch[0], subsequent_batch[1])
            subs_loss = self.subsequent_model.compute_loss(subsequent_batch, pred_subs)
            subs_correct = (torch.round(torch.sigmoid(pred_subs)) == subsequent_batch[2]).sum().item()
            subs_total = subsequent_batch[2].shape[0]
        else:
            subs_loss = torch.tensor(0.0).to(self.device)
            subs_correct = 0
            subs_total = 1
        
        # 8. Termination model
        if abs(self.loss_weights[7].item()) > 0.0001:
            pred_term = self.termination_model(val_final_input)
            term_loss, corr, tot = self.termination_model.compute_loss(batch, pred_term)
            term_correct = corr
            term_total = tot
        else:
            term_correct = 0
            term_total = 1
            
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
        return itemized_loss, (subs_correct, subs_total), (term_correct, term_total)
        
    def unroll_forward(self, batch=None, last_embedding=None, actions=None, eval=True):
        """
        Generates the next step and termination probability for one timestep
        in the future. Either batch (a tuple of obs, dem, ac, missing, rewards, discounted_rewards, in_lens)
        or last_embedding (a tensor of shape N x L x E where E is the embedding
        dimension of the transformer) AND actions (tensor of shape N x L x A) 
        is required.
        
        Returns a tuple (next state, termination, embedding) where next state
        is N x L x O, termination is N x L, and embedding is N x L x E.
        """
        assert batch is not None or (last_embedding is not None and actions is not None)
        
        if eval:
            self.eval()
            torch.set_grad_enabled(False)
            
        src_mask = generate_square_subsequent_mask(self.max_seq_len).to(self.device)
        if batch is not None:
            obs, dem, ac, _, _, _, _ = batch
            obs = obs.to(self.device)
            dem = dem.to(self.device)
            ac = ac.to(self.device)
            _, _, final_embed = self.model(obs, dem, ac, src_mask)            
        else:
            final_embed = self.model.unroll_from_embedding(last_embedding, actions, src_mask)
            
        _, _, pred_next = self.next_state_model(final_embed)
        val_final_input = final_embed if self.value_input is None else F.leaky_relu(self.value_input(final_embed))            
        pred_term = torch.sigmoid(self.termination_model(val_final_input)).squeeze(-1)
            
        if eval:
            torch.set_grad_enabled(True)
            
        return pred_next, pred_term, final_embed
        
    def train(self):
        self.overall_model.train()
        
    def eval(self):
        self.overall_model.eval()
        
    def state_dict(self):
        return {
            "dynamics": self.model.state_dict(),
            "current_state": self.current_state_model.state_dict(),
            "next_state": self.next_state_model.state_dict(),
            "value_input": self.value_input.state_dict(),
            "reward": self.reward_model.state_dict(),
            "return": self.return_model.state_dict(),
            "action": self.action_model.state_dict(),
            "subsequent": self.subsequent_model.state_dict(),
            "termination": self.termination_model.state_dict()
        }