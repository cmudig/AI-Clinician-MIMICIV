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
    
    def __init__(self, state_dim, embed_dim, variance_regularizer=1.0, timestep_weight_eps=0.1, predict_delta=True, num_steps=1, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder = nn.Linear(embed_dim, state_dim)
        self.variance_decoder = nn.Linear(embed_dim, state_dim)
        self.variance_regularizer = variance_regularizer
        self.timestep_weight_eps = timestep_weight_eps
        self.predict_delta = predict_delta
        self.num_steps = num_steps
        self.device = device
        
    def init_weights(self):
        self.decoder.bias.data.zero_()
        torch.nn.init.xavier_normal_(self.decoder.weight.data) # uniform_(-initrange, initrange)
        self.variance_decoder.bias.data.fill_(1.0)
        torch.nn.init.xavier_normal_(self.variance_decoder.weight.data) # uniform_(-initrange, initrange)
        
    def forward(self, embedding):
        output_mu = self.decoder(embedding)
        logvar = self.variance_decoder(embedding)
        return output_mu, logvar, torch.distributions.Normal(output_mu, torch.exp(logvar) ** 0.5)
    
    def compute_loss(self, in_batch, model_outputs):
        """
        Compute the NLL loss of the model's outputs compared to the next states
        found in the inputs.
        
        in_batch: tuple (state, demog, action, missing, rewards, values, seq_len)
        model_outputs: tuple containing mean, logvar, and 
            torch.distributions.Normal of shape (N, L, S) where S is state_dim
        """
        in_state, _, _, in_missing_mask, _, _, seq_lens = in_batch
        _, logvar, distro = model_outputs
        
        L = in_state.shape[1]
        assert L > self.num_steps
        next_state_vec = in_state[:,self.num_steps:,:].clone()
        if self.predict_delta:
            next_state_vec -= in_state[:,:in_state.shape[1] - self.num_steps,:]

        if self.num_steps > 0:
            next_state_vec = torch.cat((next_state_vec, torch.zeros(next_state_vec.shape[0], 1, next_state_vec.shape[2]).to(self.device)), 1)
            timestep_weights = self.timestep_weight_eps + next_state_vec ** 2  # Upweight timesteps with more change
        else:
            timestep_weights = torch.ones_like(next_state_vec).to(self.device)
        neg_log_likelihood = -distro.log_prob(next_state_vec)
        overall_loss = neg_log_likelihood + self.variance_regularizer * logvar
        overall_loss *= timestep_weights
        
        loss_mask = torch.logical_and(torch.arange(L)[None, :, None].to(self.device) < seq_lens[:, None, None] - self.num_steps,
                                        ~in_missing_mask)

        loss_masked = overall_loss.where(loss_mask, torch.tensor(0.0).to(self.device))
        return loss_masked.sum() / loss_mask.sum()

class FullyConnected2Layer(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, out_dim)

    def forward(self, state):
        out = F.leaky_relu(self.l1(state))
        return self.l2(out)
    
class ValuePredictionModel(nn.Module):
    def __init__(self, embed_dim, predict_rewards=False, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = FullyConnected2Layer(embed_dim, embed_dim, 1)
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
    
class ActionPredictionModel(nn.Module):
    """
    A model that predicts the action that gave rise to the transition between
    pairs of states.
    """
    def __init__(self, embed_dim, action_dim, device='cpu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = FullyConnected2Layer(embed_dim * 2, embed_dim, action_dim)
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
        for i in range(embeddings.shape[0]):
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
        return self.loss_fn(model_outputs, actions)
    