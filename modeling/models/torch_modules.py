import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modeling.data_structures import SumTree

# https://github.com/AntonP999/Sparse_autoencoder

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_size=10):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_size)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_size=10):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, output_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return x
    
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_size=10, loss_fn=F.mse_loss, lr=1e-3, l2=0.):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_size = latent_size
        self.E = Encoder(input_dim, latent_size)
        self.D = Decoder(input_dim, latent_size)
        self.loss_fn = loss_fn
        self._rho_loss = None
        self._loss = None
        self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        
    def forward(self, x):
        # x = x.view(-1, 28*28)
        h = self.E(x)
        self.data_rho = h.mean(0) # calculates rho from encoder activations
        out = self.D(h)
        return out
    
    def decode(self, h):
        with torch.no_grad():
            return self.D(h)
    
    def rho_loss(self, rho, size_average=True):        
        dkl = - rho * torch.log(self.data_rho) - (1-rho)*torch.log(1-self.data_rho) # calculates KL divergence
        if size_average:
            self._rho_loss = dkl.mean()
        else:
            self._rho_loss = dkl.sum()
        return self._rho_loss
    
    def loss(self, x, target, **kwargs):
        # target = target.view(-1, 28*28)
        self._loss = self.loss_fn(x, target, **kwargs)
        return self._loss
    
def train_autoencoder(models, train_loader, log=None, beta=0.5, rho=0.05):
    train_size = 0
    running_loss = {k: 0.0 for k in models}
    for m in models.values(): m.train()
    for batch_idx, (data,) in enumerate(train_loader): # pbar:
        for k, model in models.items():
            model.optim.zero_grad()
            inputs = data.clone().detach()
            output = model(inputs)
            rho_loss = model.rho_loss(rho)
            loss = model.loss(output, data) + beta * rho_loss
            loss.backward()
            model.optim.step()
            running_loss[k] = running_loss[k] + loss.item()
            train_size += 1
            
        if batch_idx % 100 == 0:
            # losses = " ".join(["{}: {:.6f}".format(k, running_loss[k] / train_size) for k, m in models.items()])
            # pbar.set_description("Losses " + losses) 

            if log is not None:
                for k in models:
                    log[k].append(running_loss[k])
     
def test_autoencoder(models, loader, log=None, beta=0.5, rho=0.05):
    test_size = 0

    test_loss = {k: 0. for k in models}
    rho_loss = {k: 0. for k in models}
    for m in models.values(): m.eval()
    with torch.no_grad():
        for (data,) in loader:
            inputs = data.clone().detach()
            output = {k: m(inputs) for k, m in models.items()}
            for k, m in models.items():
                test_loss[k] += m.loss(output[k], data).item()
                rho_loss[k] += beta * m.rho_loss(rho).item()
            test_size += 1
    
    for k in models:
        test_loss[k] /= test_size
        rho_loss[k] /= test_size
        if log is not None:
            log[k].append((test_loss[k], rho_loss[k]))
    
    avg_lambda = lambda l: "loss: {:.4f}".format(l)
    rho_lambda = lambda p: "rho_loss: {:.4f}".format(p)
    line = lambda i, l, p: "{}: ".format(i) + avg_lambda(l) + "\t" + rho_lambda(p)

    lines = "\n".join([line(k, test_loss[k], rho_loss[k]) for k in models])
    report = lines        
    print(report)
    return test_loss, rho_loss

def embed_autoencoder(model, loader):
    embs = []
    model.eval()
    with torch.no_grad():
        for (data,) in loader:
            inputs = data.clone().detach()
            output = model.E(inputs).numpy()
            embs.append(output)
    return np.concatenate(embs)           

### Dueling DQN

class PrioritizedBuffer:

    def __init__(self, max_size, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.current_length = 0

    def push(self, state, action, reward, next_state, done):
        # TODO fold this max value method into SumTree implementation
        priority = 1.0 if self.current_length == 0 else max(self.sum_tree.tree[self.sum_tree.capacity:].max(), 1.0)
        self.current_length = self.current_length + 1
        #priority = td_error ** self.alpha
        experience = (state, action, reward, next_state, done)
        self.sum_tree.add(priority, experience)

    def sample(self, batch_size):
        batch_idx, batch, IS_weights = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(batch_size):
            p = 0.0
            a = segment * i
            b = segment * (i + 1)

            while p < 1e-6: 
                s = random.uniform(a, b)
                idx, p, data = self.sum_tree.get(s)

            batch_idx.append(idx)
            batch.append(data)
            prob = p / p_sum
            IS_weight = (self.sum_tree.total() * prob) ** (-self.beta)
            if np.isnan(IS_weight) or np.isinf(IS_weight):
                print(self.sum_tree.total(), prob, IS_weight, idx, p, data)
            IS_weights.append(IS_weight)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), batch_idx, IS_weights

    def update_priority(self, idx, td_error):
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length
    
    
UPDATE_NN_EVERY = 1        # how often to update the network
UPDATE_MEM_EVERY = 20          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000     # how often to update the hyperparameters

class DuelingDQN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(1).unsqueeze(1)) # added the 1 here?
        
        return qvals
