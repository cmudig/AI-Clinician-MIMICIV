"""
This module provides an adapter to the models developed by Killian et al.,
included in this repo under the 'sota' module.
"""
import numpy as np
from ai_clinician.modeling.columns import C_OUTCOME
from ai_clinician.modeling.models.base_ import BaseModel
from ai_clinician.modeling.models.discrete_bcq import FC_BC, DiscreteBCQ
from ai_clinician.modeling.models.sota import AE, AIS, CDE, DST, DDM, RNN, ODERNN
from ai_clinician.preprocessing.columns import *
from itertools import chain
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ai_clinician.preprocessing.derived_features import compute_oasis, compute_sapsii
from ai_clinician.modeling.models.sota.common import mask_from_lengths
from ai_clinician.modeling.models.sota.NeuralCDE.metamodel import NeuralCDE
from ai_clinician.modeling.models.sota.NeuralCDE import controldiffeq

SOTA_DEM_COLS = [C_GENDER, C_MECHVENT, C_RE_ADMISSION, C_AGE, C_WEIGHT]
SOTA_OBS_COLS = [
    C_GCS, C_HR, C_SYSBP, C_MEANBP, C_DIABP, C_RR, C_TEMP_C, C_FIO2_1,
    C_POTASSIUM, C_SODIUM, C_CHLORIDE, C_GLUCOSE, C_MAGNESIUM, C_CALCIUM,
    C_HB, C_WBC_COUNT, C_PLATELETS_COUNT, C_PTT, C_PT, C_ARTERIAL_PH,
    C_PAO2, C_PACO2, C_ARTERIAL_BE, C_HCO3, C_ARTERIAL_LACTATE, C_PAO2_FIO2,
    C_SPO2, C_BUN, C_CREATININE, C_SGOT, C_SGPT, C_TOTAL_BILI, C_INR
]

class SOTAModel(BaseModel):
    def __init__(self,
                 architecture_name,
                 num_actions=25,
                 state_dim=33,
                 context_input=True,
                 context_dim=5,
                 hidden_size=64,
                 batch_size=128,
                 reward_val=1,
                 device='cpu',
                 hyperparameters={},
                 metadata=None):
        super(SOTAModel, self).__init__()
        self.architecture_name = architecture_name
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.context_input = context_input
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size
        self.metadata = metadata
        self.hyperparameters = hyperparameters
        self.reward_val = 1
        if self.context_input:
            self.input_dim = self.state_dim + self.context_dim + self.num_actions
        else:
            self.input_dim = self.state_dim + self.num_actions
        
        if self.architecture_name == 'AIS':          
            self.container = AIS.ModelContainer(device, 1, 1) # ais_gen_model, ais_pred_model) - these were configurable before but setting them to true
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.num_actions, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.num_actions)
            
        elif self.architecture_name == 'AE':
            self.container = AE.ModelContainer(device)
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.num_actions, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.num_actions)
              
        elif self.architecture_name == 'DST':
            self.container = DST.ModelContainer(device)            
            self.gen = self.container.make_encoder(self.input_dim, self.hidden_size,
                                                   gru_n_layers = self.hyperparameters.get('gru_n_layers', 2),
                                                   augment_chs = self.hyperparameters.get('augment_chs', 8))
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim,
                                                    self.hyperparameters.get('decoder_hidden_units', 64))    

        elif self.architecture_name == 'DDM':
            self.container = DDM.ModelContainer(device)   
            
            self.gen = self.container.make_encoder(self.state_dim, self.hidden_size, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.state_dim,self.hidden_size)
            self.dyn = self.container.make_dyn(self.num_actions,self.hidden_size)
            self.all_params = chain(self.gen.parameters(), self.pred.parameters(), self.dyn.parameters())
            
            self.inv_loss_coef = 10
            self.dec_loss_coef = 0.1
            self.max_grad_norm = 50
        
        elif self.architecture_name == 'RNN':
            self.container = RNN.ModelContainer(device)
            
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.num_actions, context_input=self.context_input, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.num_actions)
            
        elif self.architecture_name == 'CDE':
            self.container = CDE.ModelContainer(device)
            self.gen = self.container.make_encoder(self.input_dim + 1,
                                                   self.hidden_size, 
                                                   hidden_hidden_channels = self.hyperparameters.get('encoder_hidden_hidden_channels', 100),
                                                   num_hidden_layers = self.hyperparameters.get('encoder_num_hidden_layers', 4))
            self.pred = self.container.make_decoder(self.hidden_size,
                                                    self.state_dim,
                                                    self.hyperparameters.get('decoder_num_layers', 3),
                                                    self.hyperparameters.get('decoder_num_units', 100))
            
        elif self.architecture_name == 'ODERNN':
            self.container = ODERNN.ModelContainer(device)
            
            self.gen = self.container.make_encoder(self.input_dim, self.hidden_size, self.hyperparameters)
            self.pred = self.container.make_decoder(self.hidden_size,
                                                    self.state_dim, 
                                                    self.hyperparameters.get('decoder_n_layers', 3), 
                                                    self.hyperparameters.get('decoder_n_units', 100))
        else:
            raise NotImplementedError
        
        # For the policy learned on this state, the Killian et al. paper uses
        # a discrete batch-constrained Q-network.
        self.policy = DiscreteBCQ(
            num_actions,
            hidden_size,
            self.device,
            BCQ_threshold=0.3,
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={
                "lr": 1e-6
            },
            polyak_target_update=True,
            target_update_frequency=1,
            tau=0.01
        )
        
        # Additionally, keep a physician policy learned using behavior cloning.
        behav_input = self.state_dim + (self.context_dim if self.context_input else 0)
        self.physpol = FC_BC(behav_input, self.num_actions, 64).to(self.device)

    def format_trajectories(self, data, actions, metadata):
        """
        Generates data objects that can be passed as the `X` value for other
        prediction methods for this SOTAModel. The models developed by Killian
        et al. use a slightly different data input format than the others in
        this repo, in that they use a different column order, format individual 
        trajectories as tensors, and incorporate previous actions into the state
        input. This method can be called directly to pass into the X value of
        compute_cde_coefs, compute_states, compute_Q, compute_V, and
        compute_probabilities.
        
        The input to this method should be a MIMICzs-formatted dataframe, a
        vector of actions corresponding to the length of raw_data, and a metadata
        dataframe containing outcome data.
        """
        assert len(actions) == len(data) == len(metadata)
        
        # Adapted from rl_representations: scripts/split_sepsis_cohort.py
        data_trajectory = {}
        all_acuities = pd.DataFrame({
            C_SOFA: data[C_SOFA],
            'OASIS': compute_oasis(data),
            'SAPSII': compute_sapsii(data)
        })
        unique_stays = metadata[C_ICUSTAYID].unique()
        for i in unique_stays:
            # bar.update()
            traj_mask = metadata[C_ICUSTAYID] == i
            data_trajectory[i] = {}
            data_trajectory[i]['dem'] = torch.Tensor(data.loc[traj_mask, SOTA_DEM_COLS].values).to('cpu')
            data_trajectory[i]['obs'] = torch.Tensor(data.loc[traj_mask, SOTA_OBS_COLS].values).to('cpu')
            data_trajectory[i]['actions'] = torch.Tensor(actions[traj_mask].astype(np.int32)).to('cpu').long()
            data_trajectory[i]['rewards'] = torch.Tensor((1 - 2 * metadata.loc[traj_mask, C_OUTCOME].values).astype(np.int32)).to('cpu')
            data_trajectory[i]['acuity'] = torch.Tensor(all_acuities[traj_mask].values).to('cpu')

        horizon = metadata[C_BLOC].max() + 1
        observations = torch.zeros((len(unique_stays), horizon, len(SOTA_OBS_COLS)))
        demographics = torch.zeros((len(unique_stays), horizon, len(SOTA_DEM_COLS))) 
        actions = torch.zeros((len(unique_stays), horizon-1, self.num_actions))
        lengths = torch.zeros((len(unique_stays)), dtype=torch.int)
        times = torch.zeros((len(unique_stays), horizon))
        rewards = torch.zeros((len(unique_stays), horizon))
        acuities = torch.zeros((len(unique_stays), horizon-1, len(all_acuities.columns)))
        action_temp = torch.eye(25)
        for ii, traj in enumerate(unique_stays):
            obs = data_trajectory[traj]['obs']
            dem = data_trajectory[traj]['dem']
            action = data_trajectory[traj]['actions'].view(-1,1)
            reward = data_trajectory[traj]['rewards']
            acuity = data_trajectory[traj]['acuity']
            length = obs.shape[0]
            lengths[ii] = length
            temp = action_temp[action].squeeze(1)
            observations[ii] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
            demographics[ii] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
            actions[ii] = torch.cat((temp, torch.zeros((horizon-length-1, self.num_actions), dtype=torch.float)))
            acuities[ii] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))
            times[ii] = torch.Tensor(range(horizon))
            rewards[ii] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))

        return demographics, observations, actions, lengths, times, acuities, rewards, torch.arange(demographics.shape[0])
    
    def unroll_trajectories(self, X, rolled_array):
        """
        Converts a formatted array such as one returned by compute_states,
        compute_Q, compute_probabilities, etc. to a flattened one. All but the
        last dimension of the array will be flattened and subsetted so that the
        first dimension of the returned array is the same as the input to
        format_trajectories.
        """
        flattened = []
        for row, length in zip(rolled_array, X[3]):
            flattened.append(row[:length].reshape(-1, row.shape[1]))
        return np.concatenate(flattened)
        
    def compute_cde_coefs(self, X):
        """
        Prepares to run the CDE by computing coefficients for each data item.
        """
        dataset = TensorDataset(*X)
        loader = DataLoader(dataset, batch_size=X[0].shape[0], shuffle=False)
        for dem, ob, ac, l, _, scores, _, _ in loader: 
            max_length = int(l.max().item())
            ob = ob[:,:max_length,:].to(self.device)
            dem = dem[:,:max_length,:].to(self.device)
            ac = ac[:,:max_length,:].to(self.device)
            scores = scores[:,:max_length,:].to(self.device)
            l = l.to(self.device)
            break
        
        ac_shifted = torch.cat((torch.zeros(ac.shape[0], 1, ac.shape[-1]).to(self.device), ac[:, :-1, :]), dim = 1) # action at t-1
        if self.context_input:
            obs_data = torch.cat((ob, dem, ac_shifted), dim = -1)
        else:
            obs_data = torch.cat((ob, ac_shifted), dim = -1)
        obs_mask = mask_from_lengths(l, max_length, self.device).unsqueeze(-1).expand(*obs_data.shape)
        obs_data[~obs_mask] = torch.tensor(float('nan')).to(self.device)
        times = torch.arange(max_length, device=self.device).float()    
        
        augmented_X = torch.cat((times.unsqueeze(0).repeat(obs_data.size(0), 1).unsqueeze(-1)
                                , obs_data), dim = 2)
        coeffs = controldiffeq.natural_cubic_spline_coeffs(times, augmented_X)
        return coeffs
        
    def train(self, X_train, actions_train, metadata_train, X_val=None, actions_val=None, metadata_val=None):
        raise NotImplementedError("Training SOTA models must be done using the rl_representations repository (https://github.com/MLforHealth/rl_representations).")
    
    def compute_states(self, X, coefs=None):
        assert self.architecture_name != "CDE" or coefs is not None, "coefs is required if using CDE"
        dataset = TensorDataset(*X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []
        with torch.no_grad():
            for dem, ob, ac, l, t, scores, rewards, idx in loader:
                dem = dem.to(self.device)
                ob = ob.to(self.device)
                ac = ac.to(self.device)
                l = l.to(self.device)
                t = t.to(self.device)
                scores = scores.to(self.device)
                rewards = rewards.to(self.device)

                max_length = int(l.max().item())

                ob = ob[:,:max_length,:]
                dem = dem[:,:max_length,:]
                ac = ac[:,:max_length,:]
                scores = scores[:,:max_length,:]
                rewards = rewards[:,:max_length]

                self.gen.eval()
                self.pred.eval()

                if self.architecture_name in ['AE', 'AIS', 'RNN']:
                    if self.context_input:
                        inputs = torch.cat((ob, dem,
                                            torch.cat((torch.zeros((ob.shape[0],1,ac.shape[-1])).to(self.device),ac[:,:-1,:]),dim=1)),dim=-1)
                        representations = self.gen(inputs)
                    else:
                        representations = self.gen(torch.cat((ob, torch.cat((torch.zeros((ob.shape[0],1,ac.shape[-1])).to(self.device), ac[:,:-1,:]),dim=1)), dim=-1))

                elif self.architecture_name == 'DDM':
                    if self.context_input:
                        representations = self.gen(torch.cat((ob, dem), dim=-1))
                    else:
                        representations = self.gen(ob)
                
                elif self.architecture_name in ['DST', 'ODERNN']:
                    _, _, representations = self.container.loop(ob, dem, ac, scores, l, max_length,
                                                                self.context_input,
                                                                corr_coeff_param = 0,
                                                                device=self.device)
                    representations = representations.detach()
                    
                elif self.architecture_name == 'CDE':
                    _, _, representations = self.container.loop(ob, dem, ac, scores, l, max_length,
                                                                self.context_input,
                                                                corr_coeff_param = 0,
                                                                device=self.device,
                                                                coefs = coefs,
                                                                idx = idx) 
                    representations = representations.detach()
                    
                results.append(representations)
        return torch.cat(results).cpu().numpy()
        
    def compute_Q(self, X=None, states=None, actions=None, unroll=True):
        assert X is not None or states is not None, "At least one of states or X must not be None"
        if states is None:
            states = self.compute_states(X)
        
        emb_dataset = TensorDataset(torch.from_numpy(states))
        emb_loader = DataLoader(emb_dataset, batch_size=self.batch_size, shuffle=False)
        val_Q = []
        with torch.no_grad():
            for (val_states,) in emb_loader:
                val_Q.append(self.policy.Q(val_states.float().to(self.device))[0].detach().numpy())
        val_Q = np.concatenate(val_Q)
        val_Q = np.clip(val_Q, -self.reward_val, self.reward_val)
        if unroll:
            val_Q = self.unroll_trajectories(X, val_Q)
        if actions is not None:
            assert unroll, "Can't filter by actions without unrolling"
            return val_Q[np.arange(len(val_Q)), actions]
        return val_Q
        
    def compute_V(self, X, unroll=True):
        val_Q = self.compute_Q(X=X, unroll=unroll)
        val_V = val_Q.mean(axis=-1)
        return val_V
    
    def compute_probabilities(self, X=None, states=None, actions=None, unroll=True):
        assert X is not None or states is not None, "At least one of states or X must not be None"
        if states is None:
            states = self.compute_states(X)
        
        emb_dataset = TensorDataset(torch.from_numpy(states))
        emb_loader = DataLoader(emb_dataset, batch_size=self.batch_size, shuffle=False)
        val_probs = []
        with torch.no_grad():
            for (val_states,) in emb_loader:
                val_probs.append(F.softmax(self.policy.Q(val_states.float().to(self.device))[0], dim=1).detach().numpy())
        val_probs = np.concatenate(val_probs)
        if unroll:
            val_probs = self.unroll_trajectories(X, val_probs)
        if actions is not None:
            assert unroll, "Can't filter by actions without unrolling"
            return val_probs[np.arange(len(val_probs)), actions]
        return val_probs 
       
    def compute_physician_probabilities(self, X=None, states=None, actions=None, unroll=True):
        """
        Returns the probabilities for each state and action according to the
        physician policy, which is learned using the same state representation
        model as the learned policy.
        
        If actions is not None, it must match the value of unroll. For instance,
        if unroll is False, the actions matrix must be the same shape as the
        formatted trajectory data. If unroll is True, the actions matrix should
        be 1D.
        """
        assert X is not None, "SOTAModel.compute_physician_probabilities does not support using learned representations"
        dataset = TensorDataset(*X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        results = []
        # TODO Test
        with torch.no_grad():
            for demog, obs_state, _, _, _, _, _, _ in loader:

                obs_state = obs_state.to(self.device)
                demog = demog.to(self.device)

                cur_obs = obs_state[:,:-1,:]
                cur_demog = demog[:,:-1,:]

                if self.context_input:  # Gather the probability from the observed behavior policy
                    inputs = torch.cat((cur_obs.flatten(end_dim=1), cur_demog.flatten(end_dim=1)), dim=-1)
                else:
                    inputs = cur_obs.flatten(end_dim=1)
                print(inputs.shape)
                p_obs = F.softmax(self.physpol(inputs), dim=-1).reshape(cur_obs.shape[:2] + (-1,))
                print(p_obs.shape)
                results.append(p_obs.detach().numpy())
        results = np.concatenate(results, axis=0)
        if unroll:
            results = self.unroll_trajectories(X, results)
        if actions is not None:
            assert unroll, "Can't filter by actions without unrolling"
            return results[np.arange(results.shape[0]),actions]
        return results
        
    def _make_state_dict(self):
        save_dict = {
            'representation': {
                'gen': self.gen.state_dict(),
                'pred': self.pred.state_dict(),   
            },
            'policy': {
                'Q': self.policy.Q.state_dict(),
                'Q_target': self.policy.Q.state_dict()                
            },
            'physpol': self.physpol.state_dict()
        }
        if self.architecture_name == 'DDM':
            save_dict['representation']['dyn'] = self.dyn.state_dict()
        return save_dict
        
    def save(self, filepath, metadata=None):
        """
        Saves the model as a pickle to the given filepath.
        """
        model_data = {
            'model_type': 'SOTAModel',
            'architecture_name': self.architecture_name,
            'state': self._make_state_dict(),
            'params': {
                'state_dim': self.state_dim,
                'num_actions': self.num_actions,
                'context_input': self.context_input,
                'context_dim': self.context_dim,
                'hidden_size': self.hidden_size,
                'batch_size': self.batch_size,
                'reward_val': self.reward_val,
                'device': self.device,
                'hyperparameters': self.hyperparameters,
            }
        }
        if metadata is not None:
            model_data['metadata'] = metadata 
        torch.save(model_data, filepath)
    
    def load_state_model(self, checkpoint_path):
        """
        Loads a SOTA model from a checkpoint path (generated by the rl_representations
        source code).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gen.load_state_dict(checkpoint['gen_state_dict'])
        self.pred.load_state_dict(checkpoint['pred_state_dict'])
        if self.architecture_name == 'DDM':
            self.dyn.load_state_dict(checkpoint['dyn_state_dict'])
        
    def load_policy(self, checkpoint_path):
        """
        Loads a DiscreteBCQ model from a checkpoint path (generated by the
        rl_representations source code).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.Q.load_state_dict(checkpoint['policy_Q_function'])
        self.policy.Q_target.load_state_dict(checkpoint['policy_Q_target'])
        
    def load_physician_policy(self, checkpoint_path):
        """
        Loads a physician policy learned using Behavior Cloning from a
        checkpoint path.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.physpol.load_state_dict(checkpoint)
        
    @classmethod
    def load(cls, filepath, device='cpu'):
        """
        Loads a model from a pickle file at the given filepath.
        """
        model_data = torch.load(filepath, map_location=device)
        assert model_data['model_type'] == 'SOTAModel', 'Invalid model type for SOTAModel'
        model = cls(model_data['architecture_name'],
                    **model_data['params'])
        state_info = model_data['state']
        model.gen.load_state_dict(state_info['representation']['gen'])
        model.pred.load_state_dict(state_info['representation']['pred'])
        if model.architecture_name == 'DDM':
            model.dyn.load_state_dict(state_info['representation']['dyn'])
        model.policy.Q.load_state_dict(state_info['policy']['Q'])
        model.policy.Q_target.load_state_dict(state_info['policy']['Q_target'])
        model.physpol.load_state_dict(state_info['physpol'])

        model.metadata = model_data.get('metadata', None)
        return model
    
if __name__ == '__main__':
    # model = SOTAModel('RNN')
    # model.load_from_checkpoint('/Users/vsivaram/Documents/research/icu/rl_representations/data/results/rnn_s64_l1e-4_rand25_sepsis/rnn_checkpoints/checkpoint.pt')
    # model.save("/Users/vsivaram/Documents/research/icu/AI-Clinician-MIMICIV/data/sota_models/rnn.pt")
    model = SOTAModel.load("/Users/vsivaram/Documents/research/icu/AI-Clinician-MIMICIV/data/sota_models/rnn.pt")
    from ai_clinician.preprocessing.utils import load_csv
    from ai_clinician.modeling.models.common import fit_action_bins
    MIMICraw = load_csv("/Users/vsivaram/Documents/research/icu/AI-Clinician-MIMICIV/data/model_220222/test/MIMICraw.csv")
    MIMICzs = load_csv("/Users/vsivaram/Documents/research/icu/AI-Clinician-MIMICIV/data/model_220222/test/MIMICzs.csv")
    metadata = load_csv("/Users/vsivaram/Documents/research/icu/AI-Clinician-MIMICIV/data/model_220222/test/metadata.csv")
    all_actions, action_medians, action_bins = fit_action_bins(
        MIMICraw[C_INPUT_STEP],
        MIMICraw[C_MAX_DOSE_VASO],
        n_action_bins=5
    )
    print("Formatting")
    X = model.format_trajectories(MIMICzs, all_actions, metadata)
    print("Computing states")
    states = model.compute_states(X)
    np.save("/Users/vsivaram/Documents/research/icu/AI-Clinician-MIMICIV/data/sota_models/rnn_embeddings.npy", states)