import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from .transformer_model import generate_square_subsequent_mask
from ai_clinician.preprocessing.columns import *

class SepsisDynamicsEnvironment:
    def __init__(self, model, normer, state_decoder=None, device='cpu'):
        super().__init__()
        self.model = model
        self.normer = normer
        self.device = device
        self.state_decoder = state_decoder
        self.current_state = None # normalized representation
        self.current_demog = None # demog representation
        self.obs_history = []
        self.action_history = []
        self.traj_lengths = []
        self.embed_history = None
        
    def set_state(self, initial_obs, initial_demog, initial_action=None):
        """
        Sets the current state of the environment.
        
        initial_states: dataframe (N, O) where N is the batch size, O is
            the observation dimension. These should be in un-
            normalized values.
        initial_demog: dataframe (N, D) where N is the batch size, D is
            the demographics dimension. These should be in un-
            normalized values.
        """
        if isinstance(initial_obs, list):
            self.obs_history = []
            for obs in initial_obs:
                curr_state, self.current_demog = self.normer.transform_state(obs, initial_demog)
                self.obs_history.append(curr_state)
            self.traj_lengths = np.ones(self.obs_history[0].shape[0]) * len(self.obs_history)
        else:
            self.current_state, self.current_demog = self.normer.transform_state(initial_obs, initial_demog)
            self.obs_history = [self.current_state]
            self.traj_lengths = np.ones(initial_obs.shape[0])
        if initial_action is not None:
            self.action_history = [torch.from_numpy(self.normer.transform_action(actions))
                                   for actions in initial_action]
        else:
            self.action_history = []
        self.embed_history = None
        
    def step(self, actions):
        """
        Updates the current state of the environment by taking
        the given actions.
        
        actions: (N, A) where N is the batch size, A is the
            action dimension. These should be un-normalized
            fluid and vasopressor dosages.
            
        Returns:
            next state: dataframe (N, O)
            rewards: N
            termination: N. If a trajectory is terminated its
                subsequent states should not be taken into account.
        """
        norm_actions = torch.from_numpy(self.normer.transform_action(actions))
        self.action_history.append(norm_actions)
        self.model.eval()
        with torch.no_grad():
            ac = torch.from_numpy(np.stack(self.action_history, axis=1)).float().to(self.device)
            in_lens = torch.from_numpy(self.traj_lengths).to(self.device)        
            src_mask = generate_square_subsequent_mask(ac.shape[1]).to(self.device)
            
            if self.embed_history is None:
                obs = torch.from_numpy(np.stack(self.obs_history, axis=1)).float().to(self.device)
                dem = torch.from_numpy(np.tile(self.current_demog.reshape(self.current_demog.shape[0], 1, self.current_demog.shape[1]), 
                                               (1, len(self.obs_history), 1))).float().to(self.device)

                # Run transformer
                _, initial_embed, final_embed = self.model.model(obs, dem, ac, src_mask)
            else:
                final_embed = self.model.model.unroll_from_embedding(self.embed_history, ac[:,:-1,:], src_mask)

            # Predict next state
            pred_next, pred_next_std, pred_distro = self.model.next_state_model(final_embed)
            pred_next_mean = pred_next[:,-1,:].cpu().numpy()
            pred_next_lower = (pred_next - 1.96 * torch.clamp(F.softplus(pred_next_std), 0, 10))[:,-1,:].cpu().numpy()
            pred_next_upper = (pred_next + 1.96 * torch.clamp(F.softplus(pred_next_std), 0, 10))[:,-1,:].cpu().numpy()
            pred_sampled = np.clip(pred_distro.sample()[:,-1,:].cpu().numpy(),
                                   pred_next_lower,
                                   pred_next_upper)
            if self.state_decoder is not None:
                pred_next_mean = self.state_decoder(pred_next_mean)
                pred_next_lower = self.state_decoder(pred_next_lower)
                pred_next_upper = self.state_decoder(pred_next_upper)
                pred_sampled = self.state_decoder(pred_sampled)
            pred_next_mean = self.normer.inverse_transform_obs(pred_next_mean)
            pred_next_lower = self.normer.inverse_transform_obs(pred_next_lower)
            pred_next_upper = self.normer.inverse_transform_obs(pred_next_upper)
            pred_sampled = self.normer.inverse_transform_obs(pred_sampled)

            val_final_input = final_embed if self.model.value_input is None else F.leaky_relu(self.model.value_input(final_embed))

            # Reward and termination model
            pred_reward = self.model.reward_model(val_final_input)[0][:,-1].cpu().numpy()
            pred_term = torch.sigmoid(self.model.termination_model(val_final_input))[:,-1].cpu().numpy().flatten()
            terminated_flags = np.logical_or(self.traj_lengths < len(self.action_history),
                                             np.random.uniform(size=pred_term.shape) < pred_term)
            
            if self.embed_history is not None:
                self.embed_history = torch.cat((self.embed_history[:,0:1,:], final_embed), 1)
            else:
                self.embed_history = final_embed
            
        # Update states and sequence lengths
        self.obs_history.append(pred_next_mean)
        self.traj_lengths[~terminated_flags] += 1
        return pred_next_mean, pred_next_lower, pred_next_upper, pred_sampled, pred_reward, terminated_flags
    
    
def evaluate_dynamics_model_real_trajectories(eval_dataset, normer, env, num_initial_steps=24, num_to_simulate=12, batch_size=32, sample=None, sample_outputs=False):
    """
    Runs the dynamics model on a dataset of real trajectories, using the real
    set of actions performed by the clinicians. This can be used to evaluate
    the quality of the dynamics model on trajectories that are known.
    
    Args:
        eval_dataset: A DynamicsDataset instance containing trajectories to
            run on.
        normer: A DynamicsDataNormalizer that should be used to convert the
            observations back and forth to dynamics model space.
        env: A SepsisDynamicsEnvironment instance that contains a dynamics
            model to run.
        num_initial_steps: The number of timesteps to provide as context to
            the dynamics model.
        num_to_simulate: The number of timesteps to generate forward after the
            context timesteps. Only trajectories that have at least
            num_initial_steps + num_to_simulate timesteps available will be
            evaluated.
        batch_size: Batch size for input to the model.
        sample: If not None, a number indicating how many trajectories to
            sample from the dataset to evaluate. Otherwise, evaluates all
            trajectories with sufficient length.
        sample_outputs: If True, sample from the next state distribution,
            otherwise use the next state mean.
            
    Returns:
        A dataframe of new trajectory observations, containing the icustayid
        column which corresponds to the stay IDs from the eval dataset; an array
        of ICU stay IDs from the dataset that were used for evaluation; and an
        array of start indexes indicating which timestep was used as the first
        context timestep for each trajectory.
    """
    # First determine which trajectory IDs from the eval dataset to use
    valid_trajectory_indexes = pd.Series(np.arange(len(eval_dataset.stay_ids))).groupby(pd.Series(eval_dataset.stay_ids)).transform(lambda g: len(g) >= num_initial_steps + num_to_simulate)
    valid_traj_ids = np.unique(eval_dataset.stay_ids[valid_trajectory_indexes])
    if sample:
        valid_traj_ids = np.random.choice(valid_traj_ids, size=sample, replace=False)
    print(len(valid_traj_ids), "out of", len(np.unique(eval_dataset.stay_ids)), "trajectories with sufficient length")

    max_traj_length = num_initial_steps + num_to_simulate

    trajs = []
    start_indexes = []

    for traj_idx in tqdm.tqdm(range(0, len(valid_traj_ids), batch_size)):
        sample_traj_indexes = valid_traj_ids[traj_idx:traj_idx + batch_size]

        # Create initial context matrix for each trajectory        
        start_obs = []
        start_dem = []
        start_action = []
        future_actions = []
        sample_start_idxs = []
        for sample_traj_index in sample_traj_indexes:
            mask = eval_dataset.stay_ids == sample_traj_index
            start_idx = np.random.choice(max(1, mask.sum() - num_to_simulate - num_initial_steps))
            sample_start_idxs.append(start_idx)
            mask = mask
            obs = eval_dataset.observations[mask][start_idx:]
            dem = eval_dataset.demographics[mask][start_idx:]
            ac = eval_dataset.actions[mask][start_idx:]
            start_obs.append(obs[:num_initial_steps])
            start_action.append(normer.inverse_transform_action(ac[:num_initial_steps]))
            future_actions.append(normer.inverse_transform_action(ac[num_initial_steps:num_initial_steps + num_to_simulate]))
            start_dem.append(dem[:num_initial_steps])
        start_action = np.stack(start_action, axis=0)
        action_history = [start_action[:,i,:] for i in range(start_action.shape[1])]
        future_actions = np.stack(future_actions, axis=0)
        future_actions = [future_actions[:,i,:] for i in range(future_actions.shape[1])]
        start_obs = np.stack(start_obs, axis=0)
        state_history = [normer.inverse_transform_obs(start_obs[:,i,:]) for i in range(start_obs.shape[1])]
        start_dem = np.stack(start_dem, axis=0)
        start_dem = normer.inverse_transform_dem(start_dem[:,0,:])
        start_indexes.append(np.array(sample_start_idxs))
        
        # Add necessary fields like icustayid, timestep time
        for i, obs in enumerate(state_history):
            obs[C_BLOC] = i + 1
            obs[C_TIMESTEP] = 0
            obs["terminated"] = 0
            obs[C_ICUSTAYID] = sample_traj_indexes
        
        # Pass the context matrices to environment
        env.set_state(state_history, start_dem, initial_action=action_history[:-1])

        # Keep track of whether each trajectory has terminated or not
        terminated = np.zeros(len(sample_traj_indexes))
        
        # Iterate over timesteps and generate the next states
        while terminated.sum() < len(terminated) and len(state_history) < max_traj_length:
            last_terminated_flag = terminated
            mean_next_state, _, _, next_state, rewards, terminated = env.step(action_history[-1])
            
            # Add necessary columns
            next_state = next_state if sample_outputs else mean_next_state
            next_state[C_ICUSTAYID] = sample_traj_indexes
            next_state[C_BLOC] = len(state_history) + 1
            next_state[C_TIMESTEP] = (len(state_history) - 1) * 3600
            next_state["terminated"] = last_terminated_flag
            action_history.append(future_actions.pop(0))
            state_history.append(next_state)
            last_terminated_flag = terminated

        obs_df = pd.concat(state_history).sort_values([C_ICUSTAYID, C_BLOC])
        obs_df = obs_df[obs_df["terminated"] == 0]
        trajs.append(obs_df.drop(columns=["terminated"]))
        traj_idx += batch_size
        
    all_sim_trajs = pd.concat(trajs)
    start_indexes = np.concatenate(start_indexes)
    
    return all_sim_trajs, valid_traj_ids, start_indexes

def visualize_trajectory(traj_id, 
                         true_dataset, 
                         normer, 
                         sim_traj_sets, 
                         sim_actions=None, 
                         start_index=0, 
                         features_of_interest=[C_HR, C_TEMP_C, C_MEANBP, C_ARTERIAL_LACTATE, C_SOFA]):
    """
    Creates a plot of the given patient trajectory, including both the true
    patient state and one or more simulated patient states.
    
    Args:
        traj_id: The stay ID for the trajectory to visualize
        true_dataset: A DynamicsDataset to get the true patient values from
        normer: A DynamicsDataNormalizer object to remove normalization from the
            dataset
        sim_traj_sets: A list of dataframes that contain simulated patient states
            to visualize. Each of these dataframes will be searched for the
            matching patient and plotted on the same graph
        sim_actions: If provided, a list of numpy arrays specifying the actions
            taken for each simulated trajectory. The length of the list should
            be the same as sim_traj_sets, and each array in the list should have
            the same number of rows as the corresponding entry in sim_traj_sets,
            and two columns (representing IV fluids and vasopressors). If one of
            the elements in the list is None, the actions will be assumed to be
            the same as the true actions.
        start_index: The index in the true trajectory at which the simulation
            starts.
        features_of_interest: The list of features to plot.
    Example:
    
    ```
    all_sim_trajs, valid_traj_ids, start_indexes = evaluate_dynamics_model_real_trajectories(...)
    
    # Choose a random index from the trajectory IDs for which simulated
    # trajectories were generated
    traj_idx = np.random.choice(len(valid_traj_ids))
    visualize_trajectory(
        valid_traj_ids[traj_idx],
        val_dataset,
        normer,
        [all_sim_trajs],
        start_index=start_indexes[traj_idx]
    )
    ```
    """
    import matplotlib.pyplot as plt
    true_traj_mask = true_dataset.stay_ids == traj_id

    true_states = normer.inverse_transform_obs(true_dataset.observations[true_traj_mask]).reset_index(drop=True)
    true_actions = normer.inverse_transform_action(true_dataset.actions[true_traj_mask])
    
    sim_trajs = [sim_trajs[sim_trajs[C_ICUSTAYID] == traj_id] for sim_trajs in sim_traj_sets]
    if sim_actions is not None:
        sim_traj_actions = [sim_ac[sim_trajs[C_ICUSTAYID] == traj_id] if sim_ac is not None else true_actions[start_index:start_index + (sim_trajs[C_ICUSTAYID] == traj_id).sum()]
                                for sim_ac, sim_trajs in zip(sim_actions, sim_traj_sets)]

    plt.figure(figsize=(5, 10), dpi=70)
    plt.subplot(len(features_of_interest) + 2, 1, 1)
    plt.plot(true_actions[:,0])
    if sim_actions is not None:
        for action_set in sim_traj_actions:
            plt.plot(np.arange(start_index, start_index + len(action_set)), action_set[:,0])
    plt.title("Fluids")
    plt.subplot(len(features_of_interest) + 2, 1, 2)
    plt.plot(true_actions[:,1])
    if sim_actions is not None:
        for action_set in sim_traj_actions:
            plt.plot(np.arange(start_index, start_index + len(action_set)), action_set[:,1])
    plt.title("Vasopressors")

    for i, feature in enumerate(features_of_interest):
        plt.subplot(len(features_of_interest) + 2, 1, i + 3)
        plt.plot(true_states[feature].values)
        for sim_traj in sim_trajs:
            plt.plot(np.arange(start_index, start_index + len(sim_traj)), sim_traj[feature].values)
        plt.title(feature)
    plt.xlabel("Timestep (1 hr)")
    plt.tight_layout()
    plt.show()
    