import numpy as np
import pandas as pd
import tqdm
import argparse
import os
import shutil
import pickle
from .models.ai_clinician import *
from .models.common import *
from .columns import C_OUTCOME
from ai_clinician.preprocessing.utils import load_csv
from ai_clinician.preprocessing.columns import *
from sklearn.model_selection import train_test_split

tqdm.tqdm.pandas()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Evaluates an AI Clinician model on the MIMIC-IV test set.'))
    parser.add_argument('data', type=str,
                        help='Model data directory (should contain train and test directories)')
    parser.add_argument('model', type=str,
                        help='Path to pickle file containing the model')
    parser.add_argument('--out', dest='out_path', type=str, default=None,
                        help='Path to pickle file at which to write out results (optional)')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Decay for reward values (default 0.99)')
    parser.add_argument('--soften-factor', dest='soften_factor', type=float, default=0.01,
                        help='Amount by which to soften factors (random actions will be chosen this proportion of the time)')
    parser.add_argument('--num-iter-ql', dest='num_iter_ql', type=int, default=6,
                        help='Number of bootstrappings to use for TD learning (physician policy)')
    parser.add_argument('--num-iter-wis', dest='num_iter_wis', type=int, default=750,
                        help='Number of bootstrappings to use for WIS estimation (AI policy)')
    args = parser.parse_args()
    
    data_dir = args.data
    model = AIClinicianModel.load(args.model)
    assert model.metadata is not None, "Model missing metadata needed to generate actions"

    n_cluster_states = model.n_cluster_states
    n_actions = model.n_actions
    action_bins = model.metadata['actions']['action_bins']

    MIMICraw = load_csv(os.path.join(data_dir, "test", "MIMICraw.csv"))
    MIMICzs = load_csv(os.path.join(data_dir, "test", "MIMICzs.csv"))
    metadata = load_csv(os.path.join(data_dir, "test", "metadata.csv"))
    unique_icu_stays = metadata[C_ICUSTAYID].unique()
    
    # Bin vasopressor and fluid actions
    print("Create actions")    
    actions = transform_actions(
        MIMICraw[C_INPUT_STEP],
        MIMICraw[C_MAX_DOSE_VASO],
        action_bins
    )
    
    np.seterr(divide='ignore', invalid='ignore')
    
    blocs = metadata[C_BLOC].values
    stay_ids = metadata[C_ICUSTAYID].values
    outcomes = metadata[C_OUTCOME].values

    print("Evaluate on MIMIC test set")
    states = model.compute_states(MIMICzs.values)
    
    records = build_complete_record_sequences(
        metadata,
        states,
        actions,
        model.absorbing_states,
        model.rewards
    )
    
    test_bootql = evaluate_physician_policy_td(
        records,
        model.physician_policy,
        args.gamma,
        args.num_iter_ql,
        model.n_cluster_states
    )
    
    phys_probs = model.compute_physician_probabilities(states=states, actions=actions)
    model_probs = model.compute_probabilities(states=states, actions=actions)
    test_bootwis, _,  _ = evaluate_policy_wis(
        metadata,
        phys_probs,
        model_probs,
        model.rewards,
        args.gamma,
        args.num_iter_wis
    )

    model_stats = {}
    
    model_stats['test_bootql_0.95'] = np.quantile(test_bootql, 0.95)   #PHYSICIANS' 95# UB
    model_stats['test_bootql_mean'] = np.nanmean(test_bootql)
    model_stats['test_bootql_0.99'] = np.quantile(test_bootql, 0.99)
    model_stats['test_bootwis_mean'] = np.nanmean(test_bootwis)    
    model_stats['test_bootwis_0.01'] = np.quantile(test_bootwis, 0.01)  
    wis_95lb = np.quantile(test_bootwis, 0.05)  #AI 95# LB, we want this as high as possible
    model_stats['test_bootwis_0.05'] = wis_95lb
    model_stats['test_bootwis_0.95'] = np.quantile(test_bootwis, 0.95)
    print("Results:", model_stats)
    
    # TODO Implement other evaluation strategies
    
    if args.out_path is not None:
        with open(args.out_path, "wb") as file:
            pickle.dump(model_stats, file)
    print('Done.')