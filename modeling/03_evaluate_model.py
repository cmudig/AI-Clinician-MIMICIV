import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import shutil
import pickle
from modeling.models.ai_clinician import *
from modeling.models.common import *
from modeling.columns import C_OUTCOME
from preprocessing.utils import load_csv
from preprocessing.columns import *
from sklearn.model_selection import train_test_split

tqdm.pandas()

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
    with open(args.model, 'rb') as file:
        model_data = pickle.load(file)

    Q = model_data['Qon']
    optimal_actions = model_data['optimal_actions']
    transitionr = model_data['T']
    R = model_data['R']
    physpol = model_data['physician_policy']
    clusterer = model_data['clusterer']
    action_bins = model_data['action_bins']
    rewards = model_data['rewards']
    
    n_cluster_states = len(clusterer.cluster_centers_)
    n_states = n_cluster_states + 2 # discharge and death are absorbing states
    absorbing_states = [n_cluster_states + 1, n_cluster_states] # absorbing state numbers
    n_action_bins = len(action_bins[0])
    n_actions = n_action_bins * n_action_bins

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
    states = clusterer.predict(MIMICzs.values)
    
    # Create qldata3
    qldata3 = build_complete_record_sequences(
        blocs,
        stay_ids,
        states,
        actions,
        outcomes,
        absorbing_states,
        rewards
    )
    
    test_bootql, test_bootwis = evaluate_policy(
        qldata3,
        optimal_actions,
        physpol,
        n_cluster_states,
        soften_factor=args.soften_factor,
        gamma=args.gamma,
        num_iter_ql=args.num_iter_ql,
        num_iter_wis=args.num_iter_wis
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