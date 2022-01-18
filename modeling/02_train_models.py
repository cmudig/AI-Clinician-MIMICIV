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

    parser = argparse.ArgumentParser(description=('Builds multiple models using '
        'the AI Clinician RL technique.'))
    parser.add_argument('data', type=str,
                        help='Model data directory (should contain train and test directories)')
    parser.add_argument('--worker-label', dest='worker_label', type=str, default='',
                        help='Label to suffix output files')
    parser.add_argument('--save', dest='save_behavior', type=str, default='best',
                        help='Models to save (best [default], all, none)')
    parser.add_argument('--val-size', dest='val_size', type=float, default=0.2,
                        help='Proportion of data to use for validation')
    parser.add_argument('--n-models', dest='n_models', type=int, default=100,
                        help='Number of models to build')
    parser.add_argument('--cluster-fraction', dest='cluster_fraction', type=float, default=0.25,
                        help='Fraction of patient states to sample for state clustering')
    parser.add_argument('--n-cluster-init', dest='n_cluster_init', type=int, default=32,
                        help='Number of cluster initializations to try in each replicate')
    parser.add_argument('--n-cluster-states', dest='n_cluster_states', type=int, default=750,
                        help='Number of states to define through clustering')
    parser.add_argument('--n-action-bins', dest='n_action_bins', type=int, default=5,
                        help='Number of action bins for fluids and vasopressors')
    parser.add_argument('--reward', dest='reward', type=int, default=100,
                        help='Value to assign as positive reward if discharged from hospital, or negative reward if died')
    parser.add_argument('--transition-threshold', dest='transition_threshold', type=int, default=5,
                        help='Prune state-action pairs with less than this number of occurrences in training data')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99,
                        help='Decay for reward values (default 0.99)')
    parser.add_argument('--soften-factor', dest='soften_factor', type=float, default=0.01,
                        help='Amount by which to soften factors (random actions will be chosen this proportion of the time)')
    parser.add_argument('--num-iter-ql', dest='num_iter_ql', type=int, default=6,
                        help='Number of bootstrappings to use for TD learning (physician policy)')
    parser.add_argument('--num-iter-wis', dest='num_iter_wis', type=int, default=750,
                        help='Number of bootstrappings to use for WIS estimation (AI policy)')
    args = parser.parse_args()
    
    n_cluster_states = args.n_cluster_states
    n_states = n_cluster_states + 2 # discharge and death are absorbing states
    absorbing_states = [n_cluster_states + 1, n_cluster_states] # absorbing state numbers
    rewards = [args.reward, -args.reward]

    n_action_bins = args.n_action_bins
    n_actions = n_action_bins * n_action_bins # for both vasopressors and fluids

    data_dir = args.data
    out_dir = os.path.join(data_dir, "models")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    model_specs_dir = os.path.join(out_dir, "model_params{}".format('_' + args.worker_label if args.worker_label else ''))
    if os.path.exists(model_specs_dir):
        shutil.rmtree(model_specs_dir)
    os.mkdir(model_specs_dir)

    MIMICraw = load_csv(os.path.join(data_dir, "train", "MIMICraw.csv"))
    MIMICzs = load_csv(os.path.join(data_dir, "train", "MIMICzs.csv"))
    metadata = load_csv(os.path.join(data_dir, "train", "metadata.csv"))
    unique_icu_stays = metadata[C_ICUSTAYID].unique()
    
    # Bin vasopressor and fluid actions
    print("Create actions")    
    all_actions, action_medians, action_bins = fit_action_bins(
        MIMICraw[C_INPUT_STEP],
        MIMICraw[C_MAX_DOSE_VASO],
        n_action_bins=n_action_bins
    )
    
    all_model_stats = []
    max_wis_lb = 0 # Save the best model with value > 0
    min_wis_lb = 1e9 # Also save the worst model

    np.seterr(divide='ignore', invalid='ignore')
    
    for modl in range(args.n_models):
        print("Model {} of {}".format(modl, args.n_models))
    
        # split into a train and validation set
        train_ids, val_ids = train_test_split(unique_icu_stays, test_size=args.val_size)
        train_indexes = metadata[metadata[C_ICUSTAYID].isin(train_ids)].index
        val_indexes = metadata[metadata[C_ICUSTAYID].isin(val_ids)].index

        X_train = MIMICzs.iloc[train_indexes]
        X_val = MIMICzs.iloc[val_indexes]
        blocs_train = metadata[C_BLOC].values[train_indexes]
        blocs_val = metadata[C_BLOC].values[val_indexes]
        stay_ids_train = metadata[C_ICUSTAYID].values[train_indexes]
        stay_ids_val = metadata[C_ICUSTAYID].values[val_indexes]
        outcomes_train = metadata[C_OUTCOME].values[train_indexes]
        outcomes_val = metadata[C_OUTCOME].values[val_indexes]
        actions_train = all_actions[train_indexes]
        actions_val = all_actions[val_indexes]
        
        model_stats = {}
        
        # Find best clustering solution
        print("Clustering")
        clusterer, states_train = cluster_states(
            X_train.values,
            fit_fraction=args.cluster_fraction,
            n_cluster_init=args.n_cluster_init,
            n_clusters=n_cluster_states)
        
        # Create qldata3
        qldata3 = build_complete_record_sequences(
            blocs_train,
            stay_ids_train,
            states_train,
            actions_train,
            outcomes_train,
            absorbing_states,
            rewards
        )
        
        ####### BUILD MODEL ########
        physpol, transitionr, R = compute_physician_policy(
            qldata3,
            n_states,
            n_actions,
            absorbing_states,
            reward_val=args.reward,
            transition_threshold=args.transition_threshold
        )
        
        print("Policy iteration")
        Q = compute_optimal_policy(physpol, transitionr, R, gamma=args.gamma)
        optimal_actions = Q.argmax(axis=1)
        
        ####### EVALUATE ON MIMIC TRAIN SET ########

        print("Evaluate on MIMIC training set")
        train_bootql, train_bootwis = evaluate_policy(
            qldata3,
            optimal_actions,
            physpol,
            n_cluster_states,
            soften_factor=args.soften_factor,
            gamma=args.gamma,
            num_iter_ql=args.num_iter_ql,
            num_iter_wis=args.num_iter_wis
        )

        model_stats['train_bootql_mean'] = np.nanmean(train_bootql)
        model_stats['train_bootql_0.99'] = np.quantile(train_bootql, 0.99)
        model_stats['train_bootql_0.95'] = np.quantile(train_bootql, 0.99)
        model_stats['train_bootwis_mean'] = np.nanmean(train_bootwis)  #we want this as high as possible
        model_stats['train_bootwis_0.05'] = np.quantile(train_bootwis, 0.05)  #we want this as high as possible
        model_stats['train_bootwis_0.95'] = np.quantile(train_bootwis, 0.95)  #we want this as high as possible

        ####### EVALUATE ON MIMIC VALIDATION SET ########
        
        print("Evaluate on MIMIC validation set")
        states_val = clusterer.predict(X_val.values)
        
        # Create qldata3
        qldata3 = build_complete_record_sequences(
            blocs_val,
            stay_ids_val,
            states_val,
            actions_val,
            outcomes_val,
            absorbing_states,
            rewards
        )
        
        val_bootql, val_bootwis = evaluate_policy(
            qldata3,
            optimal_actions,
            physpol,
            n_cluster_states,
            soften_factor=args.soften_factor,
            gamma=args.gamma,
            num_iter_ql=args.num_iter_ql,
            num_iter_wis=args.num_iter_wis
        )

        model_stats['val_bootql_0.95'] = np.quantile(val_bootql, 0.95)   #PHYSICIANS' 95# UB
        model_stats['val_bootql_mean'] = np.nanmean(val_bootql)
        model_stats['val_bootql_0.99'] = np.quantile(val_bootql, 0.99)
        model_stats['val_bootwis_mean'] = np.nanmean(val_bootwis)    
        model_stats['val_bootwis_0.01'] = np.quantile(val_bootwis, 0.01)  
        wis_95lb = np.quantile(val_bootwis, 0.05)  #AI 95# LB, we want this as high as possible
        model_stats['val_bootwis_0.05'] = wis_95lb
        model_stats['val_bootwis_0.95'] = np.quantile(val_bootwis, 0.95)
        print("95% LB: {:.2f}".format(wis_95lb))
        
        all_model_stats.append(model_stats)
        
        best = False
        worst = False
        if wis_95lb > max_wis_lb:
            best = True
            max_wis_lb = wis_95lb
        if wis_95lb < min_wis_lb:
            worst = True
            min_wis_lb = wis_95lb
        
        save_name = None
        if args.save_behavior == 'best' and best:
            save_name = 'best_model'
        elif args.save_behavior == 'best' and worst:
            save_name = 'worst_model'
        elif args.save_behavior == 'all':
            save_name = 'model_' + str(modl)
        if save_name:
            print("Saving model:", save_name)
            model_data = {
                'model_num': modl,
                'Qon': Q,
                'optimal_actions': optimal_actions,
                'physician_policy': physpol,
                'T': transitionr,
                'R': R,
                'clusterer': clusterer,
                'train_ids': train_ids,
                'val_ids': val_ids,
                'train_bootql': train_bootql,
                'train_bootwis': train_bootwis,
                'val_bootql': val_bootql,
                'val_bootwis': val_bootwis,
                'action_medians': action_medians,
                'action_bins': action_bins,
                'n_cluster_states': n_cluster_states,
                'absorbing_states': absorbing_states,
                'rewards': rewards
            }
            save_path = os.path.join(
                model_specs_dir,
                '{}.pkl'.format(save_name))
            with open(save_path, 'wb') as file:
                pickle.dump(model_data, file)

    all_model_stats = pd.DataFrame(all_model_stats)
    all_model_stats.to_csv(os.path.join(out_dir, "model_stats{}.csv".format('_' + args.worker_label if args.worker_label else '')),
                           float_format='%.6f')
    print('Done.')