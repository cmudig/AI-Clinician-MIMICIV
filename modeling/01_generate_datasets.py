import numpy as np
import pandas as pd
from tqdm
import argparse
import os
from preprocessing.utils import load_csv
from modeling.columns import *
from preprocessing.columns import *
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tqdm.tqdm.pandas()

def preprocess_normalized_data(MIMICzs):
    """Performs ad-hoc normalization on the normalized variables."""
    
    MIMICzs[pd.isna(MIMICzs)] = 0
    MIMICzs[C_MAX_DOSE_VASO] = np.log(MIMICzs[C_MAX_DOSE_VASO] + 6)   # MAX DOSE NORAD 
    MIMICzs[C_INPUT_STEP] = 2 * MIMICzs[C_INPUT_STEP]   # increase weight of this variable
    return MIMICzs

def clip_and_log_transform(data, log_gamma=0.1):
    """Performs a log transform log(gamma + x), and clips x values less than zero to zero."""
    print("Clipping log columns with values less than zero at proportions:", (data < 0).mean())
    return np.log(log_gamma + np.clip(data, 0, None))

def save_data_files(dir, MIMICraw, MIMICzs, metadata):
    MIMICraw.to_csv(os.path.join(dir, "MIMICraw.csv"), index=False)
    MIMICzs.to_csv(os.path.join(dir, "MIMICzs.csv"), index=False)
    metadata.to_csv(os.path.join(dir, "metadata.csv"), index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Generates a train/test '
        'split of the MIMIC-IV dataset, and generates files labeled '
        '{train|test}/MIMICraw.npy and {train|test}/MIMICzs.npy.'))
    parser.add_argument('input', type=str,
                        help='Data directory (should contain mimic_dataset.csv and sepsis_cohort.csv)')
    parser.add_argument('output', type=str,
                        help='Directory in which to output')
    parser.add_argument('--train-size', dest='train_size', type=float, default=0.6,
                        help='Proportion of data to use in training (default 0.6)')
    parser.add_argument('--outcome', dest='outcome_col', type=str, default='died_in_hosp',
                        help='Name of column to use for outcomes (probably "died_in_hosp" [default] or "morta_90")')

    args = parser.parse_args()

    in_dir = args.input
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Find sepsis cohort in the mimic dataset
    mdp_data = load_csv(os.path.join(in_dir, "mimic_dataset.csv"))
    sepsis_cohort = load_csv(os.path.join(in_dir, "sepsis_cohort.csv"))

    MIMICtable = mdp_data[mdp_data[C_ICUSTAYID].isin(sepsis_cohort[C_ICUSTAYID])].reset_index(drop=True)
    assert args.outcome_col in MIMICtable.columns, "Outcome column '{}' not found in MIMICtable".format(args.outcome_col)


    # find patients who died in ICU during data collection period
    icuuniqueids = MIMICtable[C_ICUSTAYID].unique()
    train_ids, test_ids = train_test_split(icuuniqueids, train_size=args.train_size)
    train_indexes = MIMICtable[MIMICtable[C_ICUSTAYID].isin(train_ids)].index
    test_indexes = MIMICtable[MIMICtable[C_ICUSTAYID].isin(test_ids)].index
    print("Training: {} IDs ({} rows)".format(len(train_ids), len(train_indexes)))
    print("Test: {} IDs ({} rows)".format(len(test_ids), len(test_indexes)))

    MIMICraw = MIMICtable[ALL_FEATURE_COLUMNS]

    print("Proportion of NA values:", MIMICraw.isna().sum() / len(MIMICraw))

    no_norm_scores = MIMICtable[AS_IS_COLUMNS].astype(np.float64).values - 0.5
    scores_to_norm = np.hstack([MIMICtable[NORM_COLUMNS].astype(np.float64).values,
                                clip_and_log_transform(MIMICtable[LOG_NORM_COLUMNS])])
    scaler = StandardScaler()
    
    normed_train = scaler.fit_transform(scores_to_norm[train_indexes])
    normed_test = scaler.transform(scores_to_norm[test_indexes])
    
    MIMICzs_train = pd.DataFrame(np.hstack([no_norm_scores[train_indexes], normed_train]), columns=ALL_FEATURE_COLUMNS)
    MIMICzs_train = preprocess_normalized_data(MIMICzs_train)
    MIMICzs_test = pd.DataFrame(np.hstack([no_norm_scores[test_indexes], normed_test]), columns=ALL_FEATURE_COLUMNS)
    MIMICzs_test = preprocess_normalized_data(MIMICzs_test)

    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        
    metadata = MIMICtable[[C_BLOC, C_ICUSTAYID, args.outcome_col]].rename({args.outcome_col: C_OUTCOME}, axis=1)
    
    # Save files
    print("Saving files")
    save_data_files(train_dir,
                    MIMICraw.iloc[train_indexes],
                    MIMICzs_train,
                    metadata.iloc[train_indexes])
    save_data_files(test_dir,
                    MIMICraw.iloc[test_indexes],
                    MIMICzs_test,
                    metadata.iloc[test_indexes])    
    print("Done.")
    