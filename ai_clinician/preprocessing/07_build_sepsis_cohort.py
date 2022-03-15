import pandas as pd
import numpy as np
import os
import argparse
import tqdm
from ai_clinician.preprocessing.columns import *
from ai_clinician.preprocessing.utils import load_csv

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def outlier_stay_ids(df):
    """Returns a list of ICU stay IDs that should be removed from the dataset."""
    outliers = set()
    
    # check for patients with extreme UO = outliers = to be deleted (>40 litres of UO per 4h!!)
    outliers |= set(df[df[C_OUTPUT_STEP] > 12000][C_ICUSTAYID].unique())

    # some have bili = 999999
    outliers |= set(df[df[C_TOTAL_BILI] > 10000][C_ICUSTAYID].unique())

    # check for patients with extreme INTAKE = outliers = to be deleted (>10 litres of intake per 4h!!)
    outliers |= set(df[df[C_INPUT_STEP] > 10000][C_ICUSTAYID].unique())

    return outliers
    
def treatment_stopped_stay_ids(df):
    a = df[[C_BLOC, C_ICUSTAYID, C_MORTA_90, C_MAX_DOSE_VASO, C_SOFA]]
    grouped = a.groupby(C_ICUSTAYID)
    d = pd.merge(grouped.agg('max'),
                grouped.size().rename(C_NUM_BLOCS),
                how='left',
                left_index=True,
                right_index=True).drop(C_BLOC, axis=1)
    last_bloc = a.sort_values(C_BLOC, ascending=False).drop_duplicates(C_ICUSTAYID).rename({
        C_MAX_DOSE_VASO: C_LAST_VASO,
        C_SOFA: C_LAST_SOFA
    }, axis=1).drop(C_MORTA_90, axis=1)
    d = pd.merge(d,
                last_bloc,
                how='left',
                left_index=True,
                right_on=C_ICUSTAYID).set_index(C_ICUSTAYID, drop=True)
    
    stopped_treatment = d[
        (d[C_MORTA_90] == 1) & 
        (pd.isna(d[C_LAST_VASO]) | (d[C_LAST_VASO] < 0.01)) &
        (d[C_MAX_DOSE_VASO] > 0.3) &
        (d[C_LAST_SOFA] >= d[C_SOFA] / 2) &
        (d[C_NUM_BLOCS] < 20)
    ].index
    
    return stopped_treatment

def died_in_icu_stay_ids(df):
    # exclude patients who died in ICU during data collection period
    died_in_icu = df[
        (df[C_DIED_WITHIN_48H_OF_OUT_TIME] == 1) &
        (df[C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH] < 24)
    ][C_ICUSTAYID].unique()
    return died_in_icu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Filters the state/action '
        'dataframe to generate a sepsis cohort.'))
    parser.add_argument('input', type=str,
                        help='Path to patient states and actions CSV file')
    parser.add_argument('qstime', type=str,
                        help='Path to qstime.csv file')
    parser.add_argument('output', type=str,
                        help='Directory in which to write output')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is ../data/ directory)')
    parser.add_argument('--no-outlier-exclusion', dest='outlier_exclusion', default=True, action='store_false',
                        help="Don't exclude outliers by lab values")
    
    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(PARENT_DIR, 'data')
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    df = load_csv(args.input)
    qstime = load_csv(args.qstime)
    qstime = qstime.set_index(C_ICUSTAYID, drop=True)
    
    print("Before filtering:", len(set(df[C_ICUSTAYID])), "ICU stays")  # count before

    if args.outlier_exclusion:
        outliers = outlier_stay_ids(df)
        print(len(outliers), "outliers to remove")
        df = df[~df[C_ICUSTAYID].isin(outliers)]
        
    stopped_treatment = treatment_stopped_stay_ids(df)
    print(len(stopped_treatment), "stays to remove because treatment was stopped and patient died")
    df = df[~df[C_ICUSTAYID].isin(stopped_treatment)]

    died_in_icu = died_in_icu_stay_ids(df)    
    print(len(died_in_icu), "patients to remove because died in ICU during data collection")
    df = df[~df[C_ICUSTAYID].isin(died_in_icu)]

    print("After filtering:", len(set(df[C_ICUSTAYID])), "ICU stays")  # count after

    sepsis = df.groupby('icustayid').agg({
        C_MORTA_90: 'first',
        C_SOFA: 'max',
        C_SIRS: 'max',
    }).rename({
        C_SOFA: C_MAX_SOFA,
        C_SIRS: C_MAX_SIRS,
    }, axis=1)
    sepsis = sepsis[sepsis[C_MAX_SOFA] >= 2]
    print(len(sepsis), "patients with max SOFA >= 2")
    
    sepsis = pd.merge(sepsis, qstime[C_ONSET_TIME], how='left', left_index=True, right_index=True)
    print("Write")
    sepsis.to_csv(os.path.join(out_dir, "sepsis_cohort.csv"))
