import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from .utils import load_csv
from .columns import *
from .imputation import impute_icustay_ids

tqdm.pandas()

# Files to preprocess using this script
item_id_file_list = [
    'labs_le.csv',
    'labs_ce.csv', 
    'ce01000000.csv',
    'ce10000002000000.csv',
    'ce20000003000000.csv',
    'ce30000004000000.csv',
    'ce40000005000000.csv',
    'ce50000006000000.csv',
    'ce60000007000000.csv',
    'ce70000008000000.csv',
    'ce80000009000000.csv',
    'ce900000010000000.csv',
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=('Preprocesses chart and lab '
        'events to save space and deduplicate itemids to a smaller set of '
        'distinct event types. Also combines microbio and culture dataframes '
        'to produce a single bacterio CSV, and performs other boilerplate tasks '
        'on the raw data.'))
    parser.add_argument('--in', dest='input_dir', type=str, default=None,
                        help='Directory in which to read files (default is data/raw_data directory)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Directory in which to output (default is data/intermediates directory)')
    parser.add_argument('--no-events', dest='no_simplify_events', action='store_true',
                        help="If passed, don't preprocess ce and le dataframes")
    parser.add_argument('--no-bacterio', dest='no_bacterio', action='store_true',
                        help="If passed, don't produce bacterio.csv")

    args = parser.parse_args()

    in_dir = args.input_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data', 'raw_data')
    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data', 'intermediates')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not args.no_simplify_events:        
        for file_name in tqdm(item_id_file_list):
            if file_name.startswith("labs"):
                ref_vals = REF_LABS
            elif file_name.startswith("ce"):
                ref_vals = REF_VITALS
            else:
                print("Unknown file name '{}', skipping".format(file_name))
                continue
            
            df = pd.read_csv(os.path.join(in_dir, file_name), dtype={C_ITEMID: int})
            
            # Build a converter that maps itemid codes to the row of the reference list in which they are present
            converter = {x: i + 1
                        for i, ids in enumerate(ref_vals)
                        for x in ids}
            
            # Replace each itemid using the converter
            df[C_ITEMID].replace(converter, inplace=True)
            df.to_csv(os.path.join(out_dir, file_name), index=False)

    demog = load_csv(os.path.join(in_dir, 'demog.csv'))
    abx = load_csv(os.path.join(in_dir, 'abx.csv'))

    if not args.no_bacterio:
        # Produce bacterio dataframe by combining microbio and culture
        print("Generating bacterio")
        culture = load_csv(os.path.join(in_dir, 'culture.csv'))
        microbio = load_csv(os.path.join(in_dir, 'microbio.csv'))

        # use chartdate to fill in empty charttimes
        ii = microbio[C_CHARTTIME].isnull()
        microbio.loc[ii, C_CHARTTIME] = microbio.loc[ii, C_CHARTDATE]
        microbio.loc[:, C_CHARTDATE] = 0
        
        cols = [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_CHARTTIME]
        bacterio = pd.concat([
            microbio[cols], culture[cols]
        ], ignore_index=True)
        
        # Impute icu stay IDs from demog
        print("Bacterio has {}/{} null ICU stay IDs, imputing...".format(pd.isna(bacterio[C_ICUSTAYID]).sum(), len(bacterio)))
        bacterio.loc[pd.isna(bacterio[C_ICUSTAYID]), C_ICUSTAYID] = impute_icustay_ids(demog, bacterio[pd.isna(bacterio[C_ICUSTAYID])])
        print("Now:", pd.isna(bacterio[C_ICUSTAYID]).sum(), "nulls")

        bacterio = bacterio[~pd.isna(bacterio[C_ICUSTAYID])]
        bacterio.to_csv(os.path.join(out_dir, "bacterio.csv"), index=False)
        
    # Process other files
    
    # Remove abx entries with no start date or ICU stay ID
    print("Trimming unusable entries [abx]")
    abx = abx[(~pd.isna(abx[C_STARTDATE])) & (~pd.isna(abx[C_ICUSTAYID]))]
    
    print("Abx has {}/{} nulls, imputing...".format(pd.isna(abx[C_ICUSTAYID]).sum(), len(abx)))
    abx.loc[pd.isna(abx[C_ICUSTAYID]), C_ICUSTAYID] = impute_icustay_ids(demog, abx[pd.isna(abx[C_ICUSTAYID])])
    print("Now:", pd.isna(abx[C_ICUSTAYID]).sum(), "nulls")
    abx.to_csv(os.path.join(out_dir, "abx.csv"), index=False)
    
    # Correct nans in demog
    print("Correcting nans [demog]")
    demog.loc[pd.isna(demog[C_MORTA_90]), C_MORTA_90] = 0
    demog.loc[pd.isna(demog[C_MORTA_HOSP]), C_MORTA_HOSP] = 0
    demog.loc[pd.isna(demog[C_ELIXHAUSER]), C_ELIXHAUSER] = 0
    demog.to_csv(os.path.join(out_dir, "demog.csv"), index=False)
    
    # compute normalized rate of infusion and add it as a column to inputMV
    # if we give 100 ml of hypertonic fluid (600 mosm/l) at 100 ml/h (given in 1h) it is 200 ml of NS equivalent
    # so the normalized rate of infusion is 200 ml/h (different volume in same duration)
    print("Computing normalized rate of infusion [fluid_mv]")
    inputMV = load_csv(os.path.join(in_dir, 'fluid_mv.csv'))
    inputMV.loc[:, C_NORM_INFUSION_RATE] = inputMV[:][C_TEV] * inputMV[:][C_RATE] / inputMV[:][C_AMOUNT]
    inputMV.to_csv(os.path.join(out_dir, "fluid_mv.csv"), index=False)