import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from ai_clinician.preprocessing.utils import load_csv
from ai_clinician.preprocessing.columns import *
from ai_clinician.preprocessing.imputation import impute_icustay_ids

tqdm.pandas()

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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

    in_dir = args.input_dir or os.path.join(PARENT_DIR, 'data', 'raw_data')
    out_dir = args.output_dir or os.path.join(PARENT_DIR, 'data', 'intermediates')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not args.no_simplify_events:        
        paths = [p for p in os.listdir(in_dir) if p.startswith("labs") or p.startswith("ce")]
        for file_name in tqdm(paths):
            if file_name.startswith("labs"):
                ref_vals = REF_LABS
            elif file_name.startswith("ce"):
                ref_vals = REF_VITALS
            
            df = pd.read_csv(os.path.join(in_dir, file_name), dtype={C_ITEMID: int})
            
            # Build a converter that maps itemid codes to the row of the reference list in which they are present
            converter = {x: i + 1
                        for i, ids in enumerate(ref_vals)
                        for x in ids}
            
            # Replace each itemid using the converter
            df[C_ITEMID].replace(converter, inplace=True)
            df.to_csv(os.path.join(out_dir, file_name), index=False)

    demog = load_csv(os.path.join(in_dir, 'demog.csv'), null_icustayid=True)
    abx = load_csv(os.path.join(in_dir, 'abx.csv'), null_icustayid=True)

    if not args.no_bacterio:
        # Produce bacterio dataframe by combining microbio and culture
        print("Generating bacterio")
        culture = load_csv(os.path.join(in_dir, 'culture.csv'), null_icustayid=True)
        microbio = load_csv(os.path.join(in_dir, 'microbio.csv'), null_icustayid=True)

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
    
    print("Correcting nans [mechvent]")
    mechvent = load_csv(os.path.join(in_dir, 'mechvent.csv'), null_icustayid=True)
    mechvent = mechvent[~pd.isna(mechvent[C_ICUSTAYID])]
    mechvent.to_csv(os.path.join(out_dir, "mechvent.csv"), index=False)
    
    print("Correcting nans [vaso_mv]")
    vaso_mv = load_csv(os.path.join(in_dir, 'vaso_mv.csv'), null_icustayid=True)
    vaso_mv = vaso_mv[~pd.isna(vaso_mv[C_ICUSTAYID])]
    vaso_mv.to_csv(os.path.join(out_dir, "vaso_mv.csv"), index=False)
    
    try:
        vaso_cv = load_csv(os.path.join(in_dir, 'vaso_cv.csv'), null_icustayid=True)
        print("Correcting nans [vaso_cv]")
        vaso_cv = vaso_cv[~pd.isna(vaso_cv[C_ICUSTAYID])]
        vaso_cv.to_csv(os.path.join(out_dir, "vaso_cv.csv"), index=False)
    except FileNotFoundError:
        print("No vaso_cv file found, skipping")
    
    try:
        fluid_cv = load_csv(os.path.join(in_dir, 'fluid_cv.csv'), null_icustayid=True)
        print("Correcting nans [fluid_cv]")
        fluid_cv = fluid_cv[~pd.isna(fluid_cv[C_ICUSTAYID])]
        fluid_cv.to_csv(os.path.join(out_dir, "fluid_cv.csv"), index=False)
    except FileNotFoundError:
        print("No fluid_cv file found, skipping")
