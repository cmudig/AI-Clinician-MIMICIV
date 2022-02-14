import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from ai_clinician.preprocessing.columns import *
from ai_clinician.preprocessing.provenance import ProvenanceWriter
from ai_clinician.preprocessing.utils import load_csv
from ai_clinician.preprocessing.imputation import fixgaps, knn_impute
from ai_clinician.preprocessing.derived_features import compute_pao2_fio2, compute_shock_index, compute_sofa, compute_sirs

def correct_features(df, provenance=None):
    # CORRECT GENDER
    df[C_GENDER] = df[C_GENDER] - 1

    # CORRECT AGE > 200 yo
    ii = df[C_AGE] > 150
    if provenance:
        provenance.record("clamp age", row=df.loc[ii].index, col=C_AGE)
    df.loc[ii, C_AGE] = 91

    # FIX MECHVENT
    if provenance:
        provenance.record("replace NaN mechvent", row=df.loc[pd.isna(df[C_MECHVENT])].index, col=C_MECHVENT)
    df.loc[pd.isna(df[C_MECHVENT]), C_MECHVENT] = 0
    df.loc[df[C_MECHVENT] > 0, C_MECHVENT] = 1

    # FIX Elixhauser missing values
    if provenance:
        provenance.record("replace NaN elixhauser with median", row=df.loc[pd.isna(df[C_ELIXHAUSER])].index, col=C_MECHVENT)
    df.loc[pd.isna(df[C_ELIXHAUSER]), C_ELIXHAUSER] = np.nanmedian(
        df[C_ELIXHAUSER])  # use the median value / only a few missing data points

    # vasopressors / no NAN
    if provenance:
        provenance.record("zero NaN median dose vaso", row=df.loc[pd.isna(df[C_MEDIAN_DOSE_VASO])].index, col=C_MEDIAN_DOSE_VASO)
    df.loc[pd.isna(df[C_MEDIAN_DOSE_VASO]), C_MEDIAN_DOSE_VASO] = 0
    if provenance:
        provenance.record("zero NaN max dose vaso", row=df.loc[pd.isna(df[C_MAX_DOSE_VASO])].index, col=C_MAX_DOSE_VASO)
    df.loc[pd.isna(df[C_MAX_DOSE_VASO]), C_MAX_DOSE_VASO] = 0
    
    return df    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Imputes missing values in the '
        'state/action dataframe using linear interpolation and k-nearest '
        'neighbors; replaces incorrect values of demographic features; and '
        'computes SOFA and SIRS.'))
    parser.add_argument('input', type=str,
                        help='Path to patient states and actions CSV file')
    parser.add_argument('output', type=str,
                        help='Path at which to write output')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is data/ directory)')
    parser.add_argument('--resolution', dest='resolution', type=float, default=4.0,
                        help='Timestep resolution in hours (default 4.0)')
    parser.add_argument('--no-correct-features', dest='correct_features', default=True, action='store_false',
                        help="Don't correct features NaN values")
    parser.add_argument('--no-interpolation', dest='interpolation', default=True, action='store_false',
                        help="Don't perform linear interpolation on columns where missingness is low")
    parser.add_argument('--no-knn', dest='knn', default=True, action='store_false',
                        help="Don't perform KNN imputation")
    parser.add_argument('--no-computed-features', dest='computed_features', default=True, action='store_false',
                        help="Don't compute PaO2/FiO2, shock index, SOFA, SIRS")
    parser.add_argument('--mask-file', dest='mask_file', default=None, type=str,
                        help="Path to write a mask file indicating where values were changed (+1 if a value was added or changed, or -1 if a value was removed)")
    parser.add_argument('--provenance-dir', dest='provenance_dir', default=None, type=str,
                        help="Path to directory in which to write provenance files (indicating sources and reasons for all changes)")

    args = parser.parse_args()
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = args.data_dir or os.path.join(base_path, 'data')

    df = load_csv(args.input)
    old_df = df.copy() if args.mask_file else None
    
    provenance = ProvenanceWriter(args.provenance_dir, verbose=True) if args.provenance_dir else None
    
    if args.correct_features:
        print("Correcting features")
        df = correct_features(df, provenance=provenance)

    if args.interpolation:
        miss = pd.isna(df).sum() / len(df)
        impute_columns = (miss > 0) & (miss < 0.05)  # less than 5# missingness
        for col in df.loc[:,impute_columns].columns:
            print("Linear interpolation on", col)
            new_vals = fixgaps(df[col])
            if provenance:
                provenance.record("linear interpolation", row=(df[col] != new_vals).index, col=col)
            df[col] = new_vals
    
    if args.knn:
        knn_cols = CHART_FIELD_NAMES + LAB_FIELD_NAMES
        df[knn_cols] = knn_impute(df[knn_cols], na_threshold=0.9, provenance=provenance)

    if args.computed_features:
        print("Computing P/F")
        df[C_PAO2_FIO2] = compute_pao2_fio2(df)
        print("Computing shock index")
        df[C_SHOCK_INDEX] = compute_shock_index(df)
        print("Computing SOFA")
        df[C_SOFA] = compute_sofa(df, timestep_resolution=args.resolution)
        print("Computing SIRS")
        df[C_SIRS] = compute_sirs(df)

    print("Write")
    df.to_csv(args.output, index=False, float_format='%g')
    
    if provenance:
        provenance.close()
    
    if args.mask_file:
        print("Write mask file")
        
        # Compare the old dataframe to the new one and see where values have cropped up
        mask_series = {
            C_BLOC: df[C_BLOC],
            C_ICUSTAYID: df[C_ICUSTAYID],
            C_TIMESTEP: df[C_TIMESTEP]
        }
        
        for col in set(old_df.columns) & set(df.columns):
            if col in mask_series: continue
            old = old_df[col]
            new = df[col]
            mask_series[col] = np.where((pd.isna(old) & ~pd.isna(new)) | (~pd.isna(old) & (old != new)),
                                        1, np.where(~pd.isna(old) & pd.isna(new), -1, 0))
            
        pd.DataFrame(mask_series).to_csv(args.mask_file, index=False)
            
            