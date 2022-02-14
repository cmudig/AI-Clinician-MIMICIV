import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from ai_clinician.preprocessing.columns import C_ICUSTAYID
from ai_clinician.preprocessing.utils import load_csv, load_intermediate_or_raw_csv
from ai_clinician.preprocessing.derived_features import calculate_onset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Calculates the presumed time '
        'of sepsis onset for each patient, and generates a sepsis_onset.csv file '
        'in the data/intermediates directory.'))
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is data/ directory)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Directory in which to output (default is data/intermediates directory)')

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    out_dir = args.output_dir or os.path.join(data_dir, 'intermediates')

    abx = load_intermediate_or_raw_csv(data_dir, "abx.csv")
    bacterio = load_csv(os.path.join(data_dir, "intermediates", "bacterio.csv"))
    onset_data = pd.DataFrame([onset for onset in
                            (calculate_onset(abx, bacterio, stay_id)
                                for stay_id in tqdm(abx[C_ICUSTAYID].unique()))
                            if onset is not None])
    onset_data.to_csv(os.path.join(out_dir, "sepsis_onset.csv"), index=False)