import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import time
from preprocessing.columns import *
from preprocessing.utils import load_csv, load_intermediate_or_raw_csv

class ChartEvents:
    """
    An object that manages a set of chart event tables, and 
    supports retrieving all chart events for a given ICU stay ID.
    """
    def __init__(self, ce_dfs, stay_id_col=C_ICUSTAYID):
        super().__init__()
        self.dfs = ce_dfs
        self.ranges = [(df[stay_id_col].min(), df[stay_id_col].max())
                       for df in ce_dfs]
        self.stay_id_col = stay_id_col
        
    def fetch(self, stay_id):
        results = []
        for df, (min_id, max_id) in zip(self.dfs, self.ranges):
            if stay_id >= min_id and stay_id <= max_id:
                results.append(df[df[self.stay_id_col] == stay_id])
        if results:
            return pd.concat(results)
        return None

def time_window(df, col, center_time, lower_window, upper_window):
    """
    Returns all rows in the given data frame whose timestamp in column 'col' fall
    between center_time - lower_window and center_time + upper_window.
    """
    return df[(df[col] >= center_time - lower_window) & (df[col] <= center_time + upper_window)]

def build_patient_states(chart_events, onset_data, demog, labU, MV, MV_procedure, winb4, winaft):
    """
    Builds the patient states dataframe without imputation. Returns two dataframes,
    one containing the patient states and the other containing the qstime
    dataframe, which maps ICU stay IDs to relevant timestamps in the stay.
    """    
    combined_data = []
    infection_times = []
    from_pe = 0
    
    for _, row in tqdm(onset_data.iterrows(), total=len(onset_data), desc='Building patient states'):
        qst = row[C_ONSET_TIME]  # flag for presumed infection
        icustayid = int(row[C_ICUSTAYID])
        assert qst > 0
        d1 = demog.loc[demog[C_ICUSTAYID] == icustayid, [C_AGE, C_DISCHTIME]].values.tolist()

        if d1[0][0] < 18:  # exclude younger than 18
            continue

        # Limit to a time period of -4h and +4h around the window of interest for sepsis3 definition)
        bounds = ((winb4 + 4) * 3600, (winaft + 4) * 3600)

        # Chart events, lab events, mech vent and extubated
        temp = time_window(chart_events.fetch(icustayid), C_CHARTTIME, qst, *bounds)
        temp2 = time_window(labU[labU[C_ICUSTAYID] == icustayid], C_CHARTTIME, qst, *bounds)
        temp3 = time_window(MV[MV[C_ICUSTAYID] == icustayid], C_CHARTTIME, qst, *bounds)
        temp4 = time_window(MV_procedure[MV_procedure[C_ICUSTAYID] == icustayid], C_STARTTIME, qst, *bounds)
        
        # list of unique timestamps from all 3 sources / sorted in ascending order
        timesteps = sorted(pd.unique(pd.concat([
            temp[C_CHARTTIME], 
            temp2[C_CHARTTIME], 
            temp3[C_CHARTTIME], 
            temp4[C_STARTTIME]], ignore_index=True)))

        if len(timesteps) == 0:
            continue

        # Reformat each event, which may contain multiple items at the same timestep
        for i, timestep in enumerate(timesteps):

            # First three values: timestep, icustay ID, chart time
            item = {
                C_BLOC: i,
                C_ICUSTAYID: icustayid,
                C_TIMESTEP: timestep
            }
            
            # CHARTEVENTS: positions 4-31 (inclusive) are the 28 unique values
            # for different vitals, as stored in referenceMatrices['Refvitals']
            for _, event in temp[temp[C_CHARTTIME] == timestep].iterrows():
                if event[C_ITEMID] <= 0 or event[C_ITEMID] > len(CHART_FIELD_NAMES):
                    continue
                item[CHART_FIELD_NAMES[event[C_ITEMID] - 1]] = event[C_VALUENUM]

            # LAB VALUES: positions 32-66 (inclusive) are the 35 unique values
            # for different lab tests, as stored in referenceMatrices['Reflabs']
            for _, event in temp2[temp2[C_CHARTTIME] == timestep].iterrows():
                if event[C_ITEMID] <= 0 or event[C_ITEMID] > len(LAB_FIELD_NAMES):
                    continue
                item[LAB_FIELD_NAMES[event[C_ITEMID] - 1]] = event[C_VALUENUM]

            # MV: positions 67 and 68 store whether the measurement is relating
            # to being on a ventilator, and whether the patient was extubated
            matching_mv = (temp3[C_CHARTTIME] == timestep)
            if matching_mv.sum() > 0:
                if matching_mv.sum() > 1:
                    print(matching_mv.sum(), "MV items")
                event = temp3[matching_mv].iloc[0]
                item[C_MECHVENT] = event[C_MECHVENT]
                item[C_EXTUBATED] = event[C_EXTUBATED]
            
            # Second source of mechanical ventilation information
            if C_MECHVENT not in item and (temp4[C_STARTTIME] == timestep).sum() > 0:
                events = temp4[temp4[C_STARTTIME] == timestep]
                item[C_MECHVENT] = events[C_MECHVENT].any().astype(int)
                item[C_EXTUBATED] = events[C_EXTUBATED].any().astype(int)
                from_pe += 1
                
            combined_data.append(item)            

        infection_times.append({
            C_ICUSTAYID: icustayid,
            C_ONSET_TIME: qst,
            C_FIRST_TIMESTEP: timesteps[0], # first timestep
            C_LAST_TIMESTEP: timesteps[-1], # last timestep
            C_DISCHTIME: d1[0][1] # discharge time
        })
                
    print("Got {} items from procedure events".format(from_pe))
    state_df = pd.DataFrame(combined_data)
    qstime = pd.DataFrame(infection_times).set_index(C_ICUSTAYID)
    
    expected_columns = CHART_FIELD_NAMES + LAB_FIELD_NAMES
    for col in expected_columns:
        if col not in state_df.columns:
            print("Adding empty column '{}' (no data points)".format(col))
            state_df[col] = pd.NA
    return state_df, qstime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generates a preliminary CSV '
        'containing patient state information at each timestep for which data '
        'is available (no binning). Also generates a dataframe of relevant '
        'timestamps for each patient (qstime).'))
    parser.add_argument('output_dir', type=str,
                        help='Directory in which to output (e.g. data/intermediates/patient_states)')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is data/ directory)')
    parser.add_argument('--window-before', dest='window_before', type=int, default=49,
                        help="Number of hours before sepsis onset to include data (default 49)")
    parser.add_argument('--window-after', dest='window_after', type=int, default=25,
                        help="Number of hours after sepsis onset to include data (default 25)")
    parser.add_argument('--head', dest='head', type=int, default=None,
                        help='Number of rows at the beginning of onset data to convert to patient states')
    parser.add_argument('--filter-stays', dest='filter_stays_path', type=str, default=None,
                        help='Path to a CSV file containing an icustayid column; output will be filtered to these ICU stays')

    args = parser.parse_args()
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = args.data_dir or os.path.join(base_path, 'data')
    out_dir = args.output_dir or os.path.join(base_path, 'data', 'intermediates', 'patient_states')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    id_step = int(1e6)
    id_max = int(1e7)
    print("Reading chartevents...")
    chart_events = ChartEvents([load_intermediate_or_raw_csv(data_dir, 'ce{}{}.csv'.format(i, i + id_step))
                                for i in range(0, id_max, id_step)])
    print("Reading onset data...")
    onset_data = load_csv(os.path.join(data_dir, 'intermediates', 'sepsis_onset.csv'))
    print("Reading demog...")
    demog = load_intermediate_or_raw_csv(data_dir, 'demog.csv')
    print("Reading labs...")
    labU = pd.concat([
        load_intermediate_or_raw_csv(data_dir, 'labs_ce.csv'), 
        load_intermediate_or_raw_csv(data_dir, 'labs_le.csv')
    ], ignore_index=True)

    print("Reading mechvent...")
    MV = load_intermediate_or_raw_csv(data_dir, 'mechvent.csv')
    MV_procedure = load_intermediate_or_raw_csv(data_dir, 'mechvent_pe.csv')
    
    if args.filter_stays_path:
        print("Reading filter stays...")
        allowed_stays_df = load_csv(args.filter_stays_path)
        allowed_stays = allowed_stays_df[C_ICUSTAYID]
        old_count = len(onset_data)
        onset_data = onset_data[onset_data[C_ICUSTAYID].isin(allowed_stays)]
        print("Filtered from {} to {} ICU stay ids".format(old_count, len(onset_data)))

    if args.head: onset_data = onset_data.head(args.head)
    state_df, qstime = build_patient_states(
        chart_events,
        onset_data,
        demog,
        labU,
        MV,
        MV_procedure,
        args.window_before,
        args.window_after
    )
    
    print("Result: state_df contains {} rows, {} columns".format(len(state_df), len(state_df.columns)))
    print("Data availability:")
    for col in sorted(state_df.columns):
        nan_fraction = pd.isna(state_df[col]).sum() / len(state_df)
        if nan_fraction > 0.0:
            print(col, nan_fraction)
    print("")
    
    state_df.to_csv(os.path.join(out_dir, 'patient_states.csv'), index=False, float_format='%g')
    qstime.to_csv(os.path.join(out_dir, 'qstime.csv'), float_format='%g')