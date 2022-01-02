import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from .columns import *
from .utils import load_csv, load_intermediate_or_raw_csv

def build_states_and_actions(df, qstime, inputMV, inputpreadm, vasoMV, demog, UOpreadm, UO, timestep_resolution, winb4, winaft, head=None):
    """
    Performs two tasks: bins the data into time intervals defined by 
    timestep_resolution, and adds input and output information (vasopressors,
    fluids, and urine output).
    """
    icustayidlist = np.unique(df[C_ICUSTAYID])
    icustayidlist = sorted(icustayidlist[~pd.isna(icustayidlist)])
    print("{} ICU stay IDs".format(len(icustayidlist)))
    if head:
        icustayidlist = icustayidlist[:head]

    combined_data = []
    bin_indexes = pd.Series(index=np.arange(len(df)), dtype=pd.Int64Dtype()) # Pointers corresponding to each row in df that point to the index of combined_data that this row went into
    
    # -52 until +28 = 80 hours in total
    total_duration = (winb4 + 3) + (winaft + 3)
    for icustayid in tqdm(icustayidlist, desc='Building states and actions'):

        # CHARTEVENTS AND LAB VALUES
        temp = df.loc[df[C_ICUSTAYID] == icustayid, :]  # subtable of interest
        beg = temp[C_TIMESTEP].iloc[0]  # timestamp of first record

        # IV FLUID STUFF
        input = inputMV.loc[inputMV[C_ICUSTAYID] == icustayid, :]  # subset of interest
        startt = input[C_STARTTIME]  # start of all infusions and boluses
        endt = input[C_ENDTIME]  # end of all infusions and boluses
        rate = input[C_NORM_INFUSION_RATE]  # normalized rate of infusion (is NaN for boluses) || corrected for tonicity

        pread = inputpreadm.loc[inputpreadm[C_ICUSTAYID] == icustayid, C_INPUT_PREADM]  # preadmission volume
        if not pread.empty:  # store the value, if available
            totvol = pread.sum()
        else:
            totvol = 0  # if not documented: it's zero

        # compute volume of fluid given before start of record!!!
        t0 = 0
        t1 = beg
        # input from MV (4 ways to compute)
        infu = np.nansum(rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 +
                        rate * (endt - t0) * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 +
                        rate * (t1 - startt) * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 +
                        rate * (t1 - t0) * ((endt >= t1) & (startt <= t0)) / 3600)
        # all boluses received during this timestep, from inputMV (rate is always NaN for boluses)
        bolus = np.nansum(input.loc[pd.isna(input[C_RATE]) & (input[C_STARTTIME] >= t0) & (input[C_STARTTIME] <= t1), C_TEV])
        totvol = np.nansum([totvol, infu, bolus])

        # VASOPRESSORS
        vaso1 = vasoMV.loc[vasoMV[C_ICUSTAYID] == icustayid, :]  # subset of interest
        startv = vaso1[C_STARTTIME]  # start of VP infusion
        endv = vaso1[C_ENDTIME]  # end of VP infusions
        ratev = vaso1[C_RATESTD]  # rate of VP infusion

        # DEMOGRAPHICS / gender, age, elixhauser, re-admit, died in hosp?, died within
        # 48h of out_time (likely in ICU or soon after), died within 90d after admission?
        demog_row = demog[demog[C_ICUSTAYID] == icustayid].iloc[0]
        dem = {
            C_GENDER: demog_row[C_GENDER],
            C_AGE: demog_row[C_AGE],
            C_ELIXHAUSER: demog_row[C_ELIXHAUSER],
            C_RE_ADMISSION: demog_row[C_ADM_ORDER] > 1,
            C_DIED_IN_HOSP: demog_row[C_MORTA_HOSP],
            C_DIED_WITHIN_48H_OF_OUT_TIME: abs((demog_row[C_DOD]) - (demog_row[C_OUTTIME])) < (24 * 3600 * 2),
            C_MORTA_90: demog_row[C_MORTA_90],
            C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH: (qstime.loc[icustayid, C_DISCHTIME] - qstime.loc[icustayid, C_LAST_TIMESTEP]) / 3600
        }
        
        # URINE OUTPUT
        output = UO.loc[UO[C_ICUSTAYID] == icustayid, :]  # urine output for this icu stay
        pread = UOpreadm.loc[UOpreadm[C_ICUSTAYID] == icustayid, C_VALUE]  # preadmission UO
        if not pread.empty:  # store the value, if available
            UOtot = pread.sum()
        else:
            UOtot = 0
        # adding the volume of urine produced before start of recording! this could be
        # during the ICU stay but before the sepsis window
        UOnow = np.sum(output.loc[(output[C_CHARTTIME] >= t0) & (output[C_CHARTTIME] <= t1), C_VALUE])  # t0 and t1 defined above
        UOtot += UOnow

        for j in np.arange(0, total_duration, timestep_resolution):
            t0 = 3600 * j + beg  # left limit of time window
            t1 = 3600 * (j + timestep_resolution) + beg  # right limit of time window
            value = temp.loc[(temp[C_TIMESTEP] >= t0) & (temp[C_TIMESTEP] <= t1), :]  # index of items in this time period
            if len(value) == 0:
                continue
                
            # ICUSTAY_ID, OUTCOMES, DEMOGRAPHICS
            item = {
                C_BLOC: (j / timestep_resolution) + 1,  # 'bloc' = timestep (1,2,3...)
                C_ICUSTAYID: icustayid,       # icustay_ID
                C_TIMESTEP: int(3600 * j + beg),  # t0 = lower limit of time window
            }
            # Columns 4-11: demographics and outcomes
            item.update(dem)

            # #####################   DISCUSS ADDING STUFF HERE / RANGE, MIN, MAX ETC   ################

            # Columns 12-76: mean of chart and lab values in window
            # Are there categorical items in here?
            item.update({col: value[col].mean(skipna=True) for col in SAH_FIELD_NAMES})
            # shock index = HR/SBP and P/F (not yet computed)

            # VASOPRESSORS
            # for CV: dose at timestamps.
            # for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            # ----t0---start----end-----t1----
            # ----start---t0----end----t1----
            # -----t0---start---t1---end
            # ----start---t0----t1---end----

            # MV
            v = (((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv <= t1)) |
                ((startv >= t0) & (startv <= t1)) | ((startv <= t0) & (endv >= t1)))

            if not ratev.loc[v].empty:
                v1 = ratev.loc[v].median(skipna=True)
                v2 = ratev.loc[v].max(skipna=True)
                if not pd.isna(v1) and not pd.isna(v2):
                    item[C_MEDIAN_DOSE_VASO] = v1
                    item[C_MAX_DOSE_VASO] = v2

            # INPUT FLUID
            # input from MV (4 ways to compute)
            infu = np.nansum(rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 +
                            rate * (endt - t0) * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 +
                            rate * (t1 - startt) * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 +
                            rate * (t1 - t0) * ((endt >= t1) & (startt <= t0)) / 3600)
            # all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
            bolus = np.nansum(input.loc[pd.isna(input[C_RATE]) & (input[C_STARTTIME] >= t0) & (input[C_STARTTIME] <= t1), C_TEV])
            # sum fluid given
            totvol = np.nansum([totvol, infu, bolus])
            item[C_INPUT_TOTAL] = totvol # total fluid
            item[C_INPUT_STEP] = infu + bolus # fluid at this step

            # UO
            UOnow = np.nansum(output.loc[(output[C_CHARTTIME] >= t0) & (output[C_CHARTTIME] <= t1), C_VALUE])
            UOtot = np.nansum([UOtot, UOnow])
            item[C_OUTPUT_TOTAL] = UOtot  # total UO
            item[C_OUTPUT_STEP] = UOnow  # UO at this step

            # CUMULATED BALANCE
            item[C_CUMULATED_BALANCE] = totvol - UOtot  # cumulated balance
            
            bin_indexes[value.index] = len(combined_data) # Point to this row
            combined_data.append(item)

    result = pd.DataFrame(combined_data)
    expected_columns = DEMOGRAPHICS_FIELD_NAMES + CHART_FIELD_NAMES + LAB_FIELD_NAMES + IO_FIELD_NAMES + COMPUTED_FIELD_NAMES
    for col in expected_columns:
        if col not in result.columns:
            print("Adding empty column '{}' (no data points)".format(col))
            result[col] = pd.NA

    mapping_df = pd.DataFrame({
        C_BLOC: df[C_BLOC],
        C_ICUSTAYID: df[C_ICUSTAYID],
        C_TIMESTEP: df[C_TIMESTEP],
        C_BIN_INDEX: bin_indexes
    })
    return result, mapping_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generates a preliminary CSV '
        'containing patient state information at each timestep for which data '
        'is available (no binning). Also generates a dataframe of relevant '
        'timestamps for each patient (qstime).'))
    parser.add_argument('input', type=str,
                        help='Patient states file')
    parser.add_argument('qstime', type=str,
                        help='Path to qstime.csv file')
    parser.add_argument('output', type=str,
                        help='CSV path to write output')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Directory in which raw and preprocessed data is stored (default is data/ directory)')
    parser.add_argument('--resolution', dest='resolution', type=float, default=4.0,
                        help="Number of hours per binned timestep")
    parser.add_argument('--window-before', dest='window_before', type=int, default=49,
                        help="Number of hours before sepsis onset to include data (default 49)")
    parser.add_argument('--window-after', dest='window_after', type=int, default=25,
                        help="Number of hours after sepsis onset to include data (default 25)")
    parser.add_argument('--head', dest='head', type=int, default=None,
                        help='Number of ICU stays to convert')
    parser.add_argument('--mapping-file', dest='mapping_file', type=str, default=None,
                        help='Path to output a CSV file mapping rows of input to rows of output')

    args = parser.parse_args()
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = args.data_dir or os.path.join(base_path, 'data')

    print("Reading states...")
    df = load_csv(args.input)
    qstime = load_csv(args.qstime)
    qstime = qstime.set_index(C_ICUSTAYID, drop=True)
    
    print("Reading data files...")
    demog = load_intermediate_or_raw_csv(data_dir, 'demog.csv')
    inputpreadm = load_intermediate_or_raw_csv(data_dir, 'preadm_fluid.csv')
    inputMV = load_intermediate_or_raw_csv(data_dir, 'fluid_mv.csv')
    vasoMV = load_intermediate_or_raw_csv(data_dir, 'vaso_mv.csv')
    UOpreadm = load_intermediate_or_raw_csv(data_dir, 'preadm_uo.csv')
    UO = load_intermediate_or_raw_csv(data_dir, 'uo.csv')
    
    result, mapping = build_states_and_actions(
        df,
        qstime,
        inputMV,
        inputpreadm,
        vasoMV,
        demog,
        UOpreadm,
        UO,
        args.resolution,
        args.window_before,
        args.window_after,
        head=args.head
    )
    
    print("Writing to file")
    result.to_csv(args.output, index=False, float_format='%g')
    
    if args.mapping_file:
        mapping.to_csv(args.mapping_file, index=False, float_format='%g')