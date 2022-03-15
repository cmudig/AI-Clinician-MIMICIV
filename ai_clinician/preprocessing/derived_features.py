import pandas as pd
import numpy as np

from ai_clinician.preprocessing.columns import *

def calculate_onset(abx, bacterio, stay_id):
    matching_abs = abx.loc[abx[C_ICUSTAYID] == stay_id, C_STARTDATE].reset_index(drop=True).sort_values()
    matching_bacts = bacterio[bacterio[C_ICUSTAYID] == stay_id].reset_index(drop=True).sort_values(C_CHARTTIME)
    if matching_abs.empty or matching_bacts.empty: return None
    for _, ab_time in matching_abs.iteritems():
        # Calculate time delay between this antibiotic and all bacterio events
        dists = [abs(bact_row[C_CHARTTIME] - ab_time) / 3600 for _, bact_row in matching_bacts.iterrows()]
        min_index = np.argmin(dists)
        bact = matching_bacts.iloc[min_index]
        
        # If the antibiotic was first and the bacterio event was < 24 hours later
        if dists[min_index] <= 24 and ab_time <= bact[C_CHARTTIME]:
            return {
                C_SUBJECT_ID: bact[C_SUBJECT_ID],
                C_ICUSTAYID: stay_id,
                C_ONSET_TIME: ab_time
            }
        elif dists[min_index] <= 72 and ab_time >= bact[C_CHARTTIME]:
            return {
                C_SUBJECT_ID: bact[C_SUBJECT_ID],
                C_ICUSTAYID: stay_id,
                C_ONSET_TIME: bact[C_CHARTTIME]
            }
    return None

def compute_pao2_fio2(df):
    if C_PAO2 not in df.columns or C_FIO2_1 not in df.columns:
        return pd.NA
    
    return df[C_PAO2] / df[C_FIO2_1]
        
def compute_shock_index(df):
    # recompute SHOCK INDEX without NAN and INF
    result = df[C_SHOCK_INDEX]
    if C_HR in df.columns and C_SYSBP in df.columns:
        result = df[C_HR] / df[C_SYSBP]
        
    result[np.isinf(result)] = pd.NA
    d = np.nanmean(result)
    print("Replacing shock index with average value", d)
    result[pd.isna(result)] = d  # replace NaN with average value ~ 0.8
    return result

def compute_sofa(df, timestep_resolution=4.0):
    s = df[[C_PAO2_FIO2, C_PLATELETS_COUNT, C_TOTAL_BILI,
            C_MEANBP, C_MAX_DOSE_VASO, C_GCS, C_CREATININE,
            C_OUTPUT_STEP]]

    s1 = pd.DataFrame(
        [s[C_PAO2_FIO2] > 400, 
        (s[C_PAO2_FIO2] >= 300) & (s[C_PAO2_FIO2] < 400),
        (s[C_PAO2_FIO2] >= 200) & (s[C_PAO2_FIO2] < 300), 
        (s[C_PAO2_FIO2] >= 100) & (s[C_PAO2_FIO2] < 200),
        s[C_PAO2_FIO2] < 100], index=range(5))
    s2 = pd.DataFrame(
        [s[C_PLATELETS_COUNT] > 150, 
        (s[C_PLATELETS_COUNT] >= 100) & (s[C_PLATELETS_COUNT] < 150), 
        (s[C_PLATELETS_COUNT] >= 50) & (s[C_PLATELETS_COUNT] < 100), 
        (s[C_PLATELETS_COUNT] >= 20) & (s[C_PLATELETS_COUNT] < 50), 
        s[C_PLATELETS_COUNT] < 20],
        index=range(5))
    s3 = pd.DataFrame(
        [s[C_TOTAL_BILI] < 1.2, 
        (s[C_TOTAL_BILI] >= 1.2) & (s[C_TOTAL_BILI] < 2), 
        (s[C_TOTAL_BILI] >= 2) & (s[C_TOTAL_BILI] < 6), 
        (s[C_TOTAL_BILI] >= 6) & (s[C_TOTAL_BILI] < 12), 
        s[C_TOTAL_BILI] > 12],
        index=range(5))
    s4 = pd.DataFrame([s[C_MEANBP] >= 70, 
        (s[C_MEANBP] < 70) & (s[C_MEANBP] >= 65), 
        (s[C_MEANBP] < 65), 
        (s[C_MAX_DOSE_VASO] > 0) & (s[C_MAX_DOSE_VASO] <= 0.1), 
        s[C_MAX_DOSE_VASO] > 0.1],
        index=range(5))
    s5 = pd.DataFrame(
        [s[C_GCS] > 14, 
        (s[C_GCS] > 12) & (s[C_GCS] <= 14), 
        (s[C_GCS] > 9) & (s[C_GCS] <= 12), 
        (s[C_GCS] > 5) & (s[C_GCS] <= 9), 
        s[C_GCS] <= 5],
        index=range(5))
    s6 = pd.DataFrame(
        [s[C_CREATININE] < 1.2, 
        (s[C_CREATININE] >= 1.2) & (s[C_CREATININE] < 2), 
        (s[C_CREATININE] >= 2) & (s[C_CREATININE] < 3.5), 
        ((s[C_CREATININE] >= 3.5) & (s[C_CREATININE] < 5)) | (s[C_OUTPUT_STEP] < 500 * timestep_resolution / 24),
        (s[C_CREATININE] > 5) | (s[C_OUTPUT_STEP] < 200 * timestep_resolution / 24)], 
        index=range(5))

    ms1 = s1.idxmax(axis=0)
    ms2 = s2.idxmax(axis=0)
    ms3 = s3.idxmax(axis=0)
    ms4 = s4.idxmax(axis=0)
    ms5 = s5.idxmax(axis=0)
    ms6 = s6.idxmax(axis=0)
    return ms1 + ms2 + ms3 + ms4 + ms5 + ms6

def compute_sirs(df):
    s = df[[C_TEMP_C, C_HR, C_RR, C_PACO2, C_WBC_COUNT]]

    s1 = (s[C_TEMP_C] >= 38) | (s[C_TEMP_C] <= 36)  # count of points for all criteria of SIRS
    s2 = (s[C_HR] > 90)
    s3 = (s[C_RR] >= 20) | (s[C_PACO2] <= 32)
    s4 = (s[C_WBC_COUNT] >= 12) | (s[C_WBC_COUNT] < 4)
    return s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int)