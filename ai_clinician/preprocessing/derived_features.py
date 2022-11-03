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

def compute_sapsii(df):
    """ Calculate the SAPSII score provided the dataframe of raw patient features. """
    age_values = np.array([0, 7, 12, 15, 16, 18])
    hr_values = np.array([11, 2, 0, 4, 7])
    bp_values = np.array([13, 5, 0, 2])
    temp_values = np.array([0, 3])
    o2_values = np.array([11, 9, 6])
    output_values = np.array([11, 4, 0])
    bun_values = np.array([0, 6, 10])
    wbc_values = np.array([12, 0, 3])
    k_values = np.array([3, 0, 3])
    na_values = np.array([5, 0, 1])
    hco3_values = np.array([5, 3, 0])
    bili_values = np.array([0, 4, 9])
    gcs_values = np.array([26, 13, 7, 5, 0])
    
    sapsii = np.zeros((df.shape[0],1))
    
    cols = [
        C_AGE, C_HR, C_SYSBP, C_TEMP_C, C_PAO2_FIO2, C_OUTPUT_STEP, C_BUN,
        C_WBC_COUNT, C_POTASSIUM, C_SODIUM, C_HCO3, C_TOTAL_BILI, C_GCS
    ]
    tt = df[cols]
    
    age = np.array([ tt.iloc[:,0]<40, (tt.iloc[:,0]>=40)&(tt.iloc[:,0]<60), (tt.iloc[:,0]>=60)&(tt.iloc[:,0]<70), (tt.iloc[:,0]>=70)&(tt.iloc[:,0]<75), (tt.iloc[:,0]>=75)&(tt.iloc[:,0]<80), tt.iloc[:,0]>=80 ])
    hr = np.array([ tt.iloc[:,1]<40, (tt.iloc[:,1]>=40)&(tt.iloc[:,1]<70), (tt.iloc[:,1]>=70)&(tt.iloc[:,1]<120), (tt.iloc[:,1]>=120)&(tt.iloc[:,1]<160), tt.iloc[:,1]>=160 ])
    bp = np.array([ tt.iloc[:,2]<70, (tt.iloc[:,2]>=70)&(tt.iloc[:,2]<100), (tt.iloc[:,2]>=100)&(tt.iloc[:,2]<200), tt.iloc[:,2]>=200 ])
    temp = np.array([ tt.iloc[:,3]<39, tt.iloc[:,3]>=39 ])
    o2 = np.array([ tt.iloc[:,4]<100, (tt.iloc[:,4]>=100)&(tt.iloc[:,4]<200), tt.iloc[:,4]>=200 ])
    out = np.array([ tt.iloc[:,5]<500, (tt.iloc[:,5]>=500)&(tt.iloc[:,5]<1000), tt.iloc[:,5]>=1000 ])
    bun = np.array([ tt.iloc[:,6]<28, (tt.iloc[:,6]>=28)&(tt.iloc[:,6]<84), tt.iloc[:,6]>=84 ])
    wbc = np.array([ tt.iloc[:,7]<1, (tt.iloc[:,7]>=1)&(tt.iloc[:,7]<20), tt.iloc[:,7]>=20 ])
    k = np.array([ tt.iloc[:,8]<3, (tt.iloc[:,8]>=3)&(tt.iloc[:,8]<5), tt.iloc[:,8]>=5 ])
    na = np.array([ tt.iloc[:,9]<125, (tt.iloc[:,9]>=125)&(tt.iloc[:,9]<145), tt.iloc[:,9]>=145 ])
    hco3 = np.array([ tt.iloc[:,10]<15, (tt.iloc[:,10]>=15)&(tt.iloc[:,10]<20), tt.iloc[:,10]>=20 ])
    bili = np.array([ tt.iloc[:,11]<4, (tt.iloc[:,11]>=4)&(tt.iloc[:,11]<6), tt.iloc[:,11]>=6 ])
    gcs = np.array([ tt.iloc[:,12]<6, (tt.iloc[:,12]>=6)&(tt.iloc[:,12]<9), (tt.iloc[:,12]>=9)&(tt.iloc[:,12]<11), (tt.iloc[:,12]>=11)&(tt.iloc[:,12]<14), tt.iloc[:,12]>=14 ])
    
    for ii in range(df.shape[0]):
        sapsii[ii] = max(age_values[age[:,ii]], default=0) + max(hr_values[hr[:,ii]], default=0) + max(bp_values[bp[:,ii]], default=0) + max(temp_values[temp[:,ii]], default=0) + max(o2_values[o2[:,ii]]*df.loc[ii,C_MECHVENT], default=0) + max(output_values[out[:,ii]], default=0) + max(bun_values[bun[:,ii]], default=0) + max(wbc_values[wbc[:,ii]], default=0) + max(k_values[k[:,ii]], default=0) + max(na_values[na[:,ii]], default=0) + max(hco3_values[hco3[:,ii]], default=0) + max(bili_values[bili[:,ii]], default=0) + max(gcs_values[gcs[:,ii]], default=0)
    return sapsii.flatten()

def compute_oasis(df):
    """ Calculate the OASIS score provided the dataframe of raw patient features. """
    age_values = np.array([0, 3, 6, 9, 7])
    bp_values = np.array([4, 3, 2, 0, 3])
    gcs_values = np.array([10, 4, 3, 0])
    hr_values = np.array([4, 0, 1, 3, 6])
    rr_values = np.array([10, 1, 0, 1, 6, 9])
    temp_values = np.array([3, 4, 2, 2, 6])
    output_values = np.array([10, 5, 1, 0, 8])
    vent_value = 9
    
    oasis = np.zeros((df.shape[0],1))
    
    cols = [C_AGE, C_MEANBP, C_GCS, C_HR, C_RR, C_TEMP_C, C_OUTPUT_STEP]
    tt = df[cols]
    
    age = np.array([ tt.iloc[:,0]<24, (tt.iloc[:,0]>=24)&(tt.iloc[:,0]<=53), (tt.iloc[:,0]>53)&(tt.iloc[:,0]<=77), (tt.iloc[:,0]>77)&(tt.iloc[:,0]<=89), tt.iloc[:,0]>89 ])
    bp = np.array([ tt.iloc[:,1]<20.65, (tt.iloc[:,1]>=20.65)&(tt.iloc[:,1]<51), (tt.iloc[:,1]>=51)&(tt.iloc[:,1]<61.33), (tt.iloc[:,1]>=61.33)&(tt.iloc[:,1]<143.44), tt.iloc[:,1]>=143.44 ])
    gcs = np.array([ tt.iloc[:,2]<=7, (tt.iloc[:,2]>7)&(tt.iloc[:,1]<14), tt.iloc[:,1]==14, tt.iloc[:,1]>14 ])
    hr = np.array([ tt.iloc[:,3]<33, (tt.iloc[:,3]>=33)&(tt.iloc[:,3]<89), (tt.iloc[:,3]>=89)&(tt.iloc[:,3]<106), (tt.iloc[:,3]>=106)&(tt.iloc[:,3]<=125), tt.iloc[:,3]>125 ])
    rr = np.array([ tt.iloc[:,4]<6, (tt.iloc[:,4]>=6)&(tt.iloc[:,4]<13), (tt.iloc[:,4]>=13)&(tt.iloc[:,4]<22), (tt.iloc[:,4]>=22)&(tt.iloc[:,4]<30), (tt.iloc[:,4]>=30)&(tt.iloc[:,4]<44), tt.iloc[:,4]>=44 ])
    temp = np.array([ tt.iloc[:,5]<33.22, (tt.iloc[:,5]>=33.22)&(tt.iloc[:,5]<35.93), (tt.iloc[:,5]>=35.93)&(tt.iloc[:,5]<36.89), (tt.iloc[:,5]>=36.89)&(tt.iloc[:,5]<=39.88), tt.iloc[:,5]>39.88 ])
    out = np.array([ tt.iloc[:,6]<671.09, (tt.iloc[:,6]>=671.09)&(tt.iloc[:,6]<1427), (tt.iloc[:,6]>=1427)&(tt.iloc[:,6]<=2514), (tt.iloc[:,6]>2514)&(tt.iloc[:,6]<=6896), tt.iloc[:,6]>6896 ])
    vent = (vent_value*df[C_MECHVENT]).values
    
    for ii in range(df.shape[0]):
        oasis[ii] = max(age_values[age[:,ii]], default=0) + max(bp_values[bp[:,ii]], default=0) + max(gcs_values[gcs[:,ii]], default=0) + max(hr_values[hr[:,ii]], default=0) + max(rr_values[rr[:,ii]], default=0) + max(temp_values[temp[:,ii]], default=0) + max(output_values[out[:,ii]], default=0) + vent[ii]
        
    return oasis.flatten()
