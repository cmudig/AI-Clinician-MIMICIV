"""
Names for columns in the MIMIC-IV dataset.
"""
import pandas as pd
import numpy as np

# Demographics and timestamps
C_BLOC = 'bloc'
C_ICUSTAYID = 'icustayid'
C_CHARTTIME = 'charttime'
C_GENDER = 'gender'
C_AGE = 'age'
C_ELIXHAUSER = 'elixhauser'
C_RE_ADMISSION = 're_admission'
C_DIED_IN_HOSP = 'died_in_hosp'
C_DIED_WITHIN_48H_OF_OUT_TIME = 'died_within_48h_of_out_time'
C_MORTA_90 = 'morta_90'
C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH = 'delay_end_of_record_and_discharge_or_death'

# Chart events
C_HEIGHT = 'Height_cm'
C_WEIGHT = 'Weight_kg'
C_GCS = 'GCS'
C_RASS = 'RASS'
C_HR = 'HR'
C_SYSBP = 'SysBP'
C_MEANBP = 'MeanBP'
C_DIABP = 'DiaBP'
C_RR = 'RR'
C_SPO2 = 'SpO2'
C_TEMP_C = 'Temp_C'
C_TEMP_F = 'Temp_F'
C_CVP = 'CVP'
C_PAPSYS = 'PAPsys'
C_PAPMEAN = 'PAPmean'
C_PAPDIA = 'PAPdia'
C_CI = 'CI'
C_SVR = 'SVR'
C_INTERFACE = 'Interface'
C_FIO2_100 = 'FiO2_100'
C_FIO2_1 = 'FiO2_1'
C_O2FLOW = 'O2flow'
C_PEEP = 'PEEP'
C_TIDALVOLUME = 'TidalVolume'
C_MINUTEVENTIL = 'MinuteVentil'
C_PAWMEAN = 'PAWmean'
C_PAWPEAK = 'PAWpeak'
C_PAWPLATEAU = 'PAWplateau'

# Labs
C_POTASSIUM = 'Potassium'
C_SODIUM = 'Sodium'
C_CHLORIDE = 'Chloride'
C_GLUCOSE = 'Glucose'
C_BUN = 'BUN'
C_CREATININE = 'Creatinine'
C_MAGNESIUM = 'Magnesium'
C_CALCIUM = 'Calcium'
C_IONISED_CA = 'Ionised_Ca'
C_CO2_MEQL = 'CO2_mEqL'
C_SGOT = 'SGOT'
C_SGPT = 'SGPT'
C_TOTAL_BILI = 'Total_bili'
C_DIRECT_BILI = 'Direct_bili'
C_TOTAL_PROTEIN = 'Total_protein'
C_ALBUMIN = 'Albumin'
C_TROPONIN = 'Troponin'
C_CRP = 'CRP'
C_HB = 'Hb'
C_HT = 'Ht'
C_RBC_COUNT = 'RBC_count'
C_WBC_COUNT = 'WBC_count'
C_PLATELETS_COUNT = 'Platelets_count'
C_PTT = 'PTT'
C_PT = 'PT'
C_ACT = 'ACT'
C_INR = 'INR'
C_ARTERIAL_PH = 'Arterial_pH'
C_PAO2 = 'paO2'
C_PACO2 = 'paCO2'
C_ARTERIAL_BE = 'Arterial_BE'
C_ARTERIAL_LACTATE = 'Arterial_lactate'
C_HCO3 = 'HCO3'
C_ETCO2 = 'ETCO2'
C_SVO2 = 'SvO2'

# Ventilation
C_MECHVENT = 'mechvent'
C_EXTUBATED = 'extubated'

# Computed
C_SHOCK_INDEX = 'Shock_Index'
C_PAO2_FIO2 = 'PaO2_FiO2'

# Vasopressors, input/output
C_MEDIAN_DOSE_VASO = 'median_dose_vaso'
C_MAX_DOSE_VASO = 'max_dose_vaso'
C_INPUT_TOTAL = 'input_total'
C_INPUT_STEP = 'input_step'
C_OUTPUT_TOTAL = 'output_total'
C_OUTPUT_STEP = 'output_step'
C_CUMULATED_BALANCE = 'cumulated_balance'

######### Onset data

C_ONSET_TIME = "onset_time"
C_FIRST_TIMESTEP = "first_timestep"
C_LAST_TIMESTEP = "last_timestep"

######### Derived dataframes

C_BLOC = "bloc"
C_TIMESTEP = "timestep"
C_BIN_INDEX = "bin_index"
C_SOFA = "SOFA"
C_SIRS = "SIRS"
C_LAST_VASO = "last_vaso"
C_LAST_SOFA = "last_SOFA"
C_NUM_BLOCS = "num_blocs"
C_MAX_SOFA = "max_SOFA"
C_MAX_SIRS = "max_SIRS"

######### For raw data

C_HADM_ID = "hadm_id"
C_SUBJECT_ID = "subject_id"
C_STARTDATE = "startdate"
C_ENDDATE = "enddate"
C_STARTTIME = "starttime"
C_ENDTIME = "endtime"
C_CHARTTIME = "charttime"
C_CHARTDATE = "chartdate"
C_ITEMID = "itemid"
C_ADMITTIME = "admittime"
C_DISCHTIME = "dischtime"
C_ADM_ORDER = "adm_order"
C_UNIT = "unit"
C_INTIME = "intime"
C_OUTTIME = "outtime"
C_LOS = "los"
C_DOB = "dob"
C_DOD = "dod"
C_EXPIRE_FLAG = "expire_flag"
C_MORTA_HOSP = "morta_hosp"
C_VALUE = "value"
C_VALUENUM = "valuenum"
C_SELFEXTUBATED = "selfextubated"
C_INPUT_PREADM = "input_preadm"
C_AMOUNT = "amount"
C_RATE = "rate"
C_TEV = "tev"
C_RATESTD = "ratestd"
C_DATEDIFF_MINUTES = "datediff_minutes"

# Additional computed fields on raw data
C_NORM_INFUSION_RATE = "norm_infusion_rate"

RAW_DATA_COLUMNS = {
    "abx": [C_HADM_ID, C_ICUSTAYID, C_STARTDATE, C_ENDDATE],
    "culture": [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_CHARTTIME, C_ITEMID],
    "microbio": [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_CHARTTIME, C_CHARTDATE],
    "demog": [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_ADMITTIME, C_DISCHTIME,
              C_ADM_ORDER, C_UNIT, C_INTIME, C_OUTTIME, C_LOS,
              C_AGE, C_DOB, C_DOD, C_EXPIRE_FLAG, C_GENDER,
              C_MORTA_HOSP, C_MORTA_90, C_ELIXHAUSER],
    "ce": [C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM],
    "labs_ce": [C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM],
    "labs_le": [C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM],
    "mechvent": [C_ICUSTAYID, C_CHARTTIME, C_MECHVENT, C_EXTUBATED, C_SELFEXTUBATED],
    "mechvent_pe": [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_STARTTIME,
        C_ENDTIME, C_MECHVENT, C_EXTUBATED, C_SELFEXTUBATED,
        C_ITEMID, C_VALUE],
    "preadm_fluid": [C_ICUSTAYID, C_INPUT_PREADM],
    "fluid_mv": [C_ICUSTAYID, C_STARTTIME, C_ENDTIME, C_ITEMID, C_AMOUNT,
        C_RATE, C_TEV],
    "vaso_mv": [C_ICUSTAYID, C_ITEMID, C_STARTTIME, C_ENDTIME, C_RATESTD],
    "preadm_uo": [C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUE, C_DATEDIFF_MINUTES],
    "uo": [C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUE],
}

DTYPE_SPEC = {
    C_HADM_ID: pd.Int64Dtype(),
    C_SUBJECT_ID: pd.Int64Dtype(),
    C_TIMESTEP: pd.Int64Dtype(),
    C_ICUSTAYID: np.int64, # pd.Int64Dtype(),
    C_ITEMID: pd.Int64Dtype(),
}

########## Itemid mappings

REF_LABS = [
    [829, 1535, 227442, 227464, 3792, 50971, 50822],
    [837, 220645, 4194, 3725, 3803, 226534, 1536, 4195, 3726, 50983, 50824],
    [788, 220602, 1523, 4193, 3724, 226536, 3747, 50902, 50806],
    [225664, 807, 811, 1529, 220621, 226537, 3744, 50809, 50931],
    [781, 1162, 225624, 3737, 51006, 52647],
    [791, 1525, 220615, 3750, 50912, 51081],
    [821, 1532, 220635, 50960],
    [786, 225625, 1522, 3746, 50893, 51624],
    [816, 225667, 3766, 50808],
    [777, 787, 50804],
    [770, 3801, 50878, 220587],
    [769, 3802, 50861],
    [1538, 848, 225690, 51464, 50885],
    [803, 1527, 225651, 50883],
    [3807, 1539, 849, 50976],
    [772, 1521, 227456, 3727, 50862],
    [227429, 851, 51002, 51003],
    [227444, 50889],
    [814, 220228, 50811, 51222],
    [813, 220545, 3761, 226540, 51221, 50810],
    [4197, 3799, 51279],
    [1127, 1542, 220546, 4200, 3834, 51300, 51301],
    [828, 227457, 3789, 51265],
    [825, 1533, 227466, 3796, 51275, 52923, 52165, 52166, 52167],
    [824, 1286, 51274, 227465],
    [1671, 1520, 768, 220507],
    [815, 1530, 227467, 51237],
    [780, 1126, 3839, 4753, 50820],
    [779, 490, 3785, 3838, 3837, 50821, 220224, 226063, 226770, 227039],
    [778, 3784, 3836, 3835, 50818, 220235, 226062, 227036],
    [776, 224828, 3736, 4196, 3740, 74, 50802],
    [225668, 1531, 50813],
    [227443, 50882, 50803],
    [1817, 228640],
    [823, 227686, 223772]
]

REF_VITALS = [
    [226707, 226730],
    [581, 580, 224639, 226512],
    [198],
    [228096],
    [211, 220045],
    [220179, 225309, 6701, 6, 227243, 224167, 51, 455],
    [220181, 220052, 225312, 224322, 6702, 443, 52, 456],
    [8368, 8441, 225310, 8555, 8440],
    [220210, 3337, 224422, 618, 3603, 615],
    [220277, 646, 834],
    [3655, 223762],
    [223761, 678],
    [220074, 113],
    [492, 220059],
    [491, 220061],
    [8448, 220060],
    [116, 1372, 1366, 228368, 228177],
    [626],
    [467, 226732],
    [223835, 3420, 160, 727],
    [190],
    [470, 471, 223834, 227287, 194, 224691],
    [220339, 506, 505, 224700],
    [224686, 224684, 684, 224421, 3083, 2566, 654, 3050, 681, 2311],
    [224687, 450, 448, 445],
    [224697, 444],
    [224695, 535],
    [224696, 543]
]

CHART_FIELD_NAMES = [
    C_HEIGHT, C_WEIGHT, C_GCS,
    C_RASS, C_HR, C_SYSBP,
    C_MEANBP, C_DIABP, C_RR,
    C_SPO2, C_TEMP_C, C_TEMP_F,
    C_CVP, C_PAPSYS, C_PAPMEAN,
    C_PAPDIA, C_CI, C_SVR,
    C_INTERFACE, C_FIO2_100, C_FIO2_1,
    C_O2FLOW, C_PEEP, C_TIDALVOLUME,
    C_MINUTEVENTIL, C_PAWMEAN, C_PAWPEAK,
    C_PAWPLATEAU
]

LAB_FIELD_NAMES = [
    C_POTASSIUM, C_SODIUM,
	C_CHLORIDE, C_GLUCOSE, C_BUN,
	C_CREATININE, C_MAGNESIUM, C_CALCIUM,
	C_IONISED_CA, C_CO2_MEQL, C_SGOT,
	C_SGPT, C_TOTAL_BILI, C_DIRECT_BILI,
	C_TOTAL_PROTEIN, C_ALBUMIN, C_TROPONIN,
	C_CRP, C_HB, C_HT,
	C_RBC_COUNT, C_WBC_COUNT, C_PLATELETS_COUNT,
	C_PTT, C_PT, C_ACT,
	C_INR, C_ARTERIAL_PH, C_PAO2,
	C_PACO2, C_ARTERIAL_BE, C_ARTERIAL_LACTATE,
	C_HCO3, C_ETCO2, C_SVO2
]

VENT_FIELD_NAMES = [
	C_MECHVENT, C_EXTUBATED
]

COMPUTED_FIELD_NAMES = [
    C_SHOCK_INDEX,
    C_PAO2_FIO2
]

IO_FIELD_NAMES = [
    C_MEDIAN_DOSE_VASO,
    C_MAX_DOSE_VASO,
    C_INPUT_TOTAL,
    C_INPUT_STEP,
    C_OUTPUT_TOTAL,
    C_OUTPUT_STEP,
    C_CUMULATED_BALANCE
]

DEMOGRAPHICS_FIELD_NAMES = [
    C_GENDER,
    C_AGE,
    C_ELIXHAUSER,
    C_RE_ADMISSION,
    C_DIED_IN_HOSP,
    C_DIED_WITHIN_48H_OF_OUT_TIME,
    C_MORTA_90,
    C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH,
]

SAH_FIELD_NAMES = CHART_FIELD_NAMES + LAB_FIELD_NAMES + VENT_FIELD_NAMES

SAH_HOLD_DURATION = {f: v for f, v in zip(SAH_FIELD_NAMES, [
    168, 72, 6,
	6, 2, 2,
	2, 2, 2,
	2, 6, 6,
	2, 2, 2,
	2, 2, 2,
	24, 12, 12,
	12, 6, 6,
	6, 6, 6,
	6, 144, 14,
	14, 14, 28,
	28, 28, 28,
	28, 28, 28,
	28, 28, 28,
	28, 28, 28,
	28, 14, 28,
	28, 28, 28,
	28, 28, 28,
	28, 8, 8,
	8, 8, 8,
	8, 8, 8,
	6, 6, 
])}