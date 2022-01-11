import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from .columns import *
from tqdm import tqdm
from sklearn.impute import KNNImputer
from sklearn.metrics import pairwise_distances

# fill-in missing ICUSTAY IDs in bacterio and abx. We will look at their subject ID
# and find a matching ICU stay ID such that the event takes place
# within the admission
def impute_icustay_ids(demog, target, window=48 * 3600):
    """
    Finds an ICU stay ID from the demog table such that the 
    subject ID matches, and the admission in and out times wrap
    around the given chart time. If subject_id_col is None,
    uses hadm ID only.
    """
    filtered_demog = demog[[C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_INTIME, C_OUTTIME]]
    
    if C_SUBJECT_ID in target.columns:
        filtered_demog = filtered_demog[filtered_demog[C_SUBJECT_ID].isin(target[C_SUBJECT_ID])]
        subject_id_groups = filtered_demog.groupby(C_SUBJECT_ID).groups
        hadm_groups = filtered_demog.groupby(C_HADM_ID).groups

        def impute(row):
            same_subj = subject_id_groups.get(int(row[C_SUBJECT_ID]), [])
            if len(same_subj) >= 1:
                matching_rows = demog.iloc[same_subj]
                matching_rows = matching_rows[(matching_rows[C_INTIME] <= row[C_CHARTTIME] + window) &
                                              (matching_rows[C_OUTTIME] >= row[C_CHARTTIME] - window)]
                if len(matching_rows) > 0:
                    return matching_rows.iloc[0][C_ICUSTAYID]
            
            # Now check hadm ID and just grab the first one
            if not pd.isna(row[C_HADM_ID]):
                same_hadm = hadm_groups.get(int(row[C_HADM_ID]), [])
                if len(same_hadm) == 1:
                    return demog.iloc[same_hadm[0]][C_ICUSTAYID]
            return None
        return target.progress_apply(impute, axis=1)
    else:
        filtered_demog = filtered_demog[filtered_demog[C_HADM_ID].isin(target[C_HADM_ID])]
        hadm_groups = filtered_demog.groupby(C_HADM_ID).groups

        def impute(row):
            # Just check hadm ID
            if not pd.isna(row[C_HADM_ID]):
                same_hadm = hadm_groups.get(int(row[C_HADM_ID]), [])
                if len(same_hadm) == 1:
                    return demog.iloc[same_hadm[0]][C_ICUSTAYID]
            return None
        return target.progress_apply(impute, axis=1)

LOG_OUTLIERS = True

def is_outlier(col, lower=None, upper=None):
    if lower is not None and upper is not None:
        result = (col < lower) | (col > upper)
    elif lower is not None:
        result = col < lower
    elif upper is not None:
        result = col > upper
    else:
        result = np.zeros(len(col))
    if LOG_OUTLIERS: print('(' + str(result.sum()) + ' outliers) ', end='')
    return result

def fill_outliers(df, spec, provenance=None):
    """
    Remove outliers according to a specification. Each key in the
    spec dictionary should correspond to a column in df, and the value
    should be a tuple (lower, upper) indicating the lower and upper limits
    for allowed values in that column. Other values will be set to pd.NA.
    
    A modified copy of the dataframe is returned.
    """
    copy_df = df.copy()
    for col, (min_val, max_val) in spec.items():
        print('filtering', col, end=' ')
        outliers = is_outlier(copy_df[col], min_val, max_val)
        if provenance:
            provenance.record("outlier", row=copy_df.loc[outliers, C_ICUSTAYID], col=col)
        copy_df.loc[outliers, col] = pd.NA
        print('')
    return copy_df

def sample_and_hold(stay_ids, chart_times, series, hold_time, provenance=None, col_name=None):
    """
    Performs a sample-and-hold to fill in missing values when prior
    values are present.
    
    Args:
        stay_ids: A pandas Series containing ICU stay IDs
        chart_times: A pandas Series matching the length of stay_ids
            and containing chart timestamps for each event
        series: A pandas Series to be filled in
        hold_time: The number of hours to hold the previous value
        provenance: An optional ProvenanceWriter to write the source values for
            each row
        col_name: Column name to use for provenance
        
    Returns: an updated version of series with missing values filled in
    """
    old_index = series.index
    
    last_stay_id = None
    last_chart_time = None
    last_value = None
    last_index = None
    assert len(stay_ids) == len(chart_times) == len(series), "All series must have same length"
    
    stay_ids = stay_ids.values # using numpy representations for speed
    chart_times = chart_times.values
    series = series.values
    index_values = old_index.values
    new_series = series.copy()
    
    prov_indexes = np.zeros(len(series), dtype=int)
    prov_sources = np.zeros(len(series), dtype=int)
    prov_position = 0
    
    for i in tqdm(range(len(series))):
        stay_id = stay_ids[i]
        chart_time = chart_times[i]
        val = series[i]
        index_val = index_values[i]
        
        if last_stay_id is None or stay_id != last_stay_id:
            # Update the current stay ID we are looking at
            last_stay_id = stay_id
            last_chart_time = chart_time
            last_value = val
            last_index = index_val
        elif not pd.isna(val):
            # Store this value
            last_chart_time = chart_time
            last_value = val
            last_index = index_val
        elif pd.isna(val) and chart_time - last_chart_time <= hold_time * 3600:
            # Fill in the last value for this timestamp
            new_series[i] = last_value
            prov_indexes[prov_position] = index_val
            prov_sources[prov_position] = last_index
            prov_position += 1
            
    if provenance:
        provenance.record("sample and hold",
                          row=prov_indexes[:prov_position],
                          col=col_name,
                          reference_row=prov_sources[:prov_position])
    return pd.Series(new_series, index=old_index)

# Estimate FiO2 with use of various interfaces.
def fill_stepwise(x_values, lt_values, gt_values=None):
    """
    Estimates y values using the given x values in a stepwise fashion.
    
    
    Args:
        x_values: A Series containing x values.
        lt_values: A list of tuples (x, y), where x defines
            an input value and y defines the corresponding
            output value. These values will be sorted in reverse order
            by x and enumerated such that every value between x and
            the next less value x' will be set to y.
        gt_values: A similar list to lt_values, except these 
            will be run in ascending order and enumerated so that
            every value between x and the next GREATER value x'
            will be set to y. If provided, this runs before lt_values.
    Returns: a pandas Series containing y values.
    """
    result = np.zeros(len(x_values))
    if gt_values:
        for x, y in sorted(gt_values, key=lambda x: x[0]):
            result[x_values >= x] = y
    for x, y in sorted(lt_values, key=lambda x: x[0], reverse=True):
        result[x_values <= x] = y
    return pd.Series(np.where(result == 0, pd.NA, result), index=x_values.index)

def fixgaps(x):
# FIXGAPS Linearly interpolates gaps in a time series
# YOUT=FIXGAPS(YIN) linearly interpolates over NaN
# in the input time series (may be complex), but ignores
# trailing and leading NaN.
#

# R. Pawlowicz 6/Nov/99

    y = x.copy()

    bd = pd.isna(x)
    gd = x.index[~bd]

    bd.loc[:gd.min()] = False
    bd.loc[gd.max() + 1:] = False
    y[bd] = interp1d(gd,x.loc[gd])(x.index[bd].tolist())
    return y

def nearest_neighbor_impute(X, metric='nan_euclidean', provenance=None, n_jobs=None, return_distances=False, exclude_from_provenance=None):
    Xc = X.copy()
    X_vals = X.values
    if return_distances:
        D = np.zeros(X_vals.shape)
    shown_warning = False
    for col, col_name in enumerate(X.columns):
        if pd.isna(X[col_name]).sum() in (len(X), 0):
            continue
        non_nan_positions = (~pd.isna(X[col_name])).values
        non_nan_indexes = np.where(non_nan_positions)[0]
        train_data = np.delete(X_vals[non_nan_positions], col, axis=1)

        col_nan_positions = pd.isna(X[col_name]).values
        test_distances = pairwise_distances(np.delete(X_vals[col_nan_positions], col, axis=1),
                                            train_data,
                                            metric=metric,
                                            n_jobs=n_jobs)
        
        neighbors = np.where(np.isnan(test_distances), np.inf, test_distances).argmin(axis=1)
        distances = test_distances[np.arange(len(test_distances)), neighbors]
        if np.isnan(distances).sum() > 0 and not shown_warning:
            print("Warning: some points had all infinite distances")
            shown_warning = True
        # distances, neighbors = nn.kneighbors(test_distances, n_neighbors=1)
        neighbors = non_nan_indexes[neighbors]
        print(col_nan_positions[:10])
        Xc.loc[col_nan_positions, col_name] = X[col_name].values[neighbors]
        if provenance and (exclude_from_provenance is None or col_name not in exclude_from_provenance):
            provenance.record("KNN imputation", row=col_nan_positions, col=col_name, metadata=distances, reference_row=neighbors)
        if return_distances:
            D[col_nan_positions, col] = distances
    if return_distances: return Xc, D
    return Xc

def knn_impute(data, batch_size=10000, na_threshold=1, provenance=None):
    """
    Perform KNN imputation on the given dataframe. Returns a copy of the dataframe
    with missing values filled in. Only performs imputation on the fields that
    have NA values with a rate of less than na_threshold.
    """

    data = data.copy()    

    bar = tqdm(range(0, len(data), batch_size))
    for start_idx in bar:
        end_idx = min(len(data), start_idx + batch_size) - 1
        # missNotAll columns have at least one non-NaN value and will be used in 
        # training
        missNotAll = pd.isna(data.loc[start_idx:end_idx]).sum(axis=0) != (end_idx - start_idx + 1)
        # indexes_whole are indexes in the full dataframe of columns with nan 
        # rate less than na_threshold
        indexes_whole = pd.isna(data.loc[start_idx:end_idx]).mean(axis=0) <= na_threshold
        # indexes_sub are indexes in the missNotAll columns with nan rate less
        # than na_threshold
        indexes_sub = [i for i, col in enumerate(data.columns[missNotAll]) if pd.isna(data.loc[start_idx:end_idx, col]).mean() <= na_threshold]
        # exclude_col_names are the names of the columns with nan rate more than 
        # na_threshold, which should be excluded
        exclude_col_names = set([col for col in data.columns[missNotAll] if pd.isna(data.loc[start_idx:end_idx, col]).mean() > na_threshold])
        
        bar.set_description('excluding {}/{} columns'.format(len(exclude_col_names), len(data.columns)))
        data.loc[start_idx:end_idx, indexes_whole] = nearest_neighbor_impute(data.loc[start_idx:end_idx, missNotAll],
                                                                             provenance=provenance,
                                                                             n_jobs=4,
                                                                             exclude_from_provenance=exclude_col_names)[:,indexes_sub]

    return data

