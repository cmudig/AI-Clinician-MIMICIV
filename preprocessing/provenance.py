import os
import pandas as pd
import numpy as np
import time
import json
import glob

VALUE_ROW = "value_row"
VALUE_COL = "value_col"
CHANGE_TYPE = "change_type"
METADATA = "metadata"
REFERENCE_ROW = "reference_row"
REFERENCE_COL = "reference_col"

ALL_COLUMNS = [VALUE_ROW, VALUE_COL, CHANGE_TYPE, METADATA, REFERENCE_ROW, REFERENCE_COL]
    
class ProvenanceWriter:
    """
    This class handles writing a record of which values were changed and how.
    """
    
    def __init__(self, file_location, write_frequency=500000, records_per_file=5000000, verbose=False):
        self.file_location = file_location
        self.buffer = None
        self.write_frequency = write_frequency
        self.records_per_file = records_per_file
        self.change_types = {} # mapping of change type to index
        self.col_names = {}
        self.record_time = 0.0
        self.write_time = 0.0
        self.verbose = verbose
        self.start_time = time.time()
                
        if not os.path.exists(self.file_location):
            os.mkdir(self.file_location)
        for path in glob.glob(os.path.join(self.file_location, 'values_*.csv')):
            os.remove(path)

        self.record_counts = {}
        self.current_file_number = 0
        self._init_file(0)
              
    def log_times(self):
        b = time.time()
        prov_time = self.record_time + self.write_time
        run_time = b - self.start_time 
        print("Time consumption: {:.3f} s recording + {:.3f} s writing / {:.3f} s total script runtime ({:.2f}%)".format(self.record_time, self.write_time, run_time, prov_time / run_time * 100))
          
    def _init_file(self, file_number):
        self.record_counts[file_number] = 0
        with open(self._get_file_path(file_number), "w") as file:
            file.write(",".join(ALL_COLUMNS) + "\n")
            
    def _get_file_path(self, file_number):
        return os.path.join(self.file_location, "values_{}.csv".format(file_number))

    def flush(self):
        """
        Writes the current contents to file.
        """
        if self.buffer is None: return
        if self.verbose:
            print("Writing {} provenance records to file".format(len(self.buffer)))
        a = time.time()
        df = pd.DataFrame(self.buffer, columns=ALL_COLUMNS)
        df[CHANGE_TYPE] = df[CHANGE_TYPE].astype(pd.Int64Dtype())
        
        has_row = ~pd.isna(df[VALUE_ROW])
        df.loc[has_row, VALUE_ROW] = df.loc[has_row, VALUE_ROW].apply(lambda x: np.base_repr(x, base=36))
        has_row = ~pd.isna(df[REFERENCE_ROW])
        df.loc[has_row, REFERENCE_ROW] = df.loc[has_row, REFERENCE_ROW].apply(lambda x: np.base_repr(x, base=36))
        
        df[VALUE_COL] = df[VALUE_COL].astype(pd.Int64Dtype())
        df[REFERENCE_COL] = df[REFERENCE_COL].astype(pd.Int64Dtype())
        
        space_left = self.records_per_file - self.record_counts[self.current_file_number]
        if space_left >= 0:
            len_to_write = min(space_left, len(df))
            with open(self._get_file_path(self.current_file_number), "a") as file:
                file.write(df.head(len_to_write).to_csv(header=False, index=False, float_format='%.4f').rstrip() + '\n')
            self.record_counts[self.current_file_number] += len_to_write
        if space_left < len(df):
            self.current_file_number += 1
            self._init_file(self.current_file_number)
            len_to_write = len(df) - space_left
            with open(self._get_file_path(self.current_file_number), "a") as file:
                file.write(df.tail(len_to_write).to_csv(header=False, index=False, float_format='%.4f').rstrip() + '\n')
            self.record_counts[self.current_file_number] += len_to_write
            
        self.buffer = None
        b = time.time()
        self.write_time += b - a
        if self.verbose:
            self.log_times()
        
    def close(self, additional_data=None):
        """
        Finishes writing all records and writes out metadata.
        """
        self.flush()
        meta = {
            "change_types": [c for c, _ in sorted(self.change_types.items(), key=lambda x: x[1])],
            "col_names": [c for c, _ in sorted(self.col_names.items(), key=lambda x: x[1])],
            "record_counts": {"values_{}.csv".format(i): v for i, v in self.record_counts.items()}
        }
        if additional_data:
            meta["parent_info"] = additional_data
        with open(os.path.join(self.file_location, "metadata.json"), "w") as file:
            json.dump(meta, file)
           
        if self.verbose:
            self.log_times()
        
    def _make_iterable(self, value):
        if isinstance(value, (pd.Series, pd.Index)):
            return value.values
        elif isinstance(value, np.ndarray):
            return value
        elif isinstance(value, (list, set)):
            return list(value)
        else:
            return [value]

    def _get_col_indexes(self, col_names):
        return [self.col_names.setdefault(c, len(self.col_names)) if c is not None else None for c in col_names]
        
    def record(self, change_type, row=None, col=None, metadata=None, reference_row=None, reference_col=None):
        """
        Records a change. If multiple values are passed for row and/or
        col, all pairs of the rows and columns will be recorded. If
        multiple reference_rows, metadata, and reference_cols are provided, they
        will be used one-to-one with rows and cols (not all pairs).
        """
        a = time.time()
        
        row = self._make_iterable(row)
        col = self._get_col_indexes(self._make_iterable(col))
        reference_row = self._make_iterable(reference_row)
        reference_col = self._get_col_indexes(self._make_iterable(reference_col))
        metadata = self._make_iterable(metadata)
        assert len(reference_row) in (len(row), 1), "Reference row must either be length {} or 1".format(len(row))
        assert len(reference_col) in (len(col), 1), "Reference col must either be length {} or 1".format(len(col))
        assert len(metadata) in (len(row), 1), "Metadata must be either length {} or 1".format(len(row))
        change_type = self.change_types.setdefault(change_type, len(self.change_types))
                
        record_values = {
            VALUE_ROW: row,
            VALUE_COL: col,
            CHANGE_TYPE: change_type,
            METADATA: metadata,
            REFERENCE_ROW: reference_row,
            REFERENCE_COL: reference_col
        }
        
        records = np.empty((len(row) * len(col), len(ALL_COLUMNS)), dtype='object')
        for i, icol in enumerate(col):
            for j, record_col in enumerate(ALL_COLUMNS):
                if record_col == VALUE_COL:
                    records[i * len(row):(i + 1) * len(row),j] = icol
                else:
                    records[i * len(row):(i + 1) * len(row),j] = record_values[record_col]

        if self.buffer is None:
            self.buffer = records
        else:
            self.buffer = np.concatenate([self.buffer, records])
            
        b = time.time()
        self.record_time += b - a
        
        if len(self.buffer) >= self.write_frequency:
            self.flush()
