import os
import pandas as pd
import numpy as np
import time
import json
import glob
import itertools
import re
import tqdm

from preprocessing.utils import reverse_readline

VALUE_ROW = "value_row"
VALUE_COL = "value_col"
CHANGE_TYPE = "change_type"
METADATA = "metadata"
REFERENCE_ROW = "reference_row"
REFERENCE_COL = "reference_col"

WRITE_COLUMNS = [VALUE_ROW, VALUE_COL, CHANGE_TYPE, METADATA, REFERENCE_ROW, REFERENCE_COL]

QUERY_ROW = "query_row"
QUERY_COL = "query_col"
SOURCE_IDX = "source_idx"
TARGET_IDX = "target_idx"

def _make_iterable(value):
    if isinstance(value, (pd.Series, pd.Index)):
        return value.values
    elif isinstance(value, np.ndarray):
        return value
    elif isinstance(value, (list, set)):
        return list(value)
    else:
        return [value]

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
            file.write(",".join(WRITE_COLUMNS) + "\n")
            
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
        df = pd.DataFrame(self.buffer, columns=WRITE_COLUMNS)
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
        
        row = _make_iterable(row)
        col = self._get_col_indexes(_make_iterable(col))
        reference_row = _make_iterable(reference_row)
        reference_col = self._get_col_indexes(_make_iterable(reference_col))
        metadata = _make_iterable(metadata)
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
        
        records = np.empty((len(row) * len(col), len(WRITE_COLUMNS)), dtype='object')
        for i, icol in enumerate(col):
            for j, record_col in enumerate(WRITE_COLUMNS):
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


class ProvenanceReader:
    """
    Reads a set of files written by a ProvenanceWriter, and supports certain
    optimized queries on them without loading all records into memory.
    """
    def __init__(self, file_location):
        self.file_location = file_location
        with open(os.path.join(self.file_location, "metadata.json"), "r") as file:
            metadata = json.load(file)
        self.change_types = metadata["change_types"]
        self._change_types_index = {c: i for i, c in enumerate(self.change_types)}
        self.record_counts = metadata["record_counts"]
        self.record_files = sorted(self.record_counts.keys(),
                                   key=lambda fname: int(re.search('values_(\d+).csv', fname).group(1)))
        self.col_names = metadata["col_names"]
        self._col_names_index = {c: i for i, c in enumerate(self.col_names)}
        self.parent_info = metadata.get("parent_info", None)
        
    def _make_query_item(self, row=None, col=None):
        assert not (row is None and col is None)
        if row is not None and col is not None:
            return np.base_repr(row, 36) + ',' + str(self._col_names_index[col])
        elif row is not None:
            return np.base_repr(row, 36)
        elif col is not None:
            return str(self._col_names_index[col])
        
    def _searchable_string_fn(self, row=None, col=None):
        assert not (row is None and col is None)
        if row is not None and col is not None:
            return lambda line: ','.join(line.split(',')[:2])
        elif row is not None:
            return lambda line: line.split(',')[0]
        elif col is not None:
            return lambda line: line.split(',')[1]
        
    def _convert_record(self, line):
        comps = line.strip().split(',')
        return {
            VALUE_ROW: int(comps[0], 36),
            VALUE_COL: self.col_names[int(comps[1])] if comps[1] else None,
            CHANGE_TYPE: self.change_types[int(comps[2])] if comps[2] else None,
            METADATA: comps[3],
            REFERENCE_ROW: int(comps[4], 36) if comps[4] else None,
            REFERENCE_COL: self.col_names[int(comps[5])] if comps[5] else None
        }
        
    def dependencies(self, row=None, col=None, start_index=None, show_progress=True):
        """
        Retrieves all records in a dependency tree of the given rows and columns.
        Row and col can be None, single values or array-like. If they are
        both array-like, then all combinations of the rows and cols will be 
        retrieved. If one is None, then all values for that field will be
        retrieved. Both row and col cannot be None.
        
        Returns: a Pandas dataframe containing the records that were found
            along the dependency tree for the given rows and cols; and a
            dictionary of paths, keyed by (row, col) tuples pointing to
            individual data items whose dependencies were retrieved. The values
            are lists of indexes pointing to rows in the first returned value.
        """
        
        # First make a set of query items that will be used to filter lines in
        # each record file using simple string matching
        query_fn = self._searchable_string_fn(row, col)
        full_query_fn = self._searchable_string_fn(1, 1)
        row = _make_iterable(row)
        col = _make_iterable(col)
        queries = {
            self._make_query_item(r, c): (r, c)
            for r, c in itertools.product(row, col)
        }
        full_queries = {}
        
        # Iterate over records backwards, adding to search_items as we find
        # relevant ones
        results = []
        dependency_paths = {}
        
        records = self.record_files
        if start_index is not None:
            records = records[:-start_index]
        iterable = tqdm.tqdm(reversed(records), total=len(records)) if show_progress else reversed(records)
        for filename in iterable:
            count = self.record_counts[filename]
            pos = 0
            for line in reverse_readline(os.path.join(self.file_location, filename)):
                if not line.strip(): continue
                q = query_fn(line)
                parents = None
                item = None
                
                if q in queries:
                    item = self._convert_record(line)
                    parents = [(item[VALUE_ROW], item[VALUE_COL])]
                elif (full_q := full_query_fn(line)) in full_queries:
                    item = self._convert_record(line)
                    parents = set([x for x in full_queries[full_q]])
                    
                if item and parents:
                    ref_row = item[REFERENCE_ROW] if item[REFERENCE_ROW] is not None else item[VALUE_ROW]
                    ref_col = item[REFERENCE_COL] if item[REFERENCE_COL] is not None else item[VALUE_COL]
                    ref_query = self._make_query_item(ref_row, ref_col)
                    
                    # Add parents to full_queries
                    for query_item in parents:
                        full_queries.setdefault(ref_query, set())
                        full_queries[ref_query].add(query_item)
                        dependency_paths.setdefault(query_item, []).append(len(results))
                    results.append(item)
                    
                pos += 1
                if pos == count: break
        
        return pd.DataFrame(results), {k: list(reversed(path)) for k, path in dependency_paths.items()}
    
    def descendants(self, row=None, col=None):
        """
        Retrieves all records that have been generated based on the given rows
        and columns. Row and col can be None, single values or array-like. If
        they are both array-like, then all combinations of the rows and cols
        will be retrieved. If one is None, then all values for that field will
        be retrieved. Both row and col cannot be None.
        
        Returns: a Pandas dataframe containing the records that were found
            along the dependency tree for the given rows and cols; and an
            adjacency list in the form of a Pandas dataframe with four columns,
            'source_idx', 'target_idx', 'query_row', and 'query_col'. The query
            row and column indicate which row and column from the input
            arguments that edge belongs to, while source_idx and target_idx are
            indexes into the first returned dataframe.
        """
        pass
        
    def list_change_types(self):
        """
        Returns a list of strings corresponding to the different change types
        used in the provenance records.
        """
        pass
    
    def change_type(self, change_type):
        """
        Retrieves all records that involve the given change type.
        
        Returns: a Pandas dataframe containing all records that were found with
            the given change type.
        """
        pass