import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
from ai_clinician.preprocessing.columns import C_ICUSTAYID, DTYPE_SPEC, STAY_ID_OPTIONAL_DTYPE_SPEC

def load_csv(*file_paths, null_icustayid=False, **kwargs):
    """
    Attempts to load a data CSV from the file paths given, and returns the first
    one whose file path exists.
    """
    for path in file_paths:
        if os.path.exists(path):
            spec = DTYPE_SPEC
            if null_icustayid:
                spec = STAY_ID_OPTIONAL_DTYPE_SPEC
            return pd.read_csv(path, dtype=spec, **kwargs)
    raise FileNotFoundError(", ".join(file_paths))

def load_intermediate_or_raw_csv(data_dir, file_name, **kwargs):
    return load_csv(os.path.join(data_dir, "intermediates", file_name),
                     os.path.join(data_dir, "raw_data", file_name), **kwargs)

def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment