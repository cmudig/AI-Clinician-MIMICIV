import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
from .columns import DTYPE_SPEC

def load_csv(*file_paths, **kwargs):
    """
    Attempts to load a data CSV from the file paths given, and returns the first
    one whose file path exists.
    """
    for path in file_paths:
        if os.path.exists(path):
            return pd.read_csv(path, dtype=DTYPE_SPEC, **kwargs)
    raise FileNotFoundError(", ".join(file_paths))

def load_intermediate_or_raw_csv(data_dir, file_name):
    return load_csv(os.path.join(data_dir, "intermediates", file_name),
                     os.path.join(data_dir, "raw_data", file_name))