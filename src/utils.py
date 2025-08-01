import numpy as np
import pandas as pd
from scipy import stats

def remove_z_outliers_multi(df, columns, threshold=3):
    for col in columns:
        z = stats.zscore(df[col])
        df = df[abs(z) < threshold]
    return df