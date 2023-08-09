import pandas as pd
from pandas import DataFrame
import numpy as np
import scipy.stats as st

__all__ = [
    'meta_collector'
]


def meta_collector(x: DataFrame, problem_type):
    d = {}
    for name, col in x.items():
        if name == 'G_SPMS_multiply_G_SPMS':
            print('sf')
        d[name] = {
            "median": col.median(),
            "mean": col.mean(),
            "std": col.std(),
            "var": col.var(),
            # "corr": col.corrwith(label),
            "max": col.max(),
            "min": col.min(),
            "skew": col.skew(),
            "kurt": col.kurt(),
            "Q1": col.quantile(0.25),
            "Q3": col.quantile(0.75),
            "QD": (col.quantile(0.75) - col.quantile(0.25)) / 2,
            # "ks": st.entropy(col, np.random.normal(size=len(col)))
        }
    meta_df = pd.DataFrame(d).T
    meta_df['coef'] = meta_df['var'].divide(meta_df['std'])
    meta_df['range'] = meta_df['max'].sub(meta_df['min'])
    return meta_df.astype('float32')
