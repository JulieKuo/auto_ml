import pandas as pd
import numpy as np

from .util_tools import write_to_log


def loading_data(filepath):
    try:
        df = pd.read_csv(filepath)
        write_to_log(f"Data loaded rows: {df.shape[0]}, columns: {df.shape[1]}", 'info')
        return df
    except Exception as e:
        write_to_log(f"Couldn't load fail, {e}", 'error')
        pass


def contract_data_range(df, label, target, bound):
    max_label = df[label].max()
    min_label = df[label].min()
    target = np.clip(target, min_label, max_label) # 確保預測的target落在train data中的最小和最大值之間
    r = max_label - min_label # 計算label的範圍
    df_contracted = df[df[label].between(target - r * bound, target + r * bound)] # 從train data中挑選出在target附近一定範圍內的資料
    return df_contracted
