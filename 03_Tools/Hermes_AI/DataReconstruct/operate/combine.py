import pandas as pd
import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split


def combine(df1, df2):
    if not all(df1.columns == df2.columns):
        raise AssertionError("2021")
    df_combined = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    return df_combined


def label_combine(df1, df2):
    if len(df1) != len(df2):
        raise AssertionError(f"2031@{len(df1)}|{len(df2)}")
    duplicate_feature = list(df1.columns.intersection(df2.columns))
    if duplicate_feature:
        raise AssertionError(f"2032@{duplicate_feature}")
    df_combined = pd.concat([df1, df2], axis=1)
    return df_combined


def split(df, split_size, shuffle, stratify):
    if stratify == []:
        df1, df2 = train_test_split(df, test_size=split_size, shuffle=shuffle)
    else:
        df1, df2 = train_test_split(df, test_size=split_size, shuffle=shuffle, stratify=df[stratify])
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    return df1, df2


def data_leakage(df, column_sort, remove_quantile, dtype):
    if column_sort: # check whether the specified sort features are datetime format. And get the detail of datetime format.
        time_data = [(col["column_name"], col["dtype_for_process"]) for col in dtype.values() if (col["advise_dtype"] == "datetime") & (col["column_name"] in column_sort)]
    else: # find the first datetime format feature to sort the data.
        time_data = [(col["column_name"], col["dtype_for_process"]) for col in dtype.values() if (col["advise_dtype"] == "datetime")][:1]
        column_sort = [time_data[0][0]]

    # change features format: string to datetime.
    for name, format_ in time_data:
        df[name] = pd.to_datetime(df[name], format = format_)

    # sort value by datetime features.
    df = df.sort_values(column_sort).reset_index(drop = True)
    
    # get number format feature to calculate distance.
    df_num = df.select_dtypes("number")
    keep_col = [i for i in df_num.columns if i not in column_sort]
    df_num = df_num[keep_col]

    # calculate the distance of a sample from the previous sample.
    distance = [None] 
    for i in range(1, len(df_num)): 
        distance.append(np.linalg.norm(df_num.iloc[i].values - df_num.iloc[i-1].values)) 

    df_num["distance"] = distance

    # remove samples that are closer to the previous sample.
    remove_boundary = df_num["distance"].quantile(remove_quantile)
    df_num = df_num.query("distance >= @remove_boundary")
    df_filtered = df.loc[df_num.index].reset_index(drop = True)
    
    # change features format: datetime to string.
    time_col = [data[0] for data in time_data]
    df_filtered[time_col] = df_filtered[time_col].astype(str)
    
    return df_filtered


def add(a: Series, b: Series) -> Series:
    return a.add(b)


def sub(a: Series, b: Series) -> Series:
    return a.sub(b)


def divide(a: Series, b: Series) -> Series:
    return a.divide(b, fill_value=0)


def multiply(a: Series, b: Series) -> Series:
    return a.mul(b)
