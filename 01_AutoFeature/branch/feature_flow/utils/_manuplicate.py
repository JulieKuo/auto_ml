import os
import re
import json
import logging

import numpy as np
import pandas as pd

from ..preprocessing._check_format import check_head

logger = logging.getLogger(__name__)

__all__ = [
    'LoadData',
    'RenameColumn',
    'extract_label',
]


def extract_label(x, label):
    if label not in list(x.columns):
        raise ValueError(f"3106@{label}")
    y = x[label].copy()
    x = x.drop(label, axis=1)
    return x, y


def not_character(col_name):
    for t in col_name:
        # \u4e00-\u9fa5 chinese
        # \u3040-\u309f japanese hiragana
        # \u30a0-\u30ff japanese katakana
        s = re.match('[\u4e00-\u9fa5\u30a0-\u30ff\u3040-\u309f]', t)
        if s:
            return True
    return False


class LoadData(object):
    def __init__(self):
        pass

    def load(self, path, delimiter=None, encoding='utf-8', columns_to_keep=None, dtype=None, error_bad_lines=True, header=0,
         names=None, format=None, nrows=None, skiprows=None, usecols=None, low_memory=False, converters=None,
         filters=None):
        check_head(path)
        df = pd.read_csv(path, converters=converters, delimiter=delimiter, encoding=encoding, header=header, names=names, dtype=dtype,
                         low_memory=low_memory, nrows=nrows, skiprows=skiprows, usecols=usecols)

        column_count_full = len(list(df.columns.values))
        row_count = df.shape[0]
        logger.info("Loaded data from: " + str(path) + " | Columns = " + str(column_count_full) + " | Rows = " + str(row_count) + " -> " + str(len(df)))
        return self.reduce_memory_usage(df)

    @staticmethod
    def reduce_memory_usage(df, verbose=True):
        numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    for dtype in [np.int8, np.int16, np.int32, np.int64]:
                        if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                            if (df[col] == df[col].astype(dtype)).all():
                                df[col] = df[col].astype(dtype)
                                break
                else:
                    for dtype in [np.float16, np.float32, np.float64]:
                        if c_min > np.finfo(dtype).min and c_max < np.finfo(dtype).max:
                            if (df[col] == df[col].astype(dtype)).all():
                                df[col] = df[col].astype(dtype)
                                break

        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            print("Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, 100 * (start_mem - end_mem) / start_mem)
            )
        return df


class RenameColumn(object):
    def __init__(self, maps=None):
        self._rename_map = maps
        pass

    def transform(self, x, label=None):
        for i, col in enumerate(x.columns):
            if not_character(col):
                new_name = 'column_' + str(i) if col != label else 'label'
                self._rename_map[col] = new_name
        x.rename(columns=self._rename_map, inplace=True)
        return x

    def mapping(self, x, rename_map='default'):
        rename_map = self._rename_map if rename_map == 'default' else rename_map
        assert isinstance(rename_map, dict), f"bucket_map must be a dictionary, not {type(rename_map)}"
        x = x.rename(columns=rename_map)
        return x

    @property
    def get_map(self):
        return self._rename_map

    def inverse(self, x):
        inverse_map = dict(zip(self._rename_map.values(), self._rename_map.keys()))
        x = x.rename(columns=inverse_map)
        return x


def reading_dtypes(self):
    if os.path.exists(self.dtypes_path):
        with open(self.dtypes_path, 'r', encoding="UTF-8") as md:
            datatype = json.load(md)
        self.dtypes_dict = {t['column_name']: [t['advise_dtype'], t['dtype_for_process']] for t in
                            datatype.values()}
        print("Received FileHeaderAndType")
    return self