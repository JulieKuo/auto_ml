import copy

import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.cluster import KMeans

from ..feature_selection import CorrelationFilter
from ..utils import fixes

__all__ = ['yield_time_related',
           'create_time_related',
           'Bucketize',
           'DeSkew',
           'CombineRareCategory',
           'DifferentOfStep'
           ]


def yield_time_related(df, time_format=None, time_series=False):
    """
    Create feature about time relative, if got multiple timestamp column, subtract each other.
    :param df: dataframe
    :param time_format: dict
    :param time_series: bool
        If is time series data, subtract the previous sample
    :return:
    """
    # assert time_format.keys() in df.columns, 'IndexError: dataframe not include columns which time_format gave'
    yielded_df = pd.DataFrame()
    for i, (name, col) in enumerate(df[time_format.keys()].items()):
        # yielded_df[name] = pd.to_datetime(col, time_format[name])
        if time_series:
            # TODO: More indicator feature should be consider
            yielded_df[f"{name}_delta"] = (col - col.shift(1)).dt.total_seconds()
            yielded_df[f"{name}_delta"] = yielded_df[f"{name}_delta"].fillna(yielded_df[f"{name}_delta"].mode()).round(1)

        time_related_dict = create_time_related(col, formats=time_format[name], serial=i)
        yielded_df = pd.concat([yielded_df, time_related_dict], axis=1)

    if len(time_format) > 1:
        delta_df = pd.concat([df.iloc[:, i:].sub(col[1], axis="index") for i, col in enumerate(df.items())], axis=1)
        delta_df.columns = ["_sub_".join([col, j]) for i, col in enumerate(df.columns) for j in df.columns[i:]]
        yielded_df = pd.concat([yielded_df, delta_df], axis=1)

    return yielded_df


def create_time_related(x, formats, serial):
    time_related_dict = {}
    if 'Y' in formats:
        time_related_dict.update({'year_' + str(serial): x.dt.year})
        time_related_dict.update({'quarter_' + str(serial): x.dt.quarter})
        time_related_dict.update({'week_' + str(serial): x.dt.weekday})
    if 'm' in formats:
        time_related_dict.update({'month_' + str(serial): x.dt.month})
    if 'd' in formats:
        time_related_dict.update({'day_' + str(serial): x.dt.day})
        time_related_dict.update({'dayofweek_' + str(serial): x.dt.dayofweek})
    if 'H' in formats:
        time_related_dict.update({'hour_' + str(serial): x.dt.hour})
    if 'M' in formats:
        time_related_dict.update({'minute_' + str(serial): x.dt.minute})
    if 'S' in formats:
        time_related_dict.update({'second_' + str(serial): x.dt.second})
    time_related_dict = pd.DataFrame(time_related_dict, dtype='category')
    return time_related_dict


class Bucketize(object):
    def __init__(self, maps=None):
        self._bucket_map = maps

    def synthesis(self, x, bucket_size):
        self._bucket_map = dict()
        bucket_df = pd.DataFrame()
        x = x.select_dtypes(['number'])
        for name, col in x.items():
            if col.nunique() > bucket_size:
                bucket_df[f"{name}_binned"], bins = kmeans_binning(col, bucket_size)
                self._bucket_map.update({name: bins})
        return bucket_df.astype('category')

    def mapping(self, x, bucket_map='default', prefix=None, suffix="binned"):
        bucket_map = self._bucket_map if bucket_map == 'default' else bucket_map
        bucket_df = pd.DataFrame()
        assert isinstance(bucket_map, dict), f"bucket_map must be a dictionary, not {type(bucket_map)}"
        prefix, suffix = fixes(prefix, suffix)
        for name, bins in bucket_map.items():
            temp_df = pd.cut(x[name], bins, labels=range(len(bins) - 1)).astype('int')
            bucket_df[f"{prefix}{name}{suffix}"] = temp_df.astype('category')
        return bucket_df

    @property
    def get_map(self):
        return self._bucket_map


class DeSkew(object):
    def __init__(self, maps=None):
        self._deskew_map = maps

    def synthesis(self, x, threshold):
        self._deskew_map = {}
        deskew_df = pd.DataFrame()
        x = x.select_dtypes(['number'])
        for name, col in x.items():
            if abs(st.skew(col)) > threshold:
                deskewed, lamb = st.yeojohnson(col)
                # Prevent too large to run xgb
                deskewed_mean = deskewed.mean()
                if deskewed_mean > 10000:
                    deskewed /= (deskewed_mean / 10)
                deskew_df[f"{name}_PT"] = deskewed
                self._deskew_map.update({name: lamb})
        return deskew_df

    def mapping(self, x, deskew_map='default'):
        deskew_map = self._deskew_map if deskew_map == 'default' else deskew_map
        deskew_df = pd.DataFrame()
        assert isinstance(deskew_map, dict), f"deskew_map must be a dictionary, not {type(deskew_map)}"
        for column, lamb in deskew_map.items():
            deskew_df[column + '_PT'] = st.yeojohnson(x[column], lamb)
        return deskew_df

    @property
    def get_map(self):
        return self._deskew_map


class CombineRareCategory(object):
    def __init__(self, maps=None):
        self._category_map = maps

    def synthesis(self, x, unique_limit=15):
        self._category_map = dict()
        x_cat = x.select_dtypes(['category'])
        for name, col in x_cat.items():
            if col.nunique() > unique_limit:
                cate_counts = col.value_counts()
                rep_list = list(cate_counts[cate_counts / len(col) < 0.005].index)
                if 'unknown' in rep_list:
                    rep_list.remove('unknown')
                new_col = col.replace(rep_list, 'others')
                x[name] = new_col
                self._category_map[name] = list(copy.deepcopy(new_col.cat.categories))
        return x

    def mapping(self, x, category_map='default'):
        category_map = self._category_map if category_map == 'default' else category_map
        assert isinstance(category_map, dict), f"category_map must be a dictionary, not {type(category_map)}"
        for name, col in x.items():
            if name not in category_map.keys():
                continue
            col_temp = col.fillna('unknown')
            col_temp = col_temp.cat.set_categories(category_map[name])
            # col_temp = col_temp.fillna('others')
            x[name] = col_temp
        return x

    @property
    def get_map(self):
        return self._category_map


class DifferentOfStep(object):
    def __init__(self, maps=None):
        self._diff_map = maps

    def synthesis(self, x, ruler):
        self._diff_map = {'_ruler_': ruler}
        delta_df = pd.DataFrame()
        x_num = x.select_dtypes(['number'])
        time_delta = x[ruler]
        time_delta_count = time_delta.value_counts().divide(len(x)).iloc[0]
        cum_sum_stamp = time_delta.cumsum()
        for name, col in x_num.items():
            delta_df[f"{name}_delta"] = (col - col.shift(1, )).fillna(0)
            self._diff_map[name] = False
            # If at least 90% is arithmetic
            if time_delta_count < 0.9:
                if abs(np.corrcoef(col, cum_sum_stamp)[0][1]) >= 0.5:
                    delta_df[f"{name}_delta"] = delta_df[f"{name}_delta"].divide(time_delta).replace(np.inf, np.nan).fillna(method='ffill')
                    self._diff_map.update({name: True})
        return delta_df

    def mapping(self, x, diff_map='default', prefix=None, suffix="binned"):
        diff_map = self._diff_map if diff_map == 'default' else diff_map
        assert isinstance(diff_map, dict), f"bucket_map must be a dictionary, not {type(diff_map)}"
        delta_df = pd.DataFrame()
        time_delta = x[diff_map['_ruler_']]
        prefix, suffix = fixes(prefix, suffix)
        for name, col in x.items():
            if name not in diff_map.keys():
                continue
            delta_df[f"{name}_delta"] = (col - col.shift(1, )).fillna(0)
            if diff_map[name]:
                delta_df[f"{name}_delta"] = delta_df[f"{name}_delta"].divide(time_delta).replace(np.inf, np.nan).fillna(
                    method='ffill')
        return delta_df

    @property
    def get_map(self):
        return self._diff_map


def kmeans_binning(x, n_bins):
    bin_epsilon = 0.000000001
    kmodel = KMeans(n_clusters=n_bins, n_init = "auto")
    kmodel.fit(x.values.reshape(-1, 1))
    c = pd.DataFrame(kmodel.cluster_centers_, columns=['kmeans']).sort_values(by='kmeans')
    w = c.rolling(2).mean().iloc[1:]
    kmeans_bins = [-np.inf] + list(w.kmeans) + [np.inf]
    X_binned = pd.cut(x, kmeans_bins, labels=range(n_bins))
    return X_binned, kmeans_bins


