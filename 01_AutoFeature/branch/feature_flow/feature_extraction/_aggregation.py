import time

import pandas as pd
import numpy as np


class HigherOrderOperator(object):
    def __init__(self, df, feature_categorical, feature_numerical, mode='train', aggregations=('max', 'min', 'mean', 'std')):
        self.df = df
        self.feature_categorical = feature_categorical
        self.feature_numerical = feature_numerical
        self.mode = mode
        self.aggregations = aggregations
        self.func_map = {'max': np.max, 'min': np.min, 'mean': np.mean, 'std': np.std, 'var': np.var}
        assert len(aggregations) > 0 and all(aggr in self.func_map.keys() for aggr in aggregations), 'Unknown aggregation included'
        self.func_activated = [self.func_map[aggr] for aggr in aggregations]

    def transform(self):
        if len(self.feature_numerical) < 1:
            return pd.DataFrame()
        cate_feats = self.feature_categorical
        if not cate_feats:
            return pd.DataFrame()

        group_by = pd.concat([self.df.groupby(col)[self.feature_numerical].transform(func) for col in cate_feats for func in
                              self.func_activated], axis=1)
        group_by.columns = ['_'.join([cat, num, 'groupBy', func]) for cat in cate_feats for func in self.aggregations for num in self.feature_numerical]
        if self.mode == 'train':
            corr_df = group_by.sample(np.min((6000, len(group_by)))).corr()
            group_by = group_by.loc[:, ~(corr_df.abs() > 0.9).any(axis=1)]
        return group_by
