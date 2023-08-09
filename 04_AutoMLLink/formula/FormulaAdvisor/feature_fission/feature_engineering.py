import numpy as np
import scipy.stats as st
import json
import math
from utils.util_tools import not_character


class FeatureEngineering:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.rename_columns_mapping = dict()
        self.features_numerical = list()
        self.features_categorical = list()
        self.skewless_log = dict()
        self.skewless_sqrt = dict()
        self._fitted = False

    def fit_transform(self, df, label):
        df_raw = df.copy()

        for i, column in enumerate(df_raw):
            if not_character(column):
                new_name = 'column_' + str(i)
                self.rename_columns_mapping[column] = new_name
        self.rename_columns_mapping[label] = 'label'
        df_raw.rename(columns=self.rename_columns_mapping, inplace=True)

        features_numerical = df_raw.select_dtypes(include=['number']).columns.to_list()
        features_categorical = df_raw.select_dtypes(include=['object']).columns.to_list()

        print(f'  Domain feature synthesizing...', end=" ")
        n_dis = 0
        n_skew = 0

        # TODO: Improve outlier detection method
        for column in features_numerical:
            if column == 'label':
                continue
            # Power transform if skew
            # TODO: Improve this
            compensation = float(abs(min([0, df_raw[column].min()])))
            if st.skew(df_raw[column]) > 0.5:
                df_raw[column] = np.log1p(df_raw[column] + compensation)
                self.skewless_log.update({column: compensation})
                n_skew += 1
            elif st.skew(df_raw[column]) < -0.5:
                df_raw[column] = np.sqrt(df_raw[column] + compensation)
                self.skewless_sqrt.update({column: compensation})
                n_skew += 1

        print(n_dis, n_skew)
        metadata = dict()
        metadata['generator'] = [dict()]
        metadata['generator'][0].update({
            'renamed_columns_mapping': self.rename_columns_mapping,
            'features_categorical': self.features_categorical,
            'features_numerical': self.features_numerical,
            'skewless_log': self.skewless_log,
            'skewless_sqrt': self.skewless_sqrt,
        })
        with open(self.metadata_path, 'w') as outfile:
            json.dump(metadata, outfile, indent=4)

        self._fitted = True
        return df_raw

    def transform(self, df):
        df_trans = df.copy()
        if not self._fitted:
            try:
                with open(self.metadata_path, 'r') as mt:
                    metadata = json.load(mt)
            except AssertionError:
                raise AssertionError("Not fitted yet. Call 'fit_transform' with appropriate arguments before using this estimator.")
            self.rename_columns_mapping = metadata['generator'][0]['renamed_columns_mapping']
            self.features_numerical = metadata['generator'][0]['features_numerical']
            self.features_categorical = metadata['generator'][0]['features_categorical']
            self.skewless_log = metadata['generator'][0]['skewless_log']
            self.skewless_sqrt = metadata['generator'][0]['skewless_sqrt']

        if len(self.rename_columns_mapping) > 0:
            print('rename')
            df_trans = df_trans.rename(columns=self.rename_columns_mapping)

        for column in self.skewless_log.keys():
            df_trans[column] = np.log1p(df_trans[column] + self.skewless_log[column])
            df_trans[column] = df_trans[column].fillna(-1)
        for column in self.skewless_sqrt.keys():
            df_trans[column] = np.sqrt(df_trans[column] + self.skewless_sqrt[column])
            df_trans[column] = df_trans[column].fillna(-1)

        return df_trans

    def inverse_transform(self, df):
        df_inv = df.copy()
        if not self._fitted:
            try:
                with open(self.metadata_path, 'r') as mt:
                    metadata = json.load(mt)
            except AssertionError:
                raise AssertionError("Not fitted yet. Call 'fit_transform' with appropriate arguments before using this estimator.")
            self.rename_mapping = metadata['generator'][0]['renamed_columns_mapping']
            self.features_numerical = metadata['generator'][0]['features_numerical']
            self.features_categorical = metadata['generator'][0]['features_categorical']
            self.skewless_log = metadata['generator'][0]['skewless_log']
            self.skewless_sqrt = metadata['generator'][0]['skewless_sqrt']

        for column in self.skewless_log.keys():
            show = np.power(math.e, df_inv[column])
            df_inv[column] = np.power(math.e, df_inv[column]) - 1
            df_inv[column] = df_inv[column].replace(-1, 0)
        for column in self.skewless_sqrt.keys():
            df_inv[column] = np.power(df_inv[column], 2)
            df_inv[column] = df_inv[column].replace(-1, 0)

        df_inv.rename(columns=dict(zip(self.rename_mapping.values(), self.rename_mapping.keys())), inplace=True)
        return df_inv


