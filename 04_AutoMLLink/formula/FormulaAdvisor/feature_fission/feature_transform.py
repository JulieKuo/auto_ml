import copy
import logging
import warnings
import json
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import category_encoders as ce
from .category_encoder_tool import CategoryEncoderTools

logger = logging.getLogger(__name__)


class FeatureTransformers:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self._fitted = False
        self.label = None
        self.cleaner = None
        self.label_cleaner = None
        self.problem_type = None
        self.threshold = 1
        self.numerical_columns = None
        self.category_encoder_mapping = None
        self.normalization_params = dict()
        self.pca_params = dict()
        self.transform_flag = {'resample': False, 'category_encoder': False, 'normalization': False,
                               'variance_threshold': False, 'correlation': False, 'pca': False}

    def fit_transform(self, X: DataFrame, label='label', category_encoder=True, normalization=True):

        print("Executing data transform...")

        with open(self.metadata_path, 'r') as md:
            metadata = json.load(md)
        metadata['transform'] = []

        self.label = label
        X, y = self.extract_label(X)
        self.initial_columns = X.columns
        X_cate = X.select_dtypes(include=['object'])
        X_num = X.select_dtypes(include=['number'])
        self.numerical_columns = X_num.columns

        if category_encoder is not False and len(X_cate) != 0:
            cet = CategoryEncoderTools('TargetEncoder')
            X_cate, self.category_encoder_mapping = cet.activate_encoder(X_cate, y, problem_type='regression')
            X_num = pd.concat([X_num, X_cate], axis=1)
            X_cate = pd.DataFrame()
        if normalization:
            self.transform_flag['normalization'] = True
            X_num = self.scaler(X_num, 'MinMaxScaler')

        X = pd.concat([X_num, X_cate], axis=1)
        X = X[self.initial_columns]
        del X_num, X_cate

        metadata['transform'].append({
            'initial_columns': list(self.initial_columns),
            'numerical_columns': list(self.numerical_columns),
            'category_encoder_mapping': self.category_encoder_mapping,
            'normalization_params': [self.normalization_params],
            'transform_flag': [self.transform_flag],
            'label': self.label
        })
        with open(self.metadata_path, 'w') as outfile:
            json.dump(metadata, outfile, indent=4)

        self._fitted = True
        y.reset_index(drop=True, inplace=True)
        X['label'] = y
        return X

    def transform(self, X: DataFrame, af_metadata_path=None):
        if not self._fitted:
            try:
                with open(self.metadata_path, 'r') as mt:
                    metadata = json.load(mt)
            except AssertionError:
                raise AssertionError("Not fitted yet. Call 'fit_transform' with appropriate arguments before using this estimator.")
            self.initial_columns = metadata['transform'][0]['initial_columns']
            self.numerical_columns = metadata['transform'][0]['numerical_columns']
            self.category_encoder_mapping = metadata['transform'][0]['category_encoder_mapping']
            self.normalization_params = metadata['transform'][0]['normalization_params'][0]
            self.transform_flag = metadata['transform'][0]['transform_flag'][0]
            self.label = metadata['transform'][0]['label']
        has_label = self.label in X.columns
        if self.label in X.columns:
            X, y = self.extract_label(X)
        X_num = X[self.numerical_columns]
        X_cate = X[[col for col in X.columns if col not in self.numerical_columns]]

        if X_cate.empty:
            pass
        else:
            X_cate_temp = pd.DataFrame()
            for k, v in self.category_encoder_mapping.items():
                X_cate_temp[k] = X_cate[k].map(v)
            X_cate = X_cate_temp.astype(float)
            X_num = pd.concat([X_num, X_cate], axis=1)
            X_cate = pd.DataFrame()

        if self.transform_flag['normalization']:
            min_ = self.normalization_params['min']
            scale_ = self.normalization_params['scale']
            X_num *= scale_
            X_num += min_

        X = pd.concat([X_num, X_cate], axis=1)
        X = X[self.initial_columns]
        if has_label:
            X['label'] = y
        return X

    def inverse_transform(self, df):
        if not self._fitted:
            with open(self.metadata_path, 'r') as meta:
                metadata = json.load(meta)
            self.initial_columns = metadata['transform'][0]['initial_columns']
            self.numerical_columns = metadata['transform'][0]['numerical_columns']
            self.category_encoder_mapping = metadata['transform'][0]['category_encoder_mapping']
            self.transform_flag = metadata['transform'][0]['transform_flag'][0]


        # X, y = self.extract_label(df)
        X = df.copy()

        min_ = self.normalization_params['min']
        scale_ = self.normalization_params['scale']
        X[self.normalization_params['column']] = (X[self.normalization_params['column']] - np.asarray(min_)) / np.asarray(scale_)

        X_num = X[self.numerical_columns]
        X_cate = X[[col for col in X.columns if col not in self.numerical_columns]]

        X_cate_temp = pd.DataFrame()
        for k, v in self.category_encoder_mapping.items():
            v_inverse = dict(zip(v.values(), v.keys()))
            X_cate_temp[k] = X_cate[k].map(v_inverse)

        X = pd.concat([X_num, X_cate], axis=1)
        X = X[self.initial_columns]

        return X

    def extract_label(self, X):
        if self.label not in list(X.columns):
            raise ValueError(f"Provided DataFrame does not contain label column: {self.label}")
        y = X[self.label].copy()
        X = X.drop(self.label, axis=1)
        return X, y

    def progress_info(func):
        def wrap(*args, **kwargs):
            print(f"  Running {func.__name__}...", end=" ")
            result = func(*args, **kwargs)
            print("done")
            return result
        return wrap

    @progress_info
    def scaler(self, X: DataFrame, normalization):
        if normalization == 'StandardScaler':
            sc = StandardScaler()
        elif normalization == 'MinMaxScaler':
            sc = MinMaxScaler()
        else:
            raise ValueError \
                ("'normalization' should be set as StandardScaler or MinMaxScaler, got {}".format(normalization))
        if X.shape[1] == 0:
            self.transform_flag['normalization'] = False
            return pd.DataFrame()
        sc.fit(X)
        X = pd.DataFrame(sc.transform(X), columns=X.columns)
        self.normalization_params['min'] = list(sc.min_) if normalization == 'MinMaxScaler' else None
        self.normalization_params['mean'] = list(sc.mean_) if normalization == 'StandardScaler' else None
        self.normalization_params['scale'] = list(sc.scale_)
        self.normalization_params['column'] = list(X.columns)
        return X
