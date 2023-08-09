import numpy as np
import pandas as pd
import category_encoders as ce
from math import isnan

__all__ = [
    'OneHotEncoder',
    'LabelEncoder',
    'GLMMEncoder',
    'MultiClassTargetEncoder'
]


class OneHotEncoder(object):
    def __init__(self, maps=None):
        self._encoder_mapping = maps

    def synthesis(self, x):
        onehot_X = pd.get_dummies(x)
        self._encoder_mapping = list(onehot_X.columns)
        return onehot_X

    def mapping(self, x, encoder_map='default'):
        encoder_map = self._encoder_mapping if encoder_map == 'default' else encoder_map
        assert isinstance(encoder_map, dict), f"deskew_map must be a dictionary, not {type(encoder_map)}"
        onehot_x = pd.get_dummies(x)
        x_dummy = pd.DataFrame(columns=encoder_map)
        x_cate = onehot_x.align(x_dummy, join='right', axis=1, fill_value=0)[0]
        return x_cate

    @property
    def get_map(self):
        return self._encoder_mapping

    def inverse(self):
        pass


class LabelEncoder(object):
    def __init__(self):
        pass

    def synthesis(self):
        pass

    def mapping(self):
        pass

    def inverse(self):
        pass

    @property
    def get_map(self):
        return


class GLMMEncoder(object):
    def __init__(self, maps=None, binomial_target=False):
        self._encoder_map = maps
        self.binomial_target = binomial_target

    def synthesis(self, x, y):
        self._encoder_map = {}
        enc = ce.GLMMEncoder(binomial_target=self.binomial_target)

        if not pd.api.types.is_numeric_dtype(y) and self.binomial_target:
            _label_map = {label: i for i, label in enumerate(y.unique())}
            y = y.map(_label_map)
            # self._encoder_map['_label_map_'] = _label_map

        enc.fit(x, y)
        for i, col in enumerate(x.columns):
            category_map = enc.ordinal_encoder.mapping[i]['mapping']
            encoder_map = enc.mapping[col].drop(-1, axis=0)
            encoder_map.index = category_map.index
            encoder_map = encoder_map.to_dict()
            if pd.api.types.is_integer_dtype(x[col].cat.categories):
                new_key = [int(idx) if not isnan(idx) else idx for idx in encoder_map.keys()]
                encoder_map = dict(zip(new_key, encoder_map.values()))
            self._encoder_map[col] = encoder_map
        X_encoded = enc.transform(x)
        return X_encoded

    @property
    def get_map(self):
        return self._encoder_map

    def mapping(self, x, encoder_map='default'):
        encoder_map = self._encoder_map if encoder_map == 'default' else encoder_map
        assert isinstance(encoder_map, dict), f"encoder_map must be a dictionary, not {type(encoder_map)}"
        x_temp = pd.DataFrame()
        for name, map_dict in encoder_map.items():
            if 'NaN' in map_dict.keys():
                if pd.api.types.is_integer_dtype(x[name].cat.categories):
                    new_key = [int(k) if k != 'NaN' else np.nan for k in map_dict.keys()]
                    map_dict = dict(zip(new_key, map_dict.values()))
                elif pd.api.types.is_float_dtype(x[name].cat.categories):
                    new_key = [float(k) if k != 'NaN' else np.nan for k in map_dict.keys()]
                    map_dict = dict(zip(new_key, map_dict.values()))
            x_temp[name] = x[name].map(map_dict)
        x_cate = x_temp.astype(float)
        return x_cate

    def inverse(self):
        pass


class MultiClassTargetEncoder(object):
    def __init__(self, maps=None):
        self._encoder_map = maps

    def synthesis(self, x, y):
        self._encoder_map = dict()
        y = y.astype(str)  # convert to string to onehot encode
        X_encoded = pd.DataFrame()
        enc = ce.OneHotEncoder().fit(y)
        y_onehot = enc.transform(y)
        class_names = y_onehot.columns  # names of onehot encoded columns
        for class_ in class_names:
            enc = ce.TargetEncoder()
            enc.fit(x, y_onehot[class_])  # convert all categorical
            for i, col in enumerate(x.columns):
                a = enc.ordinal_encoder.mapping[i]['mapping']
                b = enc.mapping[col].drop(-1, axis=0)
                b.index = a.index
                b = {str(col) + '_' + str(class_): b.to_dict()}
                try:
                    self._encoder_map[col].update(b)
                except KeyError:
                    self._encoder_map[col] = b
            temp = enc.transform(x)  # columns for class_
            temp.columns = [str(x) + '_' + str(class_) for x in temp.columns]
            X_encoded = pd.concat([X_encoded, temp], axis=1)  # add to original dataset
        return X_encoded

    def mapping(self, x, encoder_map='default'):
        encoder_map = self._encoder_map if encoder_map == 'default' else encoder_map
        assert isinstance(encoder_map, dict), f"encoder_map must be a dictionary, not {type(encoder_map)}"
        x_temp = pd.DataFrame()
        for col, value in encoder_map.items():
            for name, map_dict in value.items():
                x_temp[name] = x[col].map(map_dict)
        x_cate = x_temp.astype(float)
        return x_cate

    @property
    def get_map(self):
        return self._encoder_map

    def inverse(self):
        pass


class LabelEncoderExt(object):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        self.label_encoder = self.label_encoder.fit(
            list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_
        return self

    def transform(self, data_list):
        new_data_list = ['Unknown' if x not in self.classes_ else x for x in data_list]

        return self.label_encoder.transform(new_data_list)

    def fit_transform(self, data_list):
        return self.fit(data_list).transform(data_list)


class MultiColumnLabelEncoder():
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X_train, X_test):
        train_out = X_train.copy()
        test_out = X_test.copy()
        if self.columns is not None:
            for col in self.columns:
                label_encoder = LabelEncoderExt()
                train_out[col] = label_encoder.fit_transform(train_out[col])
                test_out[col] = label_encoder.transform(test_out[col])
        else:
            for colname, col in output.items():
                output[col] = self.label_encoder.fit_transform(col)

        return train_out, test_out

    def fit_transform(self, X_train, X_test, y=None):
        return self.fit(X_train, y).transform(X_train, X_test)