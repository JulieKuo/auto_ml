import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

__all__ = [
    'Standardize',
    'RangeScaler',
]


class Standardize(object):
    def __init__(self, maps=None):
        self._normalize_params = maps
        pass

    def synthesis(self, x):
        self._normalize_params = {}
        sc = StandardScaler()
        sc.fit(x)
        x = pd.DataFrame(sc.transform(x), columns=x.columns)
        self._normalize_params['mean'] = list(sc.mean_)
        self._normalize_params['scale'] = list(sc.scale_)
        return x

    def mapping(self, x, normalize_params='default'):
        normalize_params = self._normalize_params if normalize_params == 'default' else normalize_params
        assert isinstance(normalize_params, dict), f"deskew_map must be a dictionary, not {type(normalize_params)}"
        x = (x - np.asarray(normalize_params['mean'])) / np.asarray(normalize_params['scale'])
        return x

    @property
    def get_map(self):
        return self._normalize_params

    def inverse(self):
        pass


class RangeScaler(object):
    def __init__(self, maps=None):
        self._normalize_params = maps
        pass

    def synthesis(self, x):
        sc = MinMaxScaler()
        sc.fit(x)
        x = pd.DataFrame(sc.transform(x), columns=x.columns)
        self._normalize_params['min'] = list(sc.min_)
        self._normalize_params['scale'] = list(sc.scale_)
        return x

    def mapping(self, x, normalize_params='default'):
        return x

    @property
    def get_map(self):
        return self._normalize_params

    def inverse(self):
        pass
