import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

__all__ = [
    'Pca'
]


class Pca:
    def __init__(self, params=None):
        self._pca_params = params
        pass

    def synthesis(self, x, n_components):
        self._pca_params = dict()
        x_shape_before = x.shape[1]
        pca = PCA(n_components=n_components)
        x = pd.DataFrame(pca.fit_transform(x))
        self._pca_params['mean'] = pca.mean_.tolist()
        self._pca_params['components'] = pca.components_.tolist()
        self._pca_params['explained_variance'] = pca.explained_variance_.tolist()
        print(f"Decomposition numerical features from {x_shape_before} to {x.shape[1]} columns")
        return x

    def mapping(self, x, pca_params='default'):
        pca_params = self._pca_params if pca_params == 'default' else pca_params
        assert isinstance(pca_params, dict), f"deskew_map must be a dictionary, not {type(pca_params)}"
        x = x - np.array(pca_params['mean'])
        x = pd.DataFrame(np.dot(x, np.array(pca_params['components']).T))
        return x

    @property
    def get_map(self):
        return self._pca_params

    def inverse(self):
        pass



