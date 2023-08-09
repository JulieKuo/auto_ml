import numpy as np
import pandas as pd
import scipy.stats as st

from ..feature_selection import correlation_filter


class UnaryPreprocess(object):
    def __init__(self, data, numeric_columns, mode='train', interaction=False, trigonometry=False, polynomial=False):
        self.data = data[numeric_columns]
        self.numeric_columns = numeric_columns
        self.polynomial = polynomial
        self.trigonometry = trigonometry
        self.interaction = interaction
        self.mode = mode

    def transform(self):
        if not self.numeric_columns:
            return pd.DataFrame()
        if self.polynomial:
            poly = self._polynomial(polynomial=['square', 'sqrt'])
        else:
            poly = pd.DataFrame()
        if self.trigonometry:
            tri = self._trigonometry(nonliner_features=['sin', 'cos', 'tan'])
        else:
            tri = pd.DataFrame()
        if self.interaction:
            inter = self._interaction(inter=['add', 'subtract', 'multiply', 'divide'])
        else:
            inter = pd.DataFrame()

        data_new = pd.concat((poly, tri, inter), axis=1)
        return data_new

    def _polynomial(self, polynomial):
        data = self.data.copy()
        if 'square' in polynomial:
            data_pow = data.apply(np.square)
            data_col = [f"pow({i})" for i in data_pow]
            data_pow.columns = data_col
        else:
            data_pow = pd.DataFrame()

        if 'sqrt' in polynomial:
            data_plus = data.loc[:, data.min() >= 0]
            if not data_plus.empty:
                data_sqrt = data_plus.apply(np.sqrt)
                data_col = [f"sqrt({i})" for i in data_sqrt]
                data_sqrt.columns = data_col
            else:
                data_sqrt = pd.DataFrame()
        else:
            data_sqrt = pd.DataFrame()
        dummy_all = pd.concat((data_pow, data_sqrt), axis=1)
        return dummy_all

    def _trigonometry(self, nonliner_features):
        data = self.data.copy()
        if 'sin' in nonliner_features:
            data_sin = np.sin(data)
            data_col = list(data_sin.columns)
            data_col = [f"sin({i})" for i in data_col]
            data_sin.columns = data_col
        else:
            data_sin = pd.DataFrame()

        if 'cos' in nonliner_features:
            data_cos = np.cos(data)
            data_col = list(data_cos)
            data_col = [f"cos({i})" for i in data_col]
            data_cos.columns = data_col
        else:
            data_cos = pd.DataFrame()

        if 'tan' in nonliner_features:
            data_tan = np.tan(data)
            data_col = list(data_tan.columns)
            data_col = [f"tan({i})" for i in data_col]
            data_tan.columns = data_col
        else:
            data_tan = pd.DataFrame()
        dummy_all = pd.concat((data_sin, data_cos, data_tan), axis=1)
        if self.mode == 'train':
            dummy_all = correlation_filter(dummy_all, data)
        return dummy_all

    def _interaction(self, inter):
        data = self.data.copy()

        if 'multiply' in inter:

            data_multiply = pd.concat([data.iloc[:, i+1:].mul(col[1], axis="index") for i, col in enumerate(data.items())], axis=1)
            data_multiply.columns = ["_multiply_".join([col, j]) for i, col in enumerate(data.columns) for j in data.columns[i+1:]]
            data_multiply.index = data.index
            if self.mode == 'train':
                data_multiply = correlation_filter(data_multiply, data)
        else:
            data_multiply = pd.DataFrame()

        if 'divide' in inter:

            data_divide = pd.concat([data.iloc[:, i+1:].div(col[1], axis="index") for i, col in enumerate(data.items())], axis=1)
            data_divide.columns = ["_divide_".join([col, j]) for i, col in enumerate(data.columns) for j in data.columns[i+1:]]
            data_divide.replace([np.inf, -np.inf], 0, inplace=True)
            data_divide.fillna(0, inplace=True)
            data_divide.index = data.index
            if self.mode == 'train':
                data_divide = correlation_filter(data_divide, data)
        else:
            data_divide = pd.DataFrame()

        if 'add' in inter:

            data_add = pd.concat([data.iloc[:, i+1:].add(col[1], axis="index") for i, col in enumerate(data.items())], axis=1)
            data_add.columns = ["_add_".join([col, j]) for i, col in enumerate(data.columns) for j in data.columns[i+1:]]
            data_add.index = data.index
            if self.mode == 'train':
                data_add = correlation_filter(data_add, data)
        else:
            data_add = pd.DataFrame()

        if 'subtract' in inter:

            data_subtract = pd.concat([data.iloc[:, i+1:].sub(col[1], axis="index") for i, col in enumerate(data.items())], axis=1)
            data_subtract.columns = ["_subtract_".join([col, j]) for i, col in enumerate(data.columns) for j in data.columns[i+1:]]
            data_subtract.index = data.index
            if self.mode == 'train':
                data_subtract = correlation_filter(data_subtract, data)
        else:
            data_subtract = pd.DataFrame()

        # get all the dummy data combined
        dummy_all = pd.concat((data_multiply, data_divide, data_add, data_subtract), axis=1)
        if self.mode == 'train':
            dummy_all.corr()

        return (dummy_all)

    def euclidean_distance(self):
        data = self.data[self.numeric_columns]
        data_ed = pd.concat([np.sqrt(data.sqrt.add(col.sqrt for col in data.itteritem()))])
        return data_ed
