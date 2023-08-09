import pandas as pd
import numpy as np
from scipy import stats

from ..statistic_filter import cramer_v
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

__all__ = ['data_impute',
           'data_clean',
           'zero_filter',
           'constant_filter',
           'variety_filter',
           'categorical_imputer',
           'numerical_imputer',
           'drop_duplicate_features',
           'correlation_filter',
           'CorrelationFilter',
           ]


# TODO: Impute data by similarity
def data_impute(df, method=None, by=None):
    other_df = df.select_dtypes(exclude=['category', 'object', 'number'])
    cat_df = df.select_dtypes(include=['category', 'object'])
    cat_df = categorical_imputer(cat_df, method='empty')
    num_df = df.select_dtypes(include=['number'])
    num_df = numerical_imputer(num_df, method='mean')
    filled_df = pd.concat([other_df, cat_df, num_df], axis=1)
    return filled_df


def data_clean(df, exclude=None):
    removed = []
    for name, col in df.items():
        drop = name in exclude
        if any([
            zero_filter(col),
            constant_filter(col),
            variety_filter(col, max_category=256),
            serial_filter(col),
            drop
        ]):
            df.drop(name, axis=1, inplace=True)
            removed.append(name)
    return df


def zero_filter(x):
    # For all zero columns, same effect as constant_filter functionality, reserve for func expansions.
    if all(x == 0):
        return True
    return False


def constant_filter(x):
    if x.nunique() == 1:
        return True


def quasi_constant_filter(x, y, threshold=0.9):
    quasi_constant_feature = []
    x_filled = data_impute(x)
    for feature in x_filled.columns:
        predominant = (x_filled[feature].value_counts() / np.float(len(x_filled))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append(feature)

    mask = (x_filled.isna().sum().div(len(x_filled)).sort_values(ascending=False) > threshold)
    qc = x_filled[mask[mask].index.tolist()]
    nominal = qc.select_dtypes(['category', 'object'])
    nominal = categorical_imputer(nominal)
    nominal_val = nominal[nominal != 'unknown']
    nominal_nan = nominal[nominal == 'unknown'].sample(len(nominal_val) * 2)
    nominal = pd.concat([nominal_val, nominal_nan], axis=0)
    ordinal = y[nominal.index]
    nominal = nominal.astype('category')
    mean = ordinal.dropna().mean()
    var = ordinal.var() * len(ordinal)
    unique = nominal.cat.categories.values
    ordinal_by_nominal = np.array([np.mean(ordinal[nominal == category]) for category in unique])
    var_by_nominal = np.sum(np.square(ordinal_by_nominal - mean) * nominal.value_counts().sort_index())
    correlation = var_by_nominal / var if var != 0 else 0
    return correlation


def variety_filter(x, max_category):
    if x.dtype == 'object':
        if x.nunique() > max_category:
            return True
    return False


def serial_filter(x, col_name=None):
    default_id_list = ['id', 'code', 'number', 'serial', 'sn', 's/n', 'sequence']
    col_name = col_name if col_name else default_id_list
    if x.dtype == 'category' or x.dtype == 'object' or x.dtype == 'datetime64[ns]':
        return False
    if all((x.shift(-1) - x)[:-1] == 1):
        return True
    if x.name.lower() in col_name:
        if all((x.shift(-1) - x)[:-1] > 0):
            return True


def categorical_imputer(x, method='empty'):
    if method == 'empty':
        x = x.fillna('unknown')
    elif method == 'mode':
        mode_of_category = x.mode()[0]
        x = x.fillna(mode_of_category)
    return x


def numerical_imputer(x, method='mean'):
    iter_imputer = IterativeImputer(random_state=46)
    x_imp = iter_imputer.fit_transform(x)
    x_imp = pd.DataFrame(x_imp)
    x_imp.columns = x.columns
    return x_imp


def drop_duplicate_features(x, exclude=None):
    X_without_dups = x.T.drop_duplicates().T
    columns_new = X_without_dups.columns.values.tolist().extend(exclude)
    return x[columns_new]


def correlation_filter(features, ori_features=None, corr_threshold=0.9, method='spearman'):
    ori_features = ori_features if ori_features is not None else pd.DataFrame()
    features_df = pd.concat((ori_features, features), axis=1)
    features_df = pd.DataFrame(np.tril(features_df.corr(method=method).values, -1), columns=features_df.columns, index=features_df.columns)
    corr_df = features_df.drop(ori_features.columns)
    new_features = features.loc[:, ~(corr_df.abs() > corr_threshold).any(axis=1)]
    return new_features


class CorrelationFilter(object):
    def __init__(self, callback=None):
        self.callback = callback

    @classmethod
    def filter(cls, df):
        to_remove = []
        categorical_df = df.select_dtypes(include=['category', 'object'])
        numerical_df = df.select_dtypes(include=['number'])
        for i, (name_A, col_A) in enumerate(categorical_df.items()):
            for name_B, col_B in categorical_df[i:].items():
                if name_A != name_B and name_A not in to_remove and name_B not in to_remove:
                    if col_A.nunique() * col_B.nunique() > 10000:
                        continue
                    corr = cramer_v(col_A, col_B)
                    if corr > 0.95:
                        to_remove.append(name_B)

        corr_df = numerical_df.corr()
        corr_df = pd.DataFrame(np.tril(corr_df.values, -1), columns=numerical_df.columns, index=numerical_df.columns)
        to_drop = corr_df[(corr_df > 0.95).any()].index.to_list()
        to_remove.extend(to_drop)

        return df.drop(to_remove, axis=1)


