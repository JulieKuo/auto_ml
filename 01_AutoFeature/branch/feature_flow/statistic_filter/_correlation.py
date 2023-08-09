import numpy as np
import pandas as pd
import scipy.stats as st

__all__ = [
    'cramer_v',
    'mix_correlation',
    'regression_corr',
    'person_cor',
]


def cramer_v(x, y):
    """
    Cramer V is a measure of association between two nominal variables(x, y),
    and giving a value between 0 and +1
    :param x: series or list, or array-list
    :param y: series or list, or array-list
    :return: float, correlation percentage of x and y
    """
    cross_table = pd.crosstab(x, y)
    if len(cross_table) == 0:
        return 0
    x2, _, _, _ = st.chi2_contingency(cross_table, correction=False)
    if x2 != 0:
        total = cross_table.sum().sum()
        categorical_correlation = np.sqrt(x2 / (total * (np.min(cross_table.shape) - 1)))
        return categorical_correlation
    else:
        return 0


def mix_correlation(ordinal, nominal):
    """
    Measure of association between nominal and ordinal variables(x, y),
    and giving a value between 0 and +1
    :param ordinal: series or list, or array-list
    :param nominal: series or list, or array-list
    :return: float, correlation percentage of x and y
    """
    mean = ordinal.dropna().mean()
    var = ordinal.var() * len(ordinal)
    try:
        unique = nominal.cat.categories.values
    except:
        pass
    ordinal_by_nominal = np.array([np.mean(ordinal[nominal == category]) for category in unique])
    var_by_nominal = np.sum(np.square(ordinal_by_nominal - mean) * nominal.value_counts().sort_index())
    correlation = var_by_nominal / var if var != 0 else 0
    return correlation


def regression_corr(x, y):
    corr = pd.DataFrame(np.corrcoef(X.T))
    corr.columns = X.columns
    corr.index = X.columns
    corr = corr[label].abs()
    corr = corr[corr.index != label]  # drop the target column
    corr = corr[corr < corr_threshold]
    self.features_to_remove.extend(corr.index)


def person_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    return np.maximum(np.minimum(result, 1.0), -1.0)

