import numpy as np
import pandas as pd
from math import isnan

from ._correlation import *
from ._divergence import *
from ..feature_extraction import sorting_hat

__all__ = [
    'regression_measure',
    'classification_measure',
]


def regression_measure(x, y, x_label, y_label):
    """

    :param x: series or list, or array-list
    :param y: series or list, or array-list
    :param x_label: series or list, or array-list
    :param y_label: series or list, or array-list
    :return:
    """
    if sorting_hat(x) == 'ordinal':
        divergence, empty_set_ratio = ordinal_divergence(x, y)
        train_corr = np.corrcoef(x, x_label)[0][1]
        test_corr = np.corrcoef(y, y_label)[0][1]
        train_corr = 0 if isnan(train_corr) else train_corr
        test_corr = 0 if isnan(test_corr) else test_corr
        corr_gap = train_corr - test_corr
    elif sorting_hat(x) == 'nominal':
        divergence, empty_set_ratio = mix_divergence(x_label, y_label, x, y)
        train_corr = mix_correlation(x_label, x)
        test_corr = mix_correlation(y_label, y)
        train_corr = 0 if isnan(train_corr) else train_corr
        test_corr = 0 if isnan(test_corr) else test_corr
        corr_gap = train_corr - test_corr
    else:
        return None
    return divergence, empty_set_ratio, corr_gap


def classification_measure(x, y, x_label, y_label):
    """

    :param x: series or list, or array-list
    :param y: series or list, or array-list
    :param x_label: series or list, or array-list
    :param y_label: series or list, or array-list
    :return:
    """
    name = x.name
    if sorting_hat(x) == 'ordinal':
        divergence, empty_set_ratio = mix_divergence(x, y, x_label, y_label)
        train_corr = mix_correlation(x, x_label)
        test_corr = mix_correlation(y, y_label)
        train_corr = 0 if isnan(train_corr) else train_corr
        test_corr = 0 if isnan(test_corr) else test_corr
        corr_gap = train_corr - test_corr
    elif sorting_hat(x) == 'nominal':
        divergence, empty_set_ratio = nominal_divergence(x, y, x_label, y_label)
        train_corr = cramer_v(x, x_label)
        test_corr = cramer_v(y, y_label)
        train_corr = 0 if isnan(train_corr) else train_corr
        test_corr = 0 if isnan(test_corr) else test_corr
        corr_gap = train_corr - test_corr
    else:
        return None
    print(name, divergence, empty_set_ratio, corr_gap)
    return divergence, empty_set_ratio, corr_gap
