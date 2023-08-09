import numpy as np
import pandas as pd
import scipy.stats as st
from pandas import DataFrame, Series

__all__ = [
    'nominal_divergence',
    'ordinal_divergence',
    'mix_divergence',
    'unlabel_nominal_divergence'
]


def nominal_divergence(x, y, x_label, y_label) -> (int, float):
    """
    x_mismatch: sum of x category ratio, which not show up in y
    x_mismatch: sum of y category ratio, which not show up in x
    """
    mismatch_ratio = []
    dist_different = []
    for i, label in enumerate(y_label.unique()):
        x_subset = x[x_label == label]
        y_subset = y[y_label == label]
        x_ratio = x_subset.value_counts() / len(x_subset)
        y_ratio = y_subset.value_counts() / len(y_subset)
        y_mismatch = y_ratio[~y_ratio.index.isin(x_ratio.index)].sum()
        x_mismatch = x_ratio[~x_ratio.index.isin(y_ratio.index)].sum()
        mismatch_ratio.append(y_mismatch * (len(y_subset) / len(y)))
        dist_different.append((np.sum(np.abs(x_ratio - y_ratio)) + x_mismatch + y_mismatch) / 2 * (len(y_subset) / len(y)))

    divergence = np.sum(dist_different)
    empty_set_ratio = np.sum(mismatch_ratio)
    return divergence, empty_set_ratio


def mix_divergence(x: Series, y: Series, x_label: Series, y_label: Series):
    """
    Calculate divergence of each subset which split by label
    """
    divergence = 0
    empty_set_ratio = 0
    for i, label in enumerate(y_label.unique()):
        train_subset = x[x_label == label]
        test_subset = y[y_label == label]

        # Skip if the two subset have identical value
        if len(set(pd.concat([train_subset, test_subset]))) == 1:
            continue

        # If one of subsets is empty, treat as divergence
        if len(train_subset) == 0 or len(test_subset) == 0:
            divergence += 1
            continue

        # Shift data to non-zero coordinator
        # TODO: It is not a formal practice, should be replace.
        compensation = abs(min([0, train_subset.min(), test_subset.min()]))
        test_subset += compensation
        train_subset += compensation

        # Good of Fit Test: whether the test data is normally distributed, to determine the follow-up test method
        # TODO: Good of Fit Test is hard to pass in real world dataset, use another measure instead of p-value.
        _, p_train = st.kstest(train_subset, 'norm')
        _, p_test = st.kstest(test_subset, 'norm')
        if p_train > 0.05 and p_test > 0.05:
            _, p = st.ttest_ind(train_subset, test_subset)
            if p < 0.05:
                divergence += 1
        else:
            s, p = st.ks_2samp(train_subset, test_subset, alternative='two-sided')
            if p < 0.05:
                divergence += 1
    return divergence, empty_set_ratio


def ordinal_divergence(x, y):
    if len(set(pd.concat([x, y]))) == 1:
        return [0, 0]
    if len(x) == 0 or len(y) == 0:
        return [0, 0]

    # Good of Fit Test: whether the test data is normally distributed, to determine the follow-up test method
    _, p_train = st.kstest(x, 'norm')
    _, p_test = st.kstest(y, 'norm')
    if p_train > 0.05 and p_test > 0.05:
        s, p = st.ttest_ind(x, y)
    else:
        s, p = st.ks_2samp(x, y, alternative='two-sided')

    # TODO: Treating statistic as divergence is a very poor method, should calculate divergence correctly, ex. KL, JS
    divergence = s
    empty_set_ratio = 0
    return divergence, empty_set_ratio


def unlabel_nominal_divergence(train, test, mismatch_threshold, category_dist_threshold):
    train_category_ratio = train.value_counts() / len(train)
    test_category_ratio = test.value_counts() / len(test)
    test_mismatch_ratio = test_category_ratio[~test_category_ratio.index.isin(train_category_ratio.index)].sum()
    train_mismatch_ratio = train_category_ratio[~train_category_ratio.index.isin(test_category_ratio.index)].sum()
    mismatch_ratio = test_mismatch_ratio
    dist_different = (np.sum(
        np.abs(train_category_ratio - test_category_ratio)) + train_mismatch_ratio + test_mismatch_ratio) / 2
    if mismatch_ratio > mismatch_threshold:
        return False
    elif dist_different > category_dist_threshold:
        return False
    else:
        return True
