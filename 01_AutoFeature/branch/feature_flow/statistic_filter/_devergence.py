import numpy as np
import pandas as pd
import scipy.stats as st

__all__ = [
    'nominal_divergence',
    'ordinal_divergence',
    'mix_divergence',
    'unlabel_nominal_divergence'
]


def nominal_divergence(x, y, x_label, y_label):
    """

    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :return: mismatch_ratio, dist_diff
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
    return [np.sum(mismatch_ratio), np.sum(dist_different)]


def mix_divergence(x, y, x_label, y_label):
    count = 0
    p_value = {}
    for i, label in enumerate(y_label.unique()):
        train_subset = x[x_label == label]
        test_subset = y[y_label == label]
        compensation = abs(min([0, train_subset.min(), test_subset.min()]))
        train_subset += compensation
        test_subset += compensation
        if len(set(pd.concat([train_subset, test_subset]))) == 1:
            count += 1
            continue
        if len(train_subset) == 0 or len(test_subset) == 0:
            count += 0
            continue

        _, p_train = st.kstest(train_subset, 'norm')
        _, p_test = st.kstest(test_subset, 'norm')
        if p_train > 0.05 and p_test > 0.05:
            _, p = st.ttest_ind(train_subset, test_subset)
            if p > 0.05:
                count += 1
        else:
            s, p = st.ks_2samp(train_subset, test_subset, alternative='two-sided')
            if s > 0.5:
                count += 1
        p_value[label] = s
    return [count, 0]


def ordinal_divergence(x, y):
    if len(set(pd.concat([x, y]))) == 1:
        return [0, 0]
    if len(x) == 0 or len(y) == 0:
        return [0, 0]
    _, p_train = st.kstest(x, 'norm')
    _, p_test = st.kstest(y, 'norm')
    if p_train > 0.05 and p_test > 0.05:
        s, p = st.ttest_ind(x, y)
    else:
        s, p = st.ks_2samp(x, y, alternative='two-sided')
    return [s, p]


def unlabel_nominal_divergence(self, train, test, mismatch_threshold, category_dist_threshold):
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
