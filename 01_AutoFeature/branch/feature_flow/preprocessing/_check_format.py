import pandas as pd
from pandas.api.types import is_numeric_dtype


__all__ = [
    'check_head',
    'CheckFormat',
]


def check_head(file_dir):
    symbols = []
    with open(file_dir, encoding="utf-8") as f:
        lines = f.read()
        header = lines.split('\n', 1)[0].replace(",", "")
        sym = "\"\'\~`!@#$%^&*+={}?"
        for s in sym:
            if s in header:
                symbols.append(s)
        if symbols:
            raise ValueError(f'3001@{symbols}')


class CheckFormat(object):
    def __init__(self, train, test, label):
        self.train = train
        self.test = test
        self.label = label
        self.quick_check()

    def quick_check(self):
        self.check_label()
        self.check_feature()

    def check_feature(self):
        train_cols = self.train.columns
        if self.test is not None:
            test_cols = self.test.columns
            col_diff = set(train_cols).difference(set(test_cols))
            assert len(col_diff) == 0, ValueError(f'3204@{col_diff}')

    def check_label(self):
        train_label = self.train[self.label]
        assert train_label.isna().sum() == 0, AssertionError(f"3101@")
        if is_numeric_dtype(self.train[self.label]):
            assert train_label.nunique != 1, AssertionError(f"3102@")
        else:
            assert train_label.nunique != 1, AssertionError(f"3103@")
            if self.test is not None:
                test_label = self.test[self.label]
                unseen_label = set(set(train_label.unique())).difference(set(test_label.unique()))
                assert len(unseen_label) == 0, ValueError(f'3104@{unseen_label}')
