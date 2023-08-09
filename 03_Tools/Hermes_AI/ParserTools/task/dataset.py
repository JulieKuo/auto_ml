import pandas as pd
import numpy as np


class TabularDataset(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        file_path = kwargs.get('file_path', None)
        name = kwargs.get('name', None)
        feature_types = kwargs.get('feature_types', None)
        df = kwargs.get('df', None)
        subsample = kwargs.get('subsample', None)
        delimiter = kwargs.get('delimiter', None)
        construct_from_df = False

        if df is not None:
            construct_from_df = True
            if not isinstance(df, pd.DataFrame):
                raise ValueError("'df' must be existing pandas DataFrame. To read dataset from file instead, use 'file_path' string argument")
            if file_path is not None:
                warnings.warn("Both 'df' and 'file_path' supplied. Creating dataset based on DataFrame 'df' rather than reading from 'file_path'")
            df = df.copy(deep=True)
        elif file_path is not None:
            construct_from_df = True
            df = load_pd.load(file_path, delimiter)
        if construct_from_df:
            if subsample is not None:
                if not isinstance(subsample, int) or subsample <= 1:
                    raise ValueError("'subsample' must be of type int and > 1")
                df.head(subsample)

            df.replace(['NA', 'na'], np.nan, inplace=True)
            super().__init__(df)
            self.file_path = file_path
            self.name = name
            self.feature_types = feature_types
            self.subsample = subsample
        else:
            super().__init__(*args, **kwargs)

