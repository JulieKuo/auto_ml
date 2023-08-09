import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import re


class DataTypeParser:

    def __init__(self, report_progress):
        self.report_progress = report_progress
        self.time_format_dict = {"^\\d{8}$": "%Y%m%d",
                                "^\\d{1,2}-\\d{1,2}-\\d{4}$": "%d-%m-%Y",
                                "^\\d{4}-\\d{1,2}-\\d{1,2}$": "%Y-%m-%d",
                                "^\\d{1,2}/\\d{1,2}/\\d{4}$": "%m/%d/%Y",
                                "^\\d{4}/\\d{1,2}/\\d{1,2}$": "%Y/%m/%d",
                                "^\\d{1,2}\\s[a-z]{3}\\s\\d{4}$": "%d %m %Y",
                                "^\\d{1,2}\\s[a-z]{4:}\\s\\d{4}$": "%d %m %Y",
                                "^\\d{12}$": "%Y%m%d%H%M",
                                "^\\d{8}\\s\\d{4}$": "%Y%m%d %H%M",
                                "^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}$": "%d-%m-%Y %H:%M",
                                "^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}$": "%Y-%m-%d %H:%M",
                                "^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}$": "%m/%d/%Y %H:%M",
                                "^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}$": "%Y/%m/%d %H:%M",
                                "^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}$": "%d %m %Y %H:%M",
                                "^\\d{1,2}\\s[a-z]{4:}\\s\\d{4}\\s\\d{1,2}:\\d{2}$": "%d %m %Y %H:%M",
                                "^\\d{14}$": "%Y%m%d%H%M%S",
                                "^\\d{8}\\s\\d{6}$": "%Y%m%d %H%M%S",
                                "^\\d{8}T\\d{6}$": "%Y%m%dT%H%M%S",
                                "^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$": "%d-%m-%Y %H:%M:%S",
                                "^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}$": "%Y-%m-%d %H:%M:%S",
                                "^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$": "%m/%d/%Y %H:%M:%S",
                                "^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}$": "%Y/%m/%d %H:%M:%S",
                                "^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$": "%d %m %Y %H:%M:%S",
                                "^\\d{1,2}\\s[a-z]{4:}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}$": "%d %m %Y %H:%M:%S",
                                "^\\d{20}$": "%Y%m%d%H%M%S%f",
                                "^\\d{8}\\s\\d{12}$": "%Y%m%d %H%M%S%f",
                                "^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}:\\d{6}$": "%d-%m-%Y %H:%M:%S:%f",
                                "^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}:\\d{6}$": "%Y-%m-%d %H:%M:%S:%f",
                                "^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}:\\d{6}$": "%m/%d/%Y %H:%M:%S:%f",
                                "^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}:\\d{6}$": "%Y/%m/%d %H:%M:%S:%f",
                                "^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}:\\d{6}$": "%d %m %Y %H:%M:%S:%f",
                                "^\\d{1,2}\\s[a-z]{4:}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}:\\d{6}$": "%d %m %Y %H:%M:%S:%f"}
        self.dtype_info = dict()

    def check_if_datetime_feature(self, X: Series):
        # if np.issubdtype(X.dtype, np.integer):
        #     return False
        if np.issubdtype(X.dtype, np.floating):
            return False
        # if 'T' in X[0] or 'Z' in X[0]:
        #     X = X.apply(lambda x: x.replace('T', ''))
        #     X = X.apply(lambda x: x.replace('Z', ''))
        for key, value in self.time_format_dict.items():
            if re.match(key, str(X[0])):
                try:
                    pd.to_datetime(X, format=value)
                    return value
                except:
                    pass
        return False

    def get_feature_types(self, X: DataFrame):
        n_cols = X.shape[1]
        curr_progress = 0.35
        for i, column in enumerate(X):
            percent = (i + 1) / n_cols
            print(f"\rAnalyzing:|" + "#" * int(percent * 40) + " " * int((1 - percent) * 40) + "|" + f"{percent * 100:.1f}%", end="")
            col_val = X[column]
            dtype = col_val.dtype
            num_unique = len(col_val.unique())
            maximum = col_val.max() if dtype != 'object' else 250
            type_family = self.get_type_family(dtype, num_unique, maximum, np.sqrt(len(col_val)) // np.log10(len(col_val)))
            unique_count = X[column].nunique()

            if 'datetime' in dtype.name:
                initial_dtype = "timestamp"
                max = X[column].max()
                min = X[column].min()
                count = int(X[column].count())
                describe = [count, "NA", "NA", min, max, unique_count]
            elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating):
                initial_dtype = "numerical"
                describe = X[column].describe()[['count', 'mean', 'std', 'min', 'max']].tolist()
                describe.append(unique_count)
            else:
                initial_dtype = 'category'
                most_occur = X[column].value_counts().index[0]
                least_occur = X[column].value_counts().index[-1]
                count = int(X[column].count())
                describe = [count, "NA", "NA", least_occur, most_occur, unique_count]

            time_format = self.check_if_datetime_feature(col_val)
            if time_format:
                type_family = time_format
                advise_type = 'datetime'
            elif type_family == 'int' or type_family == 'float':
                advise_type = 'numerical'

            else:
                advise_type = 'category'

            if unique_count == 1 and col_val.iloc[5] == 0:
                data_quality = 2
            elif unique_count == 1:
                data_quality = 1
            else:
                data_quality = 0

            self.dtype_info[i] = {"column_name": column,
                                  "initial_dtype": initial_dtype,
                                  "advise_dtype": advise_type,
                                  "dtype_for_process": type_family,
                                  "data_quality": data_quality,
                                  "describe": describe}

            step_per_col = (0.99 - 0.35) / n_cols
            curr_progress += step_per_col
            self.report_progress(status=None, percent=curr_progress)
            percent = (i + 1) / n_cols
            print(f"\rAnalyzing:|" + "#" * int(percent * 40) + " " * int((1 - percent) * 40) + "|" + f"{percent * 100:.1f}%", end="")

        return self.dtype_info

    def get_type_family(self, type, num_unique, maximum, boundary):
        try:
            if 'datetime' in type.name:
                return 'datetime'
            elif np.issubdtype(type, np.integer):
                if num_unique <= 4:
                    return 'object'
                if num_unique <= boundary and maximum < 250:
                    return 'object'
                return 'int'
            elif np.issubdtype(type, np.floating):
                if num_unique <= 3 and maximum <= 3:
                    return 'object'
                return 'float'
        except Exception as err:
            raise err

        if type.name in ['bool', 'bool_']:
            return 'bool'
        elif type.name in ['str', 'string', 'object']:
            return 'object'
        else:
            return type.name
