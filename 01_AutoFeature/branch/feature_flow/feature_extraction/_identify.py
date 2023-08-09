import logging
import re

import pandas as pd
import numpy as np

from ..utils import BINARY, MULTICLASS, REGRESSION

__all__ = ['identify_problem_type',
           'get_feature_types',
           'get_time_format',
           'set_time_format',
           'apply_time_format',
           'type_diagnosis',
           'generate_features',
           'sorting_hat'
           ]

logger = logging.getLogger(__name__)


def identify_problem_type(y):
    """ Identifies which type of prediction problem we are interested in (if user has not specified).
        Ie. binary classification, multi-class classification, or regression.
    """
    if len(y) == 0:
        raise ValueError(f"3105@")
    y = y.dropna()  # Remove missing values from y (there should not be any though as they were removed in Learner.general_data_processing())
    num_rows = len(y)

    unique_values = y.unique()
    unique_count = len(unique_values)
    if unique_count > 10:
        logger.info(f'the first 10 unique label values in data:  {list(unique_values[:10])}')
    else:
        logger.info(f'the {unique_count} unique label values in data:  {list(unique_values)}')

    MULTICLASS_LIMIT = 1000  # if numeric and class count would be above this amount, assume it is regression
    if num_rows > 1000:
        REGRESS_THRESHOLD = 0.05  # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers
    else:
        REGRESS_THRESHOLD = 0.1

    if unique_count == 2:
        problem_type = BINARY
        reason = "only two unique label-values observed"
    elif unique_values.dtype == 'object' or unique_values.dtype.name == 'category':
        problem_type = MULTICLASS
        reason = "dtype of label-column == object"
    elif np.issubdtype(unique_values.dtype, np.floating):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= REGRESS_THRESHOLD) and (unique_count <= MULTICLASS_LIMIT):
            try:
                can_convert_to_int = np.array_equal(y, y.astype(int))
                if can_convert_to_int:
                    problem_type = MULTICLASS
                    reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                else:
                    problem_type = REGRESSION
                    reason = "dtype of label-column == float and label-values can't be converted to int"
            except:
                problem_type = REGRESSION
                reason = "dtype of label-column == float and label-values can't be converted to int"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == float and many unique label-values observed"
    elif np.issubdtype(unique_values.dtype, np.integer):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= REGRESS_THRESHOLD) and (unique_count <= MULTICLASS_LIMIT):
            problem_type = MULTICLASS  # TODO: Check if integers are from 0 to n-1 for n unique values, if they have a wide spread, it could still be regression
            reason = "dtype of label-column == int, but few unique label-values observed"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == int and many unique label-values observed"
    else:
        raise NotImplementedError('label dtype', unique_values.dtype, 'not supported!')
    logger.info(f"infers prediction problem is: {problem_type}  (because {reason}).")
    return problem_type


def get_feature_types(df):
    dtype_dict = {}
    category_boundary = np.sqrt(len(df)) // np.log10(len(df))
    for name, x in df.items():
        dtype = x.dtype
        num_unique = x.nunique()
        maximum = x.max() if dtype != 'object' else 250
        type_family = _get_type_family(dtype, num_unique, maximum, category_boundary)
        dtype_dict[name] = type_family
    return dtype_dict


def set_time_format(df, dtype_dict):
    times_dict = {}
    for col_name, dtype in dtype_dict.items():
        if dtype == 'datetime':
            time_dict = get_time_format(df[[col_name]])
            times_dict.update(time_dict)
    return times_dict


def apply_time_format(df, time_format: dict):
    for name, _format in time_format.items():
        df[name] = pd.to_datetime(df[name], format=_format)
    return df


def _get_type_family(type, num_unique, maximum, boundary):
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


time_format_dict = {"^\\d{8}$": "%Y%m%d",
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
                    "^\\d{1,2}-\\d{1,2}-\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}.\\d{6}$": "%d-%m-%Y %H:%M:%S.%f",
                    "^\\d{4}-\\d{1,2}-\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}.\\d{6}$": "%Y-%m-%d %H:%M:%S.%f",
                    "^\\d{1,2}/\\d{1,2}/\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}.\\d{6}$": "%m/%d/%Y %H:%M:%S.%f",
                    "^\\d{4}/\\d{1,2}/\\d{1,2}\\s\\d{1,2}:\\d{2}:\\d{2}.\\d{6}$": "%Y/%m/%d %H:%M:%S.%f",
                    "^\\d{1,2}\\s[a-z]{3}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}.\\d{6}$": "%d %m %Y %H:%M:%S.%f",
                    "^\\d{1,2}\\s[a-z]{4:}\\s\\d{4}\\s\\d{1,2}:\\d{2}:\\d{2}.\\d{6}$": "%d %m %Y %H:%M:%S.%f"}


def get_time_format(x):
    time_format = {}
    for name, x in x.items():
        if np.issubdtype(x.dtype, np.floating):
            continue
        if type(x[0]) == pd.Timestamp:
            continue
        for key, value in time_format_dict.items():
            if re.match(key, str(x[0])):
                try:
                    pd.to_datetime(x, format=value)
                    time_format[name] = value
                except Exception:
                    pass
    return time_format


def type_diagnosis(x):
    try:
        x.replace([r'[^0-9."]'], ['0'], regex=True, inplace=True)
    except:
        try:
            # count how many Nas are there
            na_count = sum(x.isna())
            # count how many digits are there that have decimals
            count_float = np.nansum([False if r.is_integer() else True for r in x])
            # total decimals digits
            count_float = count_float - na_count  # reducing it because we know NaN is counted as a float digit
            if count_float == 0:
                x.fillna(x.mean(), inplace=True)
        except:
            pass


def generate_features(x):
    """
    Set object dtype columns as category and add 'unknown' categories to each category columns
    :param x:  dataframe
    :return: dataframe
    """
    X_features = pd.DataFrame(index=x.index)
    X_features = X_features.join(x.select_dtypes(include=['number']))
    X_categorical = x.select_dtypes(include=['object'])
    X_categorical = X_categorical.astype('category')
    for name, col in X_categorical.items():
        try:
            X_categorical[name] = col.cat.add_categories('unknown')
        except ValueError:
            pass
    X_features = X_features.join(X_categorical)
    X_features = X_features.join(x.select_dtypes(include=['datetime64[ns]']))
    return X_features


def sorting_hat(x):
    try:
        if x.dtypes == 'category' or x.dtypes == 'object':
            return 'nominal'
        elif 'int' in str(x.dtypes) or 'float' in str(x.dtypes):
            return 'ordinal'
    except AttributeError:
        if np.issubdtype(type(x), int) or np.issubdtype(type(x), float):
            return 'ordinal'

