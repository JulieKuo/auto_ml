import copy

import pandas as pd
import numpy as np

from ..utils import JsonTool, Logger, extract_label
from ..statistic_filter import *
from ..feature_selection import *
from ..feature_extraction import *
from ..transform import *


class Pipeline(object):
    """
    Pipeline of data engineering process with mate data for reusable

    Sequentially apply method filter, extraction, selection and transform

    Examples
    --------
    # >>> from flow import Pipeline
    # >>> flow = Pipeline()
    # >>> flow.setup(**params)
    #
    # >>> x_train = flow.cast(x=train, label='target', x_test=test)
    # >>> x_test = flow.shape(test)
    # >>> flow.save_mold(path='./mold.json')
    # >>> flow.load_mold(path='./mold.json')
    """
    def __init__(self, logger='default', callback=None):
        self.callback = callback
        self.logger = Logger('./log', 'pipeline.log').logger if logger == 'default' else logger
        self._meta = None
        self._params = None
        self.meta_synthesis = None
        self.meta_statistic = None
        self.meta_transform = None
        self.engine = dict()
        self.init()

    def init(self):
        """
        Call at instantiation, set params as default value and instance function
        :return:
        """
        assert self.logger is not None, ValueError("logger is 'None'")
        self.logger.info('Initialize parameters')
        self._meta = {}
        self._params = {
            "time_series": True,
            "timestamp": None,
            "drop_duplicate": False,
            "conserve_columns": [],
            "exclude_columns": [],
            "impute": "mean",
            "filter_correlation": True,
            "bucket_size": 10,
            "deskew_threshold": 0.5,
            "combine_rare_category": 0.005,
            "unary": ["polynomial", "trigonometry", "interaction"],
            "groupby": ["max", "min", "mean", "std"],
            "statistic": True,
            "allow_correlation": 0.4,
            "allow_mismatch": 0.05,
            "allow_divergence": 0.4,
            "numeric_dist": 1,
            "adversarial_valid": False,
            "resample": False,
            "normalization": "StandardScaler",
            "category_encoder": "TargetEncoder",
            "pca": False
        }
        self._bucket = Bucketize()
        self._deskew = DeSkew()
        self._crc = CombineRareCategory()
        self._dos = DifferentOfStep()
        self._encoder = None
        self._normalizer = None
        self._pca = None

    def setup(self, **kwargs):
        """
        Parameters
        ----------
        time_series : bool, default=True

        timestamp : string
            Specify columns as the main timestamp feature to generate rolling feature
            If None, will use first timestamp feature if timestamp is exist.

        drop_duplicate : bool, default=True
            Remove samples completely have same values as other sample

        conserve_columns : list
            A selection of columns to be conserved

        exclude_columns : list
            A selection of columns to be excluded

        impute : bool, default=True

        filter_correlation : bool, default=True

        bucket_size : int, default=10
            Going from a continuous variable to a categorical variable, bucketize
            numerical feature into specify bins, if over 30, it will be set to 30.

        deskew_threshold : float or int, default=0.5
            If float or int represent the threshold of skewness, if skewness beyond
            the threshold, use yeo-johnson to power-transform feature to normal distribution.
            If none, not to use this function in pipeline.

        combine_rare_category : float, default=0.005
            When set to True, all levels in categorical features below the threshold defined
             in rare_level_threshold param are combined together as a single level.

        unary : list, default=['trigonometry', 'interaction']
            A selection of unary method to be applied
            value: {'trigonometry', 'interaction'}

        groupby : list, default=['max', 'min', 'mean', 'std']
            A selection of groupby method to be applied
            value: {'max', 'min', 'mean', 'std'}

        statistic : bool, default=True

        allow_correlation : float, default=0.4

        allow_mismatch : float, default=0.05

        allow_divergence : float, default=0.4

        numeric_dist : int, default='auto'
            If int ,If 'auto'

        adversarial_valid : bool, default=False
            In-Progress

        resample : bool, default=False
            In-Progress

        normalization : string, "StandardScaler" or "MinMaxScaler"

        category_encoder : string, "OneHotEncoder" or "TargetEncoder"

        pca : float, default=False
            If float, should be between 0.0 and 1.0 and represent select the number
            of components such that the amount of variance that needs to be explained
            If False, represents not to use pca
        -------

        :return:
        """
        self._params.update(kwargs)
        self.logger.info('params has been set')

    def cast(self, x, label, x_test=None, col_type=None):
        """
        A strategy for feature extraction by modeling each feature with clean,
        synthesis, statistic and transform data automatically
        :param x: dataframe
            Input data with label, treat as training data
        :param label: str
            Column name of label
        :param x_test: dataframe(optional)
            Test data, if provided, could be calculate statistic measure to drop divergence features
        :param col_type: dict(optional)
            Specify columns' dtype, if not provided, it will be inferred by the system
        :return:
        """
        train = x.copy()
        self.logger.info('Casting feature...')
        p = self._params
        train = self.synthesis(train,
                               label,
                               col_type,
                               p['time_series'],
                               p['drop_duplicate'],
                               p['bucket_size'],
                               p['deskew_threshold'],
                               p['combine_rare_category'],
                               p['unary'],
                               p['groupby'],
                               p['conserve_columns'])

        if x_test is not None:
            test = x_test.copy()
            test = self.synthesis_shaper(test)
            train, _ = self.statistic(train,
                                      test,
                                      label,
                                      p['allow_correlation'],
                                      p['allow_mismatch'],
                                      p['allow_divergence'],
                                      p['numeric_dist'])

        train = self.transform(train, p['resample'], p['category_encoder'], p['normalization'], p['pca'])
        self.callbacks('cast', 'end', 'Complete')
        self.logger.info(f"Cast Completed. DataShape: {train.shape}")
        return train

    def shape(self, x):
        """
        Shape data by saved mold, must cast first or load mold before this step
        :param x: dataframe
            Input data to shape by mold
        :return:
        """
        self.callbacks('shape', 'shaping', 'synthesis', 0.3)
        x_sy = self.synthesis_shaper(x)
        x_st = self.statistic_shaper(x_sy)
        x_tf = self.transform_shaper(x_st)
        self.callbacks('shape', 'end', 'Complete', 1)
        self.logger.info(f"Shape Completed. DataShape: {x_tf.shape}")
        return x_tf

    def save_mold(self, path):
        """
        Save mold if a function to save cast's metadata as .mo file, which could be using by load_mold
        and shape dataset as mold's format
        :param path: os.PathLike
        :return:
        """
        self.logger.info(f'Save mold to {path}')
        assert self._params is not None
        mold = dict()
        mold['params'] = self._params
        mold['meta'] = self._meta
        JsonTool.save(mold, path)
        pass

    def load_mold(self, path):
        """

        :param path:
        :return:
        """
        self.logger.info(f'Load mold from {path}')
        mold = JsonTool.load(path)
        self._params = mold['params']
        self._meta = mold['meta']
        synthesis_meta = self._meta['synthesis']
        transform_meta = self._meta['transform']
        self._bucket = Bucketize(synthesis_meta['bucket_map'])
        self._deskew = DeSkew(synthesis_meta['deskew_map'])
        self._crc = CombineRareCategory(synthesis_meta['category_map'])
        self._dos = DifferentOfStep(synthesis_meta['delta_map'])

        # Set encoder
        if self._params['category_encoder'] == 'OneHotEncoder':
            self._encoder = OneHotEncoder(transform_meta['encoder_map'])
        if self._params['category_encoder'] == 'TargetEncoder':
            if synthesis_meta['problem_type'] == 'MULTICLASS':
                self._encoder = MultiClassTargetEncoder(transform_meta['encoder_map'])
            else:
                binomial_target = True if synthesis_meta['problem_type'] == 'BINARY' else False if synthesis_meta['problem_type'] == 'REGRESSION' else None
                self._encoder = GLMMEncoder(transform_meta['encoder_map'], binomial_target)

        # Set normalization
        if self._params['normalization'] == 'StandardScaler':
            self._normalizer = Standardize(transform_meta['normalize_vars'])
        elif self._params['normalization'] == 'MinMaxScaler':
            self._normalizer = RangeScaler(transform_meta['normalize_vars'])

        if self._params['pca']:
            self._pca = Pca(transform_meta['pca_vars'])

    def callbacks(self, func, step=None, status=None, percent=None):
        kwargs = {"func": func, "step": step, "status": status, "percent": percent}
        if step:
            self.logger.info(f"v {step}")
        if self.callback is not None:
            self.callback(**kwargs)
        # print(f"\r{func}:|" + "#" * int(percent * 40) + " " * int((1 - percent) * 40) + "|" + f"{percent * 100:.1f}%", end="")
        return

    def synthesis(self, x, label, col_type=None, time_series=False, drop_duplicate=False, bucket_size=10,
                  deskew_threshold=0.5, combine_rare_category=False, unary=None, groupby=None, conserve_columns=None):
        """
        Parameters
        ----------
        :param x: dataframe
            Input pandas dataframe with columns header and target

        :param label: string
            Label name witch intend to use in future model training,
            the name must be included in the column of the x.

        :param col_type: dict of column name -> data type, default=None
            Cast column to specified dtype by dictionary,
            use {col: dtype, …}, where col is a column label and dtype is a numpy.dtype.

        :param conserve_columns: list, default=None
            Ensure features are not removed during processing.

        :param time_series: bool, default=False
            Whether to perform the rolling feature.

        :param drop_duplicate: bool, default=False
            Remove duplicate rows.

        :param bucket_size: int or None, default=10
            Bin values into discrete intervals with K-means bins. Use bucket_size when you
            need to segment and sort data values into bins.

        :param deskew_threshold: float or None, default=0.5
            If feature skewness value greater than deskew_threshold,
            transform the feature by a Yeo-Johnson power transformation

        :param combine_rare_category: float or False, default=False
            When set to True, all levels in categorical features below the threshold defined
            in rare_level_threshold param are combined together as a single level.

        :param unary: list, default=None
            A selection of unary method to be applied
            value: {'interaction', 'trigonometry', 'polynomial'}

        :param groupby: list, default=None
            A selection of groupby method to be applied
            value: {'max', 'min', 'mean', 'std', 'var'}

        :return: x, y
        """
        self.logger.info(' ---- Synthesis ---- ')
        print(f"Fitting and generating training dataset...\nInitial shape: {x.shape}")

        conserve_columns = [] if conserve_columns is None else conserve_columns

        if col_type:
            _meta_time = set_time_format(x, col_type)
            col_type.update({col: 'float' for col, dtype in col_type.items() if dtype == 'numerical'})
        else:
            col_type = get_feature_types(x)
            _meta_time = get_time_format(x)
        col_type.update({col: 'datetime64[ns]' for col in _meta_time.keys()})
        x = apply_time_format(x, _meta_time)

        _meta_problem_type = identify_problem_type(x[label])
        col_type[label] = 'float' if _meta_problem_type == 'regression' else 'category'
        _meta_dtype = col_type

        x = x.astype(_meta_dtype)
        y = x.pop(label)
        print(f"""
Label:  {label}
Type:   {_meta_problem_type}
Dtypes: {'assigned' if col_type else 'assert'}""")

        _meta_label = label
        _meta_unique_label = y.unique().tolist() if _meta_problem_type != 'regression' else None
        _meta_initial_columns = x.columns.tolist()
        _shape = {'initial': x.shape}
        self.callbacks('synthesis', 'preprocess', 'analyze', percent=0.02)

        feature_df = generate_features(x)
        feature_df_converse = feature_df[conserve_columns]
        self.callbacks('synthesis', 'generate', 'analyze', percent=0.03)

        # Generate time related features
        time_relate_df = pd.DataFrame()
        if _meta_time:
            time_relate_df = yield_time_related(feature_df, _meta_time, time_series=time_series)
            time_relate_df = CorrelationFilter.filter(time_relate_df)
            feature_df = feature_df.drop(_meta_time.keys(), axis=1)

        # Drop id column and unique column if found
        feature_df = data_clean(feature_df, exclude=conserve_columns)
        feature_df = data_impute(feature_df)

        # TODO: Disable quasi_constant at current version, need more validation for threshold and rules.
        # feature_df = quasi_constant(feature_df, y)

        _shape.update({'clean': feature_df.shape})
        self.callbacks('synthesis', 'clean', 'analyze', percent=0.05)

        # Removed high correlation features
        feature_df = CorrelationFilter.filter(feature_df)
        _shape.update({'correlation': feature_df.shape})
        self.callbacks('synthesis', 'correlation', 'synthesis', percent=0.15)

        conserved = [col for col in conserve_columns if col not in feature_df.columns]
        feature_df[conserved] = feature_df_converse[conserved]
        _shape.update({'conserved': len(conserved)})

        _meta_keep_features = feature_df.columns.tolist()
        self.feat_category = feature_df.select_dtypes(['category']).columns.tolist()
        self.feat_numerical = feature_df.select_dtypes(['number']).columns.tolist()
        self.feat_timestamp = feature_df.select_dtypes(['datetime64[ns]']).columns.tolist()
        assert len(self.feat_category + self.feat_numerical + self.feat_timestamp) == feature_df.shape[
            1], 'feature missing'

        # TODO: Outlier detector
        # TODO: Improve outlier detection method
        # Bucketize and De-Skew numeric feature
        buckets_df = self._bucket.synthesis(feature_df, bucket_size=bucket_size) if bucket_size else pd.DataFrame()
        deskew_df = self._deskew.synthesis(feature_df, threshold=deskew_threshold) if deskew_threshold else pd.DataFrame()
        feature_df = self._crc.synthesis(feature_df) if combine_rare_category else pd.DataFrame()
        _meta_bucket_map = self._bucket.get_map if bucket_size else {}
        _meta_deskew_map = self._deskew.get_map if deskew_threshold else {}
        _meta_category_map = self._crc.get_map if combine_rare_category else {}
        self.callbacks('synthesis', 'extract', 'synthesis', percent=0.25)
        # feature_df[list(_meta_deskew_map.keys())] = deskew_df
        feature_df = pd.concat([feature_df, time_relate_df], axis=1)

        # Rolling features base on timestamp
        step_diff_df = pd.DataFrame()
        _meta_delta_map = []
        if time_series and _meta_time:
            ruler = f"{list(_meta_time.keys())[0]}_delta"
            # TODO: COMMENT
            step_diff_df = self._dos.synthesis(feature_df, ruler)
            _meta_delta_map = self._dos.get_map
        _shape.update({'new_time': time_relate_df.shape[1],
                       'new_bucket': len(_meta_bucket_map),
                       'new_deskew': deskew_df.shape[1],
                       'new_delta': step_diff_df.shape[1]})
        self.callbacks('synthesis', 'rolling', 'synthesis', percent=0.26)

        # Feature interaction and feature selection
        _meta_feat_selected = []

        # TODO: Sample size should also be considered
        # TODO: Calculate memory usage step by step instead of disable whole process
        n_group_aggr = len(self.feat_numerical) * len(self.feat_category) * 4
        n_unary_trans = len(self.feat_numerical) * (2 + 3 + (len(self.feat_numerical) - 1) * 4 / 2)
        n_synthesis_feature = n_group_aggr + n_unary_trans
        if n_synthesis_feature <= 7500:
            adv_sel = AdvancedFeatureSelectionClassic(target=label, problem_type=_meta_problem_type)

            # Unary feature synthesis and select
            if unary:
                up = UnaryPreprocess(feature_df, self.feat_numerical, interaction=False, trigonometry=False, polynomial=False)
                for attr in unary:
                    setattr(up, attr, True)
                up_features = up.transform()
                up_features = correlation_filter(up_features, feature_df[self.feat_numerical])
                up_features = pd.concat([feature_df[self.feat_numerical], up_features.fillna(0)], axis=1)

                up_selected = adv_sel.fit_transform(up_features, y, native_features=self.feat_numerical)
                up_selected = up_selected.drop(self.feat_numerical, axis=1, errors='ignore')
                _meta_feat_selected.extend(list(up_selected.columns))
                _shape.update({'unary': up_features.shape[1], 'new_unary': up_selected.shape[1]})
                feature_df = pd.concat([feature_df, up_selected], axis=1)

                self.callbacks('synthesis', 'wrap_selection-unary', 'synthesis', percent=0.6)
                del up_features, up_selected

            # Group feature synthesis and select
            if groupby:
                high_order = HigherOrderOperator(feature_df, self.feat_category, self.feat_numerical, aggregations=groupby)  # Aggr
                high_order_features = high_order.transform()
                high_order_features.dropna(axis=1, inplace=True)
                high_order_features = high_order_features.loc[:, ~(high_order_features.nunique() == 1)]

                high_order_selected = adv_sel.fit_transform(
                    pd.concat([feature_df[self.feat_numerical], high_order_features.fillna(0)], axis=1), y, native_features=self.feat_numerical)
                high_order_selected = high_order_selected.drop(self.feat_numerical, axis=1, errors='ignore')
                _meta_feat_selected.extend(list(high_order_selected.columns))
                _shape.update({'high_order': high_order_features.shape[1], 'new_aggr': high_order_selected.shape[1]})
                feature_df = pd.concat([feature_df, high_order_selected], axis=1)

                self.callbacks('synthesis', 'wrap_selection-groupby', 'synthesis', percent=0.95)
                del high_order_features, high_order_selected
            _shape.update({'selected': len(_meta_feat_selected)})

        else:
            print(f"[Warning] feature synthesis get too many feature: {n_synthesis_feature}, will not do this part")
            _shape.update({'unary': 0, 'high_order': 0, 'selected': 0})

        # Concatenate synthesized features dataframe and target
        feature_df = pd.concat([feature_df, buckets_df, step_diff_df, y], axis=1)
        feature_df[list(_meta_deskew_map.keys())] = deskew_df
        _shape.update({'concat': feature_df.shape})
        del buckets_df, deskew_df, step_diff_df

        # Drop duplicates rows
        feature_df = feature_df.drop_duplicates()
        feature_df = feature_df.reset_index(drop=True)
        _shape.update({'drop_duplicate': feature_df.shape})

        feat_count = feature_df.shape[1]
        if drop_duplicate:
            feature_df = drop_duplicate_features(feature_df, exclude=conserved)
            print(f'  Drop  duplicate features')
        self.callbacks('synthesis', 'complete', 'synthesis', percent=1)

        print(f"""
Synthesis Summary
_______________________________________________________
Method (type)          Generated            Removed   
=======================================================
Clean                                        {_shape['initial'][1] - _shape['clean'][1] - len(conserved):>4}            
_______________________________________________________
Correlation                                  {_shape['clean'][1] - _shape['correlation'][1]:>4}            
_______________________________________________________
Time-Related            {_shape['new_time']:>5}            
_______________________________________________________
Bucketize               {_shape['new_bucket']:>5}            
_______________________________________________________
De-Skew                 {_shape['new_deskew']:>5}
_______________________________________________________
Step-different          {_shape['new_delta']:>5}
_______________________________________________________
Feature Crosses         
 * Unary                {_shape['unary']:>5} (candidate)
 * Higher-Order         {_shape['high_order']:>5} (candidate)
   Wrap-Selection       {_shape['selected']:>5}
=======================================================
Duplicate column dropped: {len(y) - _shape['drop_duplicate'][0]}
Duplicate features dropped: {feat_count - feature_df.shape[1]}
Samples : {feature_df.shape[0]:>7}
features: {feature_df.shape[1]:>7}
_______________________________________________________
                """)

        self._meta['synthesis'] = {}
        self._meta['synthesis'].update({
            'label': label,
            'unique_label': _meta_unique_label,
            'problem_type': _meta_problem_type,
            'init_features': _meta_initial_columns,
            'keep_features': _meta_keep_features,
            'dtype_dict': _meta_dtype,
            'time_format': _meta_time,
            'bucket_map': _meta_bucket_map,
            'deskew_map': _meta_deskew_map,
            'delta_map': _meta_delta_map,
            'category_map': _meta_category_map,
            'selected_features': _meta_feat_selected
        })

        return feature_df

    def statistic(self, x, y, label, allow_correlation=0.4, allow_mismatch=0.05, allow_divergence=0.4, numeric_dist: int=2):
        """
        For many machine learning estimators, we expect training data and test data have same distribution
        and expect to get similar results. But in-real world cases data shift occur when there is a change
        in the data distribution. To dealing with kinds of situation, we use statistic method such like
        divergence, distribution independent, and correlation difference to test whether feature in dataset
        occur data shift or not.
        :param x: dataframe
            Training dataset
        :param y: dataframe
            Test dataset or validate dataset
        :param label: string
            The name of the label used for training afterwards
        :param allow_correlation: float
            Threshold of Correlation difference, to define tolerance of gap between
            the feature correlation of training and test, should be between 0.0 and 1.0.
            The larger allow_correlation is, the more conservative the algorithm will be.
        :param allow_mismatch: float
            Threshold of mismatch ratio, mismatch means the category in the subset
            does not appear in the relative subset, should be between 0.0 and 1.0.
            The larger allow_mismatch is, the more conservative the algorithm will be.
        :param allow_divergence: float
            Threshold of divergence, is a type of statistical distance: a measure of how
            one probability distribution train is different from a test, should be between
            0.0 and 1.0. The larger allow_divergence is, the more conservative the algorithm
            will be.
        :param numeric_dist: int
            * Deprecated since version 1.0.4
            Threshold of divergence for mix dtype divergence, The larger numeric_dist is,
            the more conservative the algorithm will be.
        :return: dataframe, dataframe
        """
        self.logger.info(' ---- Statistic ---- ')
        if label not in y.columns:
            return x, y
        assert self._meta is not None, 'Synthesis has not yet been fit'
        meta = self._meta['synthesis']
        problem_type = meta['problem_type']

        print("Evaluating statistical feature...")

        feat_category = x.select_dtypes(['category']).columns.tolist()
        feat_numerical = x.select_dtypes(['number']).columns.tolist()
        n_features = len(feat_category) + len(feat_numerical)
        print(f"Input : Categorical: {len(feat_category)}, Numerical: {len(feat_numerical)}")

        measurer = regression_measure if problem_type == 'regression' else classification_measure
        to_remove = []
        for i, name in enumerate(x.columns):
            if name == label:
                continue

            divergence, empty_set_ratio, difference = measurer(x[name], y[name], x[label], y[label])
            if np.issubdtype(type(divergence), int):
                divergence = (divergence / y[label].nunique())

            if abs(difference) >= allow_correlation or \
                    divergence >= allow_divergence or \
                    empty_set_ratio >= allow_mismatch:
                to_remove.append(name)
            self.callbacks('statistic', None, 'selecting', (i + 1) * 0.8 / n_features)
        self.callbacks('statistic', problem_type, 'selecting')

        to_keep = self._params['conserve_columns'].copy()
        to_keep.append(meta['label'])
        to_remove = [feat for feat in to_remove if feat not in to_keep]
        x_temp = x.drop(columns=to_remove, errors='ignore')
        y_temp = y.drop(columns=to_remove, errors='ignore')

        # TODO: Define a logic to avoid removing too many features
        if x_temp.shape[1] == len(to_keep) or y_temp.shape[1] == len(to_keep):
            to_remove = []
            self.logger.warn('Statistic function trying to drop all feature, skip process and keep all feature.')
        else:
            x = x_temp
            y = y_temp

        if self._params['adversarial_valid']:
            av = AdversarialValidation(target=label)
            av_to_remove = av.train(x, y)
            av_to_remove = [feat for feat in av_to_remove if feat not in to_keep]
            x = x.drop(columns=av_to_remove, errors='ignore')
            y = y.drop(columns=av_to_remove, errors='ignore')
            to_remove.extend(av_to_remove)
            self.callbacks('statistic', 'adversarial', 'selecting', percent=0.99)

        self.callbacks('statistic', 'complete', 'selecting', percent=1)
        print(f"\n{'*' * 10}Removed {len(to_remove)} low correlation features{'*' * 10}")
        print(f'Successfully completed statistic evaluation, Keep {x.shape[1] - 1} features')

        self._meta['statistic'] = {}
        self._meta['statistic'].update({'to_remove': to_remove})

        return x, y

    def transform(self, x, resample=None, category_encoder=None, normalization=None, pca=None):
        self.logger.info(' ---- Transform ---- ')
        assert self._meta is not None, 'Synthesis has not yet been fit'
        assert 1 > pca >= 0, ValueError('pca must between 0 and 1')
        print("Executing data transform...")
        meta = self._meta['synthesis']
        problem_type = meta['problem_type']
        x, y = extract_label(x, meta['label'])

        # TODO: Implement resample method
        if resample:
            if resample == 'SMOTE':
                resample_algo = None
            elif resample == 'SMOTENC':
                resample_algo = None
            elif resample == 'ADASYN':
                resample_algo = None

        x_cat = x.select_dtypes(include=['category'])
        x_num = x.select_dtypes(include=['number'])
        self.callbacks('transform', 'extract', 'transform', percent=0.2)
        if category_encoder:
            if category_encoder == 'OneHotEncoder':
                self._encoder = OneHotEncoder()
                x_cat = self._encoder.synthesis(x_cat)
            elif category_encoder == 'LabelEncoder':
                label_encoder = LabelEncoder()
                x_cat = label_encoder.synthesis(x_cat)
            elif category_encoder == 'TargetEncoder':
                if problem_type == 'binary':
                    self._encoder = GLMMEncoder(binomial_target=True)
                elif problem_type == 'regression':
                    self._encoder = GLMMEncoder(binomial_target=False)
                elif problem_type == 'multiclass':
                    # TODO: MultiClassTargetEncoder has chance lead to target leakage and cause over-fitting issue
                    self._encoder = MultiClassTargetEncoder()
                else:
                    raise AssertionError("TargetEncoder should be used in ['binary', 'multiclass] problem")
                x_cat = self._encoder.synthesis(x_cat, y)
            else:
                raise ValueError \
                    ("'category_encoder' should be set as LabelEncoder, OneHotEncoder or TargetEncoder, but got {}".format(category_encoder))
            x_num = pd.concat([x_num, x_cat], axis=1)
            x_cat = pd.DataFrame()
        _meta_encoder_map = self._encoder.get_map if category_encoder else {}
        self.callbacks('transform', 'category_encoder', 'transform', percent=0.7)

        if normalization:
            if normalization == 'StandardScaler':
                self._normalizer = Standardize()
            elif normalization == 'MinMaxScaler':
                self._normalizer = RangeScaler()
            else:
                raise ValueError \
                    ("'normalization' should be set as StandardScaler or MinMaxScaler, got {}".format(normalization))

            x_num = self._normalizer.synthesis(x_num)
        _meta_normalize_vars = self._normalizer.get_map if normalization else {}
        self.callbacks('transform', 'normalization', 'transform', percent=0.9)

        if pca:
            self._pca = Pca()
            x_num = self._pca.synthesis(x_num, pca)
        _meta_pca_vars = self._pca.get_map if pca else {}

        x = pd.concat([x_num, x_cat, y], axis=1)
        self.callbacks('transform', 'complete', 'transform', percent=1)
        self._meta['transform'] = {}
        self._meta['transform'].update({'encoder_map': _meta_encoder_map,
                                        'normalize_vars': _meta_normalize_vars,
                                        'pca_vars': _meta_pca_vars})

        return x

    def synthesis_shaper(self, x, dtypes_dict=None):
        assert self._meta is not None, 'Synthesis has not yet been fit'
        meta = self._meta['synthesis']
        label = meta['label']
        _meta_unique_label = meta['unique_label']
        _meta_problem_type = meta['problem_type']
        _meta_initial_columns = meta['init_features']
        _meta_keep_features = meta['keep_features']
        _meta_dtype = meta['dtype_dict']
        _meta_time_format = meta['time_format']
        _meta_bucket_map = meta['bucket_map']
        _meta_deskew_map = meta['deskew_map']
        _meta_delta_map = meta['delta_map']
        _meta_category_map = meta['category_map']
        _meta_feat_selected = meta['selected_features']

        print('Shaping dataset...')
        print('Initial shape: ', x.shape)
        if _meta_problem_type != 'regression':
            unseen_label = set(x[label].unique()).difference(set(_meta_unique_label))
            assert len(unseen_label) == 0, ValueError(f'3104@{unseen_label}')

        col_diff = set(_meta_keep_features).difference(set(x.columns))
        assert len(col_diff) == 0, ValueError(f'3204@{col_diff}')

        X_index = copy.deepcopy(x.index)
        for col, _format in _meta_time_format.items():
            try:
                x[col] = pd.to_datetime(x[col], format=_format)
            except Exception:
                raise TypeError(f"3202@{col};{_format}")

        for name, dtype in _meta_dtype.items():
            if dtype == 'object':
                x[name] = x[name].fillna('unknown').astype(dtype)
            elif dtype == 'category':
                x[name] = x[name].astype(dtype)
            else:
                try:
                    x[name] = x[name].replace([np.inf, -np.inf], np.nan)

                except ValueError as e:
                    print(e)
                    try:
                        x[name] = type_diagnosis(x[name])
                        x[name] = x[name].astype(dtype)
                    except:
                        raise ValueError(f'3201@{name};{dtype};{x[name].dtype.name}|{name}')
                except KeyError:
                    if name == label:
                        continue

        if label in x.columns:
            y = x.pop(label)
        else:
            y = pd.DataFrame()
        feature_df = generate_features(x)
        feature_df = data_impute(feature_df)
        del x

        time_relate_df = yield_time_related(feature_df, _meta_time_format, time_series=self._params['time_series'])
        feature_df = feature_df[_meta_keep_features]
        feat_category = feature_df.select_dtypes(['category']).columns.tolist()
        feat_numerical = feature_df.select_dtypes(['number']).columns.tolist()

        buckets_df = self._bucket.mapping(feature_df)
        deskew_df = self._deskew.mapping(feature_df)
        feature_df = self._crc.mapping(feature_df)

        feature_df[list(_meta_deskew_map.keys())] = deskew_df
        feature_df = pd.concat([feature_df, time_relate_df], axis=1)
        step_diff_df = pd.DataFrame()
        if self._params['time_series'] and _meta_time_format:
            step_diff_df = self._dos.mapping(feature_df)
        if len(_meta_feat_selected) > 0:
            up = UnaryPreprocess(feature_df, feat_numerical, mode='test', polynomial=False, trigonometry=False, interaction=False)
            for attr in self._params['unary']:
                setattr(up, attr, True)
            gb = HigherOrderOperator(feature_df, feat_category, feat_numerical, aggregations=self._params['groupby'])  # Aggr
            up_features = up.transform().fillna(0)
            gb_features = gb.transform().fillna(0)
            feat_selected = pd.concat((up_features, gb_features), axis=1)[_meta_feat_selected]
            feature_df = pd.concat((feature_df, feat_selected), axis=1)
            del up_features, gb_features

        feature_df = pd.concat([feature_df, buckets_df, step_diff_df, y], axis=1)
        feature_df.index = X_index
        return feature_df

    def statistic_shaper(self, x):
        assert self._meta is not None
        try:
            meta = self._meta['statistic']
            to_remove = meta['to_remove']
            x = x.drop(columns=to_remove)
        except Exception as err:
            pass
        return x

    def transform_shaper(self, x):
        meta = self._meta['synthesis']
        try:
            y = x.pop(meta['label'])
        except KeyError:
            y = pd.DataFrame()

        x_cat = x.select_dtypes(include=['category'])
        x_num = x.select_dtypes(include=['number'])

        if self._encoder:
            encoder_name = self._encoder.__class__.__name__
            x_cat = self._encoder.mapping(x)
            x_cat = x_cat.fillna(x_cat.mode().iloc[0])
            x_num = pd.concat([x_num, x_cat], axis=1)
            x_cat = pd.DataFrame()

        if self._normalizer:
            normalizer_name = self._normalizer.__class__.__name__
            x_num = self._normalizer.mapping(x_num)

        if self._pca:
            reducer_name = self._pca.__class__.__name__
            x_num = self._pca.mapping(x_num)

        x = pd.concat([x_num, x_cat, y], axis=1)

        return x
