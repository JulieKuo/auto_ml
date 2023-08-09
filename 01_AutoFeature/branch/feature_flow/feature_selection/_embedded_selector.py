import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from lightgbm import LGBMClassifier as lgbmc
from lightgbm import LGBMRegressor as lgbmr


class AdvancedFeatureSelectionClassic(BaseEstimator, TransformerMixin):
    """
    - Selects important features and reduces the feature space. Feature selection is based on Random Forest , Light GBM and Correlation
    - to run on multiclass classification , set the subclass argument to 'multi'
  """

    def __init__(
        self,
        target,
        problem_type="classification",
        top_features_to_pick=0.10,
        random_state=42,
        subclass="ignore",
        n_jobs=4,
        callback=None,
    ):
        self.target = target
        self.problem_type = problem_type
        self.top_features_to_pick = 1 - top_features_to_pick
        self.random_state = random_state
        self.subclass = subclass
        self.n_jobs = n_jobs
        self.report_progress = None
        self.callback = None

    def fit(self, dataset, y=None):
        self.fit_transform(dataset, y=y)
        return self

    def transform(self, dataset, y=None):
        # return the data with onlys specific columns
        data = dataset
        # self.selected_columns.remove(self.target)
        data = data[self.selected_columns_test]
        if self.target in dataset.columns:
            data[self.target] = dataset[self.target]
        return data

    def fit_transform(self, dataset, y=None, native_features=None):
        dummy_all = dataset.copy()
        # dummy_all[self.target] = dummy_all[self.target].astype("float32")

        # Random Forest
        max_fe = min(70, int(np.sqrt(len(dummy_all.columns))))
        max_sa = min(1000, int(np.sqrt(len(dummy_all))))

        if self.problem_type != "regression":
            m = rfc(
                2000,
                max_depth=5,
                max_features=max_fe,
                n_jobs=self.n_jobs,
                max_samples=max_sa,
                random_state=self.random_state,
            )
            y = y.astype(str)
        else:
            m = rfr(
                2000,
                max_depth=5,
                max_features=max_fe,
                n_jobs=self.n_jobs,
                max_samples=max_sa,
                random_state=self.random_state,
            )

        m.fit(dummy_all, y)
        self.rf_fe_imp_table = pd.DataFrame(
            m.feature_importances_,
            columns=["Importance"],
            index=dummy_all.columns,
        ).sort_values(by='Importance', ascending=False)

        min_max_scaler = MinMaxScaler()
        self.rf_fe_imp_table['Importance'] = min_max_scaler.fit_transform(self.rf_fe_imp_table)
        top = self.rf_fe_imp_table.index
        dummy_all_columns_RF = dummy_all[top].columns

        # LightGBM
        max_fe = min(70, int(np.sqrt(len(dummy_all.columns))))
        max_sa = min(
            float(1000 / len(dummy_all)),
            float(np.sqrt(len(dummy_all) / len(dummy_all))),
        )

        if self.problem_type != "regression":
            m = lgbmc(
                n_estimators=2000,
                max_depth=5,
                n_jobs=self.n_jobs,
                subsample=max_sa,
                random_state=self.random_state,
            )
            y = y.astype(str)
        else:
            m = lgbmr(
                n_estimators=2000,
                max_depth=5,
                n_jobs=self.n_jobs,
                subsample=max_sa,
                random_state=self.random_state,
            )
        m.fit(dummy_all, y)
        self.lgb_fe_imp_table = pd.DataFrame(
            m.feature_importances_,
            columns=["Importance"],
            index=dummy_all.columns,
        ).sort_values(by='Importance', ascending=False)
        self.lgb_fe_imp_table['Importance'] = min_max_scaler.fit_transform(self.lgb_fe_imp_table)
        weight_fe_imp_table = self.rf_fe_imp_table * 0.4 + self.lgb_fe_imp_table * 0.6
        weight_fe_imp_table = weight_fe_imp_table[weight_fe_imp_table.Importance > 0.005].sort_values(by='Importance', ascending=False)
        std = weight_fe_imp_table.std()
        avg = weight_fe_imp_table.mean()
        bound = float(avg + std * 2)
        selected = weight_fe_imp_table[weight_fe_imp_table['Importance'] > bound]
        unselected = weight_fe_imp_table[~(weight_fe_imp_table['Importance'] > bound)]
        selected = pd.concat([selected, unselected.sample(int(len(selected) * 0.1) + 1, random_state=self.random_state)])

        top = selected.index
        dummy_all_columns = dummy_all[top].columns

        # we can now select top correlated feature

        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        dummy_all = dummy_all.select_dtypes(['number'])
        dummy_all[self.target] = y
        if self.subclass != "multi":
            corr = pd.DataFrame(np.corrcoef(dummy_all.T))
            corr.columns = dummy_all.columns
            corr.index = dummy_all.columns
            corr = corr.drop(native_features)
            corr_df = corr[native_features]
            corr = corr.loc[~(corr_df.abs() > 0.9).any(axis=1), :]

            # corr = corr[self.target].abs().sort_values(ascending=False)[0:self.top_features_to_pick+1]
            corr = corr[self.target].abs()
            corr = corr[corr.index != self.target]
            bound = float(corr.mean() + corr.std() * 2)
            # drop the target column
            # corr = corr[corr >= corr.quantile(self.top_features_to_pick)]
            corr = corr[corr >= 0.3]
            corr = pd.DataFrame(dict(features=corr.index, value=corr)).reset_index(
                drop=True
            )
            corr = corr.drop_duplicates(subset="value")
            corr = corr["features"]
            # corr = pd.DataFrame(dict(features=corr.index,value=corr)).reset_index(drop=True)
            # corr = corr.drop_duplicates(subset='value')[0:self.top_features_to_pick+1]
            # corr = corr['features']
        else:
            corr = list()

        self.corr = corr

        self.selected_columns = list(
            set(list(dummy_all_columns) + list(corr)))

        self.selected_columns_test = dataset[self.selected_columns].columns
        return dataset[self.selected_columns]


# Boruta Feature Selection algorithm
# Base on: https://github.com/scikit-learn-contrib/boruta_py/blob/master/boruta/boruta_py.py
class Boruta_Feature_Selection(BaseEstimator, TransformerMixin):
    """
            Boruta selection algorithm based on borutaPy sklearn-contrib and
            Miron B Kursa, https://m2.icm.edu.pl/boruta/
            Selects the most important features.
              Args:
                target (str): target column name
                problem_type (str): case: classification or regression
                top_features_to_pick: to make...
                max_iteration {int): overall iterations of shuffle and train forests 
                alpha {float): p-value on which 
                the option to favour one measur to another. e.g. if value is .6 , during feature selection tug of war, correlation target measure will have a higher say.
                A value of .5 means both measure have equal say 
    """

    def __init__(self, target, problem_type='classification', top_features_to_pick=.10,
                 max_iteration=25, alpha=0.05, percentile=65,
                 random_state=42, subclass='ignore'):
        self.target = target
        self.problem_type = problem_type
        self.top_features_to_pick = 1 - top_features_to_pick
        self.random_state = random_state
        self.subclass = subclass
        self.max_iteration = max_iteration
        self.alpha = alpha
        self.percentile = percentile

    def fit(self, dataset, y=None):
        return (None)

    def transform(self, dataset, y=None):
        # return the data with columns which match the threshold
        data = dataset.copy()
        # self.selected_columns.remove(self.target)
        return (data[self.selected_columns_test])

    def fit_transform(self, dataset, y=None):
        dummy_data = dataset.copy()
        X = dummy_data
        n_sample, n_feat = X.shape
        _iter = 1
        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat, dtype=np.int)
        # counts how many times a given feature was more important than
        # the best of the shadow features
        shadow_max = list()
        hits = np.zeros(n_feat, dtype=np.int)
        tent_hits = np.zeros(n_feat)
        # make seed to get same results
        np.random.seed(self.random_state)
        while np.any(dec_reg == 0) and _iter < self.max_iteration:
            # get tentative features
            x_ind = self._get_idx(X, dec_reg)
            X_tent = X.iloc[:, x_ind].copy()
            X_boruta = X_tent.copy()
            # create boruta features
            for col in X_tent.columns:
                X_boruta["shadow_{}".format(col)] = np.random.permutation(X_tent[col])
            # train imputator
            feat_imp_X, feat_imp_shadow = self._inputator(X_boruta, X_tent, y, dec_reg)
            # add shadow percentile to history
            thresh = np.percentile(feat_imp_shadow, self.percentile)
            shadow_max.append(thresh)
            # confirm hits
            cur_imp_no_nan = feat_imp_X
            cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
            h_ = np.where(cur_imp_no_nan > thresh)[0]
            hits[h_] += 1
            # add importance to tentative hits
            tent_hits[x_ind] += feat_imp_X
            # do statistical testsa
            dec_reg = self._do_tests(dec_reg, hits, _iter)
            if _iter < self.max_iteration:
                _iter += 1

        # fix tentative onse if exist
        # print(dec_reg) #no print required
        confirmed = np.where(dec_reg == 1)[0]

        tentative = np.where(dec_reg == 0)[0]
        if len(tentative) == 0:
            confirmed_cols = X.columns[confirmed]
        else:
            median_tent = np.median(tent_hits[tentative])
            tentative_confirmed = np.where(median_tent > np.median(shadow_max))[0]
            tentative = tentative[tentative_confirmed]
            confirmed_cols = X.columns[np.concatenate((confirmed, tentative), axis=0)]

        self.confirmed_cols = confirmed_cols.tolist()
        # self.confirmed_cols.append(self.target)

        # self.selected_columns_test = dataset[self.confirmed_cols].drop(self.target, axis=1).columns

        return dataset[self.confirmed_cols]

    def _get_idx(self, X, dec_reg):
        x_ind = np.where(dec_reg >= 0)[0]
        # be sure that dataset have more than 5 columns
        if len(x_ind) < 5 and X.shape[1] > 5:
            additional = [i for i in range(X.shape[1]) if i not in x_ind]
            length = 6 - len(x_ind)
            x_ind = np.concatenate((x_ind, np.random.choice(additional, length, replace=False)))
            return x_ind
        elif len(x_ind) < 5 and X.shape[1] < 5:
            return x_ind
        else:
            return x_ind

    def _inputator(self, X_boruta, X, y, dec_reg):
        feat_imp_X = feat_imp_shadow = np.zeros(X.shape[1])
        # Random Forest
        max_fe = min(70, int(np.sqrt(len(X.columns))))
        max_sa = min(1000, int(np.sqrt(len(X))))
        if self.problem_type != 'regression':
            m = lgbmc(n_estimators=500, max_depth=9, n_jobs=-1, subsample=max_sa,
                      bagging_fraction=0.99, random_state=self.random_state)
        else:
            m = lgbmr(n_estimators=500, max_depth=9, n_jobs=-1, subsample=max_sa,
                      bagging_fraction=0.99, random_state=self.random_state)

        m.fit(X_boruta, y)
        ### store feature importance
        feat_imp_X = m.feature_importances_[:len(X.columns)]
        feat_imp_shadow = m.feature_importances_[len(X.columns):]

        return feat_imp_X, feat_imp_shadow

    def _do_tests(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = stats.binom.cdf(hits, _iter, .5).flatten()

        # as in th original Boruta, we simply do bonferroni correction
        # with the total n_feat in each iteration
        to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
        to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg