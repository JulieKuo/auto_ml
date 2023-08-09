import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


class AdversarialValidation(object):
    def __init__(self, target, random_state=46):
        self.random_state = random_state
        self.target = target
        self.adv_label = 'adv_label'

    def train(self, train, test):
        train_dummy = train.copy()
        test_dummy = test.copy()
        train_dummy[self.adv_label] = 1
        test_dummy[self.adv_label] = 0
        df_all = pd.concat([train_dummy, test_dummy], axis=0)
        x = df_all.drop(self.adv_label, axis=1)
        y = df_all[self.adv_label]
        x_cat = x.select_dtypes(include=['object']).astype('category')
        x[x_cat.columns] = x_cat
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        max_sa = min(
            float(1000 / len(x_train)),
            float(np.sqrt(len(x_train) / len(x_train))),
        )

        lgbc = LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            n_jobs=-1,
            subsample=max_sa,
            random_state=self.random_state,
            importance_type='gain'
        )
        _auc = 1
        to_remove = []
        count = 0
        while _auc > 0.6:

            lgbc.fit(x_train, y_train)

            fe_imp_table = pd.DataFrame(
                lgbc.feature_importances_,
                columns=["Importance"],
                index=x_train.columns,
            ).drop(self.target).sort_values(by='Importance', ascending=False)

            y_pred = lgbc.predict(x_test)
            y_true = y_test.values
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            _auc = metrics.auc(fpr, tpr)
            top_importance = fe_imp_table.index.tolist()[0]
            x_train = x_train.drop(top_importance, axis=1)
            x_test = x_test.drop(top_importance, axis=1)
            count += 1
            # print(f"{count}. AUC={_auc}, drop {top_importance}")
            to_remove.append(top_importance)
            # if df_all.shape[1] - x_train.shape[1] > df_all.shape[1] // 4:
            #     return []
        return to_remove
