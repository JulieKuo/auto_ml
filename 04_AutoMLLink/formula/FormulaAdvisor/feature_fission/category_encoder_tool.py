import pandas as pd
import category_encoders as ce


class CategoryEncoderTools:
    def __init__(self, category_encoder):
        self.category_encoder = category_encoder
        self.target_encoder_mapping = dict()
        self.label_encoder_mapping = dict()
        self.onehot_X_columns =[]

    def activate_encoder(self, X, y, problem_type):
        if X.empty:
            return pd.DataFrame(), self.target_encoder_mapping
        if self.category_encoder == 'TargetEncoder':
            if problem_type == 'multiclass':
                return self.target_encoder_multiclass(X, y), self.target_encoder_mapping
            if problem_type == 'binary':
                y = y == y.unique()[1]
            return self.target_encoder(X, y), self.target_encoder_mapping
        elif self.category_encoder == 'LabelEncoder':
            return self.label_encoder(X), self.label_encoder_mapping
        elif self.category_encoder == 'OneHotEncoder':
            return self.one_hot_encoder(X), self.onehot_X_columns
        else:
            raise ValueError \
                    ("'category_encoder' should be set as TargetEncoder, OneHotEncoder or LabelEncoder, got {}".format(
                    self.category_encoder))

    def progress_info(func):
        def wrap(*args, **kwargs):
            print(f"  Running {func.__name__}...", end=" ")
            result = func(*args, **kwargs)
            print("done")
            return result
        return wrap

    @progress_info
    def label_encoder(self, X):
        return self

    @progress_info
    def one_hot_encoder(self, X):
        onehot_X = pd.get_dummies(X)
        self.onehot_X_columns = list(onehot_X.columns)
        return onehot_X

    @progress_info
    def target_encoder(self, X, y):
        enc = ce.TargetEncoder()
        enc.fit(X, y)  # convert all categorical
        for i, col in enumerate(X.columns):
            a = enc.ordinal_encoder.mapping[i]['mapping']
            b = enc.mapping[col].drop(-1, axis=0)
            b.index = a.index
            b = b.to_dict()
            self.target_encoder_mapping[col] = b
        temp = enc.transform(X)  # columns for class_
        X_encoded = temp
        return X_encoded

    @progress_info
    def target_encoder_multiclass(self, X, y):
        y = y.astype(str)  # convert to string to onehot encode
        X_encoded = pd.DataFrame()
        enc = ce.OneHotEncoder().fit(y)
        y_onehot = enc.transform(y)
        class_names = y_onehot.columns  # names of onehot encoded columns
        for class_ in class_names:
            enc = ce.TargetEncoder()
            enc.fit(X, y_onehot[class_])  # convert all categorical
            for i, col in enumerate(X.columns):
                a = enc.ordinal_encoder.mapping[i]['mapping']
                b = enc.mapping[col].drop(-1, axis=0)
                b.index = a.index
                b = {str(col) + '_' + str(class_): b.to_dict()}
                try:
                    self.target_encoder_mapping[col].update(b)
                except KeyError:
                    self.target_encoder_mapping[col] = b
            temp = enc.transform(X)  # columns for class_
            temp.columns = [str(x) + '_' + str(class_) for x in temp.columns]
            X_encoded = pd.concat([X_encoded, temp], axis=1)  # add to original dataset
        return X_encoded

