@staticmethod
def detect_outliers(X: Series):
    t = X.name
    Q1 = np.percentile(X, 25)
    Q3 = np.percentile(X, 75)
    IQR = Q3 - Q1
    if IQR != 0:
        outlier_step = 100 * IQR
        outlier_list_col = X[(X < Q1 - outlier_step) | (X > Q3 + outlier_step)].index
        return outlier_list_col
    else:
        Q1 = np.percentile(X, 25)
        Q3 = np.percentile(X, 75)
        std = X.std()
        var = X.var()
        IQR = Q3 - Q1
        outlier_step = 3 * IQR
        outlier_list_col = X[(X < Q1 - outlier_step) | (X > Q3 + outlier_step)].index
        return outlier_list_col