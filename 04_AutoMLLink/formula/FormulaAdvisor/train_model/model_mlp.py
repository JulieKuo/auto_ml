from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib, os


class ModelMLP:
    def __init__(self, model_path):
        self.model_path = model_path

    def search(self, df, label = 'label', val_size = 0.2):
        # 將數據拆分為訓練集和測試集
        train, test = train_test_split(df, test_size=val_size)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        x_train = train.copy()
        y_train = x_train.pop(label)
        x_test = test.copy()
        y_test = x_test.pop(label)

        # 定義模型
        model = MLPRegressor()

        # 訓練模型
        model.fit(x_train, y_train)

        # 在測試數據上進行預測
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        # 計算均方誤差和R2分數
        train_mse = mean_squared_error(y_train, train_pred).round(2)
        test_mse = mean_squared_error(y_test, test_pred).round(2)
        train_r2 = r2_score(y_train, train_pred).round(2)
        test_r2 = r2_score(y_test, test_pred).round(2)

        # 保存模型
        path = os.path.join(self.model_path, 'model.pkl')
        joblib.dump(model, path)
        print(f"Save model to: {path}")

        return model, train_mse, test_mse, train_r2, test_r2
