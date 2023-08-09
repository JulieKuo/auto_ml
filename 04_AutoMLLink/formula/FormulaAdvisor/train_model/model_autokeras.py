import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import save_model


class ModelAutokeras:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

    def search(self, df, label='label', num_models=50, epochs=500, val_size=0.2):
        train, test = train_test_split(df, test_size=val_size)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        x_train = train.copy()
        y_train = x_train.pop(label)
        x_val = test.copy()
        y_val = x_val.pop(label)

        reg = ak.StructuredDataRegressor(
            # project_name=self.model_name,
            # directory=r"C:\Users\tzuli\Documents\python\1_AI\Formula\FormulaAdvisor\train\model",
            directory='/home/stadmin/AIPlatform/ExecutiveFile/formula/FormulaAdvisor/train/model',
            overwrite=True,
            max_trials=num_models,
            )  # It tries 3 different models.

        # Feed the structured data regressor with training data.
        reg.fit(
            # The path to the train.csv file.
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            # The name of the label column.
            epochs=epochs,
        )

        train_acc = round(reg.evaluate(x_train, y_train)[0], 2)
        test_acc = round(reg.evaluate(x_val, y_val)[0], 2)

        self.best_model = reg.export_model()
        print(self.best_model.summary())

        pred_train = self.best_model.predict(x_train)
        pred_val = self.best_model.predict(x_val)
        r2_train = r2_score(pred_train, y_train).round(2)
        r2_val = r2_score(pred_val, y_val).round(2)

        save_model(self.best_model, self.model_path)

        return self.best_model, train_acc, test_acc, r2_train, r2_val
