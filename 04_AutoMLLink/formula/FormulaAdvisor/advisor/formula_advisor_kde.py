import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from utils import loading_model
import os, time

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)


class FormulaAdvisorKde:
    def __init__(self, model_path, df, constrain_df):
        self.best_loss = np.inf
        self.advisor_formula = pd.DataFrame()
        self.df = df
        self.constrain_df = constrain_df
        self.gradient_map = {col: self._kde_gradient(df[col]) for col in df}
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.model = loading_model(os.path.join(model_path, "keras"))

    def _grad(self, model, inputs, targets):
        '''計算模型的梯度'''
        with tf.GradientTape() as tape: # 開啟一個梯度帶，以便記錄梯度計算過程
            loss_value = self._loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_variables) # tape.gradient()計算梯度，返回一個梯度張量，各層權重所相應的梯度

    def _loss(self, model, x, y):
        '''計算模型預測結果和目標數據之間的loss'''
        y_ = model(x)
        return self.loss_object(y_true=y, y_pred=y_) # 透過MSE計算loss value

    @staticmethod
    def _kde_gradient(x):
        '''使用核密度估計方法來計算一個連續變量的概率密度函數並計算其梯度，供_slope使用'''
        values = x.to_numpy()
        positions = np.mgrid[values.min():values.max():100j] # 生成一個等間隔的點集合，範圍為values的最小值到最大值之間，共有100個點
        kernel = stats.gaussian_kde(values, ) # 計算核密度估計值。根據給定數據生成一個連續的概率密度函數。
        kde_df = np.reshape(kernel(positions).T, positions.shape) # 將核密度估計值轉換為與positions相同的形狀，以將核密度估計值與位置相對應
        kde_grad = np.gradient(kde_df) # 為每個位置返回一個梯度值，表示在該位置處的密度變化率
        kde_df = pd.DataFrame(kde_df)
        grad_df = kde_df.rename({0: 'kde'}, axis='columns')
        grad_df.insert(loc=0, column='position', value=positions)
        grad_df['gradient'] = kde_grad
        # grad_df = pd.DataFrame(kde_grad, index=kde_df_unique.index, columns=['gradient'])
        return grad_df

    def _slope(self, temp):
        '''根據self.initial_formula中的列值，在self.gradient_map中查找對應的概率密度函數之梯度值，並將這些梯度值作為斜率存儲在列表v中'''
        v = []
        for point, col in zip(temp.iloc[0].values, self.initial_formula.columns):
            df = self.gradient_map[col].copy()
            try:
                y1 = df[(df.position > point)].iloc[0]['gradient'] # 從df中選取位置大於point的第一個梯度值
            except IndexError:
                y1 = 0
            slope = y1
            v.append(slope)
        return v

    def train(self, target, initial_formula, epochs=1000, fitting_rate=0.1, patient=3, time_limit=60):        
        start = time.time()
        self.initial_formula = initial_formula
        epoch_of_stuck = 0
        increase_count = 0
        beta = 0.9
        epsilon = 0.0001
        curr_formula = self.initial_formula.copy()
        curr_formula.mask(curr_formula == 0, 0.0012, inplace=True)
        v = np.array([0] * curr_formula.shape[1])
        best_loss, grads = self._grad(self.model, curr_formula, target) # 初始化梯度向量
        print('_'*100)
        # print("INIT: ", curr_formula)
        for epoch in range(epochs):
            grad_first_layer = grads[0].numpy() # 神經網路第一層參數的梯度，即輸入層的梯度
            grad_of_feats = grad_first_layer.sum(axis=1) # 特徵的梯度加總
            # grad_of_feats = np.where(grad_of_feats >= 0, np.log1p(grad_of_feats), -np.log(-grad_of_feats))
            grad_sum_with_abs = abs(grad_of_feats).sum() # 梯度總和
            grad_ratio_of_feats = 1 - (grad_of_feats * fitting_rate / grad_sum_with_abs) # 每個特徵的梯度比率
            if epoch_of_stuck > 0: # 更新停滯，調整grad_ratio_of_feats
                # print("Dude, i'm stucking... get me out of here...")
                queue = np.sort(abs(grad_of_feats)) if epoch_of_stuck == 1 else queue
                mark = queue[-1]
                queue = queue[:-1]
                if mark >= 0:
                    grad_ratio_of_feats = np.where(abs(grad_of_feats) == mark, grad_ratio_of_feats + v, 1)
                else:
                    grad_ratio_of_feats = np.where(abs(grad_of_feats) == mark, grad_ratio_of_feats - v, 1)
            else: # 更新未停滯，調整grad_ratio_of_feats
                grad_ratio_of_feats = np.where(grad_ratio_of_feats >= 1, grad_ratio_of_feats + v, grad_ratio_of_feats - v) # 透過斜率向量v來更新每個特徵的梯度比率。
            # print('curr try:', grad_ratio_of_feats)
            next_formula = curr_formula * grad_ratio_of_feats # 使用使用grad_ratio_of_feats更新formula
            # print("ORI : ", next_formula.T)
            next_formula.clip(self.constrain_df.loc['min'].to_numpy(), self.constrain_df.loc['max'].to_numpy(), axis=1, inplace=True) # 確保formul在限定範圍內
            # print("CLIP: ", next_formula.T)
            # next_formula.clip(0, 1, inplace=True)
            slope = self._slope(next_formula) * np.where(grad_ratio_of_feats != 1, grad_ratio_of_feats, 0) # 透過next_formula更新slope
            loss, grads_tmp = self._grad(self.model, next_formula, target) # 計算next_formula的新梯度和loss
            # print('slope   :', slope)
            # print('momentum:', v)
            # print('gradient:', grad_ratio_of_feats)
            # print(next_formula.T)
            # print(f"epoch: {epoch}, predict: {self.model.predict(next_formula).item()}, loss: {loss.numpy()}")
            if loss + epsilon < best_loss:
                v = np.add(np.where(slope != 1, v * beta, v), slope)
                epoch_of_stuck = 0
                # print(f"***Best loss so far: {loss}")
                best_loss = loss
                grads = grads_tmp
                next_formula.mask(next_formula < 0.001, 0, inplace=True)
                curr_formula = next_formula.copy()
                queue = []
                # print(curr_formula.T)
                # print('  Grad: ', grads[0].numpy().sum(axis=1))
            else:
                if epoch_of_stuck >= 1:
                    if len(queue) == 0:
                        # print(f"Increase fitting rate from {fitting_rate} to {fitting_rate * 2}")
                        fitting_rate *= 2
                        increase_count += 1
                        epoch_of_stuck = 0
                        if increase_count == patient:
                            break
                epoch_of_stuck += 1
                # print(f"loss value {loss} not improve from {best_loss}")

            # 時間到就終止訓練
            end = time.time()
            if ((end - start) > time_limit):
                print("Time up.")
                break
            else:
                epoch += 1

        # print("CONS: ", self.constrain_df)
        # print('#'*100)
        # print(curr_formula.T)
        
        self.inference_formula = curr_formula
        self.inference_target = self.model.predict(curr_formula)
        print(f"epoch = {epoch}")
        print(f"Target   : {target:6.3f}")
        print(f"Inference: {self.inference_target.item():6.3f}, Best loss: {best_loss:8.3f}")        
        print('-'*100)
        # print('#'*100)
        return self.inference_formula, self.inference_target

    # def evaluate(self):
    #     initial_formula_inverse = pd.DataFrame(ms.inverse_transform(self.initial_formula), columns=features)
    #     advisor_formula_inverse = pd.DataFrame(ms.inverse_transform(self.advisor_formula), columns=features)
    #     initial_formula_inverse['Inference_Target'] = self.model.predict(self.initial_formula).item()
    #     advisor_formula_inverse['Inference_Target'] = self.model.predict(self.advisor_formula).item()
    #     initial_formula_t = initial_formula_inverse.T
    #     initial_formula_t['suggest'] = advisor_formula_inverse.T
    #     initial_formula_t.columns = ['initial', 'suggest']
    #     return initial_formula_t
