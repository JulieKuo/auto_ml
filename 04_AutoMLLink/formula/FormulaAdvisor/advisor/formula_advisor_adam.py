import numpy as np
import time, os, joblib


class FormulaAdvisorAdam:
    def __init__(self, model_path, constrain_df):
        self.model_path = model_path
        self.constrain_df = constrain_df
        self.model = self.load_model()
    
    def load_model(self):
        model = joblib.load(os.path.join(self.model_path, "mlp", 'model.pkl')) # scikit learn: MLPRegressor 

        return model
    

    def predict(self, X):
        pred = self.model.predict(X)[0] # scikit learn: MLPRegressor 

        return pred
    

    def adjust_X(self, X, h, i):
        '''微幅調整X'''
        X_up = X.copy()
        X_down = X.copy()

        X_up.iloc[0, i] += h
        X_down.iloc[0, i] -= h
        
        return X_up, X_down


    def train(self, target, initial_formula, boundary = 0.1, loss_limit = 500, boundary_limit = 20, time_limit = 60,
              h = 1e-3, learn_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        start = time.time()
        preds = []
        losses = []
        best_loss = np.inf
        boundary_low = target - boundary
        boundary_high = target + boundary
        remain_boundary = boundary_limit
        epoch = 1
        X = initial_formula.copy()
        v = np.zeros(X.shape[1])
        s = np.zeros(X.shape[1])
        while True:
            # 計算所有輸入的梯度
            for i in range(X.shape[1]):
                # 計算梯度: dloss_dx = (loss(x+h) - loss(x-h)) / (2*h) = 微幅調整X時loss的變動量
                X_up, X_down = self.adjust_X(X, h, i)             
                
                loss_up = (target - self.predict(X_up)) ** 2
                loss_down = (target - self.predict(X_down)) ** 2

                dloss_dx = (loss_up - loss_down) / (2 * h)

                # 以Adam的方式更新參數，需先計算v、s
                # v = bata1 * v + (1 - beta1) * dloss_dx  # Momentum: 累積過去梯度，讓跟當前趨勢同方向的輸入有更多的更新，即沿著動量的方向越滾越快
                # s = bata2 * s + (1 - beta2) * (dloss_dx ⊙ dloss_dx) # RMSprop: 累積過去梯度，以獲得輸入被修正程度，修正大的輸入學習率會逐漸變小
                v[i] = (beta1 * v[i]) + ((1 - beta1) * dloss_dx)
                s[i] = beta2 * s[i] + (1 - beta2) * np.multiply(dloss_dx, dloss_dx)

            # 透過梯度計算新的輸入
            # x = x - learning_rate * (1 / ((s + eps) ** (1/2))) * v  # eps: 是極小值，避免s為0時發生除以0的情況
            grad = (learn_rate * (1 / ((s + eps) ** (1/2))) * v)
            new_X = X - grad # 將新輸入暫存在new_X 

            # 確認新輸入是否在25%~75%的分布範圍內，並將不在分布範圍內的新輸入的梯度轉為0，即此次不更新該輸入
            mask = [True if (new_x >= self.constrain_df.iloc[0, c]) and (new_x <= self.constrain_df.iloc[1, c]) else False for c, new_x in enumerate(new_X.values[0])]
            grad *= mask

            # 更新輸入
            X -= grad

            # 查看新預測結果
            # if self.model_type == "h2o":
            #     new_X1 = h2o.H2OFrame(X)
            # else:
            new_X1 = X.copy()
            pred = self.predict(new_X1)
            preds.append(pred)

            # 計算新參數的loss
            loss = (target - pred) ** 2
            losses.append(loss)
            print(f"Epoch {epoch} - loss: {loss:.4f},  predict: {pred:.4f}")

            # 損失函數連續n個epoches都沒下降的話就終止訓練
            if loss < best_loss:
                best_loss = loss
                remain_loss = loss_limit
            else:
                remain_loss -= 1
                if remain_loss == 0:
                    print('Early stop (unable to converge)!')
                    break
                
            # 預測產出達標就終止訓練
            if (pred < boundary_low) or (pred > boundary_high):
                remain_boundary = boundary_limit
            else:
                remain_boundary -= 1
                if remain_boundary == 0:
                    # 預測值是否在可接受範圍內
                    if (pred >= boundary_low) or (pred <= boundary_high):
                        print('Early stop (reach the standard)!')
                        break
                    else: 
                        remain_boundary += 1

            # 時間到就終止訓練
            end = time.time()
            if ((end - start) > time_limit):
                print("Time up.")
                break
            else:
                epoch += 1
            
        print('Done!')
        
        return X, pred

