import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
plt.rcParams["font.sans-serif"] = ["Taipei Sans TC Beta"]


def missing_value(file_name, df, top, chart_path):
    # 計算需要每張圖放top個features共需幾張圖
    num = df.shape[1] // top
    num += min(df.shape[1] % top, 1)

    distribution_path = os.path.join(chart_path, "missing_value", "distribution")
    count_path        = os.path.join(chart_path, "missing_value", "count")
    os.makedirs(distribution_path, exist_ok = True)
    os.makedirs(count_path, exist_ok = True)

    for i in range(1, num + 1):
        start = (i - 1) * top
        end   = min(i * top, df.shape[1])
        
        fig, ax = plt.subplots(1, 1, figsize = (20, 10), constrained_layout = True)
        msno.matrix(df.iloc[:, start:end], color = (0.4, 0.7, 1), sparkline = False, ax = ax)   
        title = f"{i}-{file_name}-缺失值分佈"
        fig.suptitle(title, fontsize = 18)
        fig.savefig(os.path.join(distribution_path, f"{title}.png"))
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize = (20, 10), constrained_layout = True)
        msno.bar(df.iloc[:, start:end], color = "#66B3FF", ax = ax) 
        title = f"{i}-{file_name}-缺失值數量"
        fig.suptitle(title, fontsize = 18)
        fig.savefig(os.path.join(count_path, f"{title}.png"))
        plt.close()



def heatmap(file_name, df, numerical, top, chart_path):
    corr = df[numerical].iloc[:, :top].corr().round(2)

    length, width = max(10, corr.shape[1] / 3 * 2), max(5, corr.shape[1] / 2)
    plt.figure(figsize = (length, width), constrained_layout = True)
    sns.heatmap(data = corr, annot = True, cmap = "RdBu")
    title = f"{file_name}-特徵相關性"
    plt.title(title)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.savefig(os.path.join(chart_path, "heatmap", f"{title}.png"))
    plt.close()



def count(file_name, df, category, top, chart_path):
    for i, feat in enumerate(category):
        top_cat = df[feat].value_counts().iloc[:top].index
        df1     = df[df[feat].isin(top_cat)]

        length, width = max(6.5, len(category) / 2), max(4.75, len(category) / 3)
        plt.figure(figsize = (length, width), constrained_layout = True)
        sns.countplot(data = df1, x = feat, width = 0.5)
        title = f"{i+1}-{file_name}-類別分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "count", f"{title}.png"))
        plt.close()



def box(file_name, df, numerical, chart_path):
    for i, feat in enumerate(numerical):
        plt.figure(constrained_layout = True)
        sns.boxplot(data = df, y = feat, width = 0.5)
        title = f"{i+1}-{file_name}-數值分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "box", f"{title}.png"))
        plt.close()



def kde(file_name, df, numerical, chart_path):
    for i, feat in enumerate(numerical):
        plt.figure(constrained_layout = True)
        sns.kdeplot(data = df, x = feat, fill = True)
        title = f"{i+1}-{file_name}-數值分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "kde", f"{title}.png"))
        plt.close()



def kde_dataset(file_names, dfs, numerical, chart_path):
    df0 = dfs[0].copy()
    df1 = dfs[1].copy()

    df0["Dataset"] = file_names[0]
    df1["Dataset"] = file_names[1]
    df = pd.concat([df0, df1], ignore_index = True)

    for i, feat in enumerate(numerical):
        plt.figure(constrained_layout = True)
        sns.kdeplot(data = df, x = feat, hue = "Dataset", fill = True)
        title = f"{i+1}-資料集分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "kde_dataset", f"{title}.png"))
        plt.close()



def adversarial(dfs, category, chart_path):
    train = dfs[0].copy()
    test  = dfs[1].copy()
    df    = pd.concat([train, test])

    # category encoding
    for col in category:
        label_encoder = LabelEncoder()
        df[col]    = label_encoder.fit_transform(df[col])
        train[col] = label_encoder.transform(train[col])
        test[col]  = label_encoder.transform(test[col])

    # add train, test label
    train["AV"] = 0
    test["AV"]  = 1

    # merge train, test
    df1 = pd.concat([train, test], ignore_index = True)

    # shuffle data
    df1_shuffled = df1.sample(frac = 1)

    # create XGBoost data structure
    X = df1_shuffled.drop(["AV"], axis = 1)
    y = df1_shuffled["AV"]
    XGBdata = xgb.DMatrix(data = X, label = y)

    # XGBoost parameters
    params = {
        "objective":   "binary:logistic",
        "eval_metric": "logloss",
        }

    # perform cross validation with XGBoost
    cross_val_results = cv(
        dtrain  = XGBdata,
        params  = params, 
        metrics = "auc", 
        )
    accuracy = cross_val_results["test-auc-mean"].iloc[-1].round(4)


    # feature importance
    classifier = XGBClassifier(eval_metric = "logloss")
    classifier.fit(X, y)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout = True)
    title = f"分佈不同的特徵\nadversarial_score: {accuracy} (越接近0.5分佈越相似)"
    plot_importance(classifier, max_num_features = 10, height  = 0.5, title = title, ax = ax)
    fig.savefig(os.path.join(chart_path, "adversarial", f"adversarial.png"))
    plt.close()