import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from xgboost import DMatrix, XGBClassifier, plot_importance, cv
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from descript import *
import os
plt.rcParams["font.sans-serif"] = ["Taipei Sans TC Beta"]



def missing_value(file_name: str, df: pd.DataFrame, top: int, chart_path: str):
    # Calculate how many charts are needed when placing n features from each chart. (n = top)
    num = df.shape[1] // top
    num += min(df.shape[1] % top, 1)

    # set folder path
    distribution_path = os.path.join(chart_path, "missing_value", "distribution")
    count_path        = os.path.join(chart_path, "missing_value", "count")
    os.makedirs(distribution_path, exist_ok = True)
    os.makedirs(count_path, exist_ok = True)

    # generate charts
    for i in range(1, num + 1):
        # features' range
        start = (i - 1) * top
        end   = min(i * top, df.shape[1])
        
        # create distribution chart
        fig, ax = plt.subplots(1, 1, figsize = (20, 10), constrained_layout = True)
        msno.matrix(df.iloc[:, start:end], color = (0.4, 0.7, 1), sparkline = False, ax = ax)   
        title = f"{i}-{file_name}-缺失值分佈"
        fig.suptitle(title, fontsize = 18)
        fig.savefig(os.path.join(distribution_path, f"{title}.png"))
        plt.close() # reduce memory usage

        # create count chart
        fig, ax = plt.subplots(1, 1, figsize = (20, 10), constrained_layout = True)
        msno.bar(df.iloc[:, start:end], color = "#66B3FF", ax = ax) 
        title = f"{i}-{file_name}-缺失值數量"
        fig.suptitle(title, fontsize = 18)
        fig.savefig(os.path.join(count_path, f"{title}.png"))
        plt.close()



def heatmap(file_name: str, df: pd.DataFrame, numerical: list, top: int, chart_path: str, target: str):
    # calculate correlation
    corr = df[numerical].iloc[:, :top].corr().round(2)

    # generate charts
    length, width = max(10, corr.shape[1] / 3 * 2), max(5, corr.shape[1] / 2)
    plt.figure(figsize = (length, width), constrained_layout = True)
    sns.heatmap(data = corr, annot = True, cmap = "RdBu")
    title = f"{file_name}-特徵相關性"
    plt.title(title)
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.savefig(os.path.join(chart_path, "heatmap", f"{title}.png"))
    plt.close()

    # generate description
    if target in numerical:
        heatmap_description(file_name, corr, chart_path, target)



def count(file_name: str, df: pd.DataFrame, category: list, top: int, chart_path: str):
    for i, feat in enumerate(category):
        # Get the top n categories in a feature. (n = top)
        top_cat = df[feat].value_counts().iloc[:top].index
        df1     = df[df[feat].isin(top_cat)]

        
        # generate charts
        length, width = max(6.5, len(category) / 2), max(4.75, len(category) / 3)
        plt.figure(figsize = (length, width), constrained_layout = True)
        sns.countplot(data = df1, x = feat, width = 0.5)
        title = f"{i+1}-{file_name}-類別分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "count", f"{title}.png"))
        plt.close()



def box(file_name: str, df: pd.DataFrame, numerical: list, chart_path: str):
    # generate charts
    for i, feat in enumerate(numerical):
        plt.figure(constrained_layout = True)
        sns.boxplot(data = df, y = feat, width = 0.5)
        title = f"{i+1}-{file_name}-數值分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "box", f"{title}.png"))
        plt.close()    
    
    # generate description
    box_description(file_name, df, numerical, chart_path)



def kde(file_name: str, df: pd.DataFrame, numerical: list, chart_path: str):
    # generate charts
    for i, feat in enumerate(numerical):
        plt.figure(constrained_layout = True)
        sns.kdeplot(data = df, x = feat, fill = True)
        title = f"{i+1}-{file_name}-數值分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "kde", f"{title}.png"))
        plt.close()
    
    # generate description
    kde_description(file_name, df, numerical, chart_path)



def kde_dataset(file_names: list, dfs: list, numerical: list, chart_path: str):
    # concat two datasets
    df0 = dfs[0].copy()
    df1 = dfs[1].copy()

    df0["Dataset"] = file_names[0]
    df1["Dataset"] = file_names[1]
    df = pd.concat([df0, df1], ignore_index = True)

    # generate charts
    for i, feat in enumerate(numerical):
        plt.figure(constrained_layout = True)
        sns.kdeplot(data = df, x = feat, hue = "Dataset", fill = True)
        title = f"{i+1}-資料集分佈-{feat}"
        plt.title(title)
        plt.savefig(os.path.join(chart_path, "kde_dataset", f"{title}.png"))
        plt.close()



def adversarial(dfs: list, category: list, chart_path: str):
    # concat two datasets to do category encoding
    train = dfs[0].copy()
    test  = dfs[1].copy()
    df    = pd.concat([train, test])

    # category encoding
    for col in category:
        label_encoder = LabelEncoder()
        df[col]    = label_encoder.fit_transform(df[col])
        train[col] = label_encoder.transform(train[col])
        test[col]  = label_encoder.transform(test[col])

    # add train and test label
    train["AV"] = 0
    test["AV"]  = 1

    # concat train and test
    df1 = pd.concat([train, test], ignore_index = True)

    # shuffle data
    df1_shuffled = df1.sample(frac = 1)

    # create XGBoost data structure
    X = df1_shuffled.drop(["AV"], axis = 1)
    y = df1_shuffled["AV"]
    XGBdata = DMatrix(data = X, label = y)

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

    # train model and get feature importance
    classifier = XGBClassifier(eval_metric = "logloss")
    classifier.fit(X, y)

    # generate a chart
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout = True)
    title = f"分佈不同的特徵\nadversarial_score: {accuracy} (越接近0.5分佈越相似)"
    plot_importance(classifier, max_num_features = 10, height  = 0.5, title = title, ax = ax)
    fig.savefig(os.path.join(chart_path, "adversarial", f"adversarial.png"))
    plt.close()