import os, json
import pandas as pd
from scipy.stats import skew



def box_description(file_name, df, numerical, chart_path):
    description = {}
    for col in numerical:
        Q1   = df[col].quantile(0.25)
        Q3   = df[col].quantile(0.75)
        IQR  = Q3 - Q1
        min_ = Q1 - (1.5 * IQR)
        max_ = Q3 + (1.5 * IQR)    

        Q1_Q3 = f"{Q1} ~ {Q3}"
        count = ((df[col] < min_) | (df[col] > max_)).sum()
        percent = round((count / len(df)) * 100, 2)

        description[col] = {
            "Q1_Q3": Q1_Q3,
            "count": str(count),
            "percent": f"{percent} %"
        }

    descript_path = os.path.join(chart_path, "box", f"{file_name}-outlier.json")
    with open(descript_path, "w") as file:
        json.dump(description, file, indent = 4)



def kde_description(file_name, df, numerical, chart_path):  
    skewness = skew(df[numerical]) # skewness = (3 * (mean - median)) / std
    skewness = pd.DataFrame(skewness, columns = ["skewness"], index = numerical)
    skewness = skewness.query("abs(skewness) > 0.75")
    description = skewness.to_dict()["skewness"]

    descript_path = os.path.join(chart_path, "kde", f"{file_name}-skew.json")
    with open(descript_path, "w") as file:
        json.dump(description, file, indent = 4)



def heatmap_description(file_name, corr, chart_path, target):
    # cut target's correlation to ["micro", "low", "medium", "high"]
    corr = corr.drop(target)
    level = pd.cut(x = abs(corr[target]), bins = [-1, 0.25, 0.5, 0.75, 1], labels = ["micro", "low", "medium", "high"])
    level = level.to_frame().reset_index()
    g = level.groupby(target)
    absolute = {}
    for group, df_group in g:
        absolute[group] = df_group["index"].to_list()

    # find the most 3 relevant features
    relative = abs(corr[target]).nlargest(3)
    relative = pd.Series(relative.index, index = ["1", "2", "3"])
    relative = relative.to_dict()

    description = {
        "absolute": absolute,
        "relative": relative
    }

    descript_path = os.path.join(chart_path, "heatmap", f"{file_name}-correlation.json")
    with open(descript_path, "w") as file:
        json.dump(description, file, indent = 4)