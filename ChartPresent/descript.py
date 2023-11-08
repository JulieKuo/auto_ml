import os, json
import pandas as pd
from scipy.stats import skew



def box_description(file_name: str, df: pd.DataFrame, numerical: list, chart_path: str):
    # If the value is not between "Q1-(1.5*IQR)" and "Q3+(1.5*IQR)", it's considered an outlier.
    description = {}
    for col in numerical:
        Q1   = df[col].quantile(0.25)
        Q3   = df[col].quantile(0.75)
        IQR  = Q3 - Q1
        min_ = Q1 - (1.5 * IQR)
        max_ = Q3 + (1.5 * IQR)    

        Q1_Q3   = f"{Q1} ~ {Q3}"
        count   = ((df[col] < min_) | (df[col] > max_)).sum()
        percent = round((count / len(df)) * 100, 2)

        description[col] = {
            "Q1_Q3":   Q1_Q3,
            "count":   str(count),
            "percent": f"{percent} %"
        }

    # save description
    descript_path = os.path.join(chart_path, "box", f"{file_name}-outlier.json")
    with open(descript_path, "w", encoding = 'utf-8') as file:
        json.dump(description, file, indent = 4, ensure_ascii = False)



def kde_description(file_name: str, df: pd.DataFrame, numerical: list, chart_path: str):
    # If the skewness value exceeds +-0.75, it's considered skewed.
    skewness    = skew(df[numerical]) # skewness = (3 * (mean - median)) / std
    skewness    = pd.DataFrame(skewness, columns = ["skewness"], index = numerical)
    skewness    = skewness.query("abs(skewness) > 0.75")
    description = skewness.to_dict()["skewness"]

    # save description
    descript_path = os.path.join(chart_path, "kde", f"{file_name}-skew.json")
    with open(descript_path, "w", encoding = 'utf-8') as file:
        json.dump(description, file, indent = 4, ensure_ascii = False)



def heatmap_description(file_name: str, corr: pd.DataFrame, chart_path: str, target: str):
    # cut target's correlation to ["micro", "low", "medium", "high"]
    corr  = corr.drop(target) # drop the target row to remove the target's own correlation
    level = pd.cut(x = abs(corr[target]), bins = [-1, 0.25, 0.5, 0.75, 1], labels = ["micro", "low", "medium", "high"]) # cut target's correlation
    level = level.to_frame().reset_index() # series to dataframe
    
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

    # save description
    descript_path = os.path.join(chart_path, "heatmap", f"{file_name}-correlation.json")
    with open(descript_path, "w", encoding = 'utf-8') as file:
        json.dump(description, file, indent = 4, ensure_ascii = False)