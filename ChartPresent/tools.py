import os, shutil, json
import pandas as pd


def get_path(input_, config):
    file_names = input_["fileNames"]
    file_paths, parser_paths = [], []
    for file_name in file_names:
        file_path = os.path.join(config["GROUP_PROJECT_FOLDER"], input_["groupId"], input_["projectId"], "Tabular", "AfterLabelMerge", file_name)
        file_paths.append(file_path)

        parser_path = os.path.join(config["GROUP_PROJECT_FOLDER"], input_["groupId"], input_["projectId"], "Tabular", "AfterLabelMerge", "ParserResult", file_name.replace(".csv", ".json"))
        parser_paths.append(parser_path)
    
    return file_names, file_paths, parser_paths


def create_folder(input_, config):
    chart_path = os.path.join(config["GROUP_PROJECT_FOLDER"], input_["groupId"], input_["projectId"], "ChartPresent")
    charts = ["missing_value", "heatmap", "count", "box", "kde", "kde_dataset", "adversarial"]
    for chart in charts:
        path = os.path.join(chart_path, chart)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok = True)
    
    return chart_path


def get_df_feat(file_paths, parser_paths):
    dfs, numericals, categories = [], [], []
    for file_path, parser_path in zip(file_paths, parser_paths):
        with open(parser_path) as f:
            feat_info = json.load(f)

        numerical, category = [], []
        for info in feat_info.values():
            if info["advise_dtype"] == "numerical":
                numerical.append(info["column_name"])
            else:
                category.append(info["column_name"])

        numericals.append(numerical)
        categories.append(category)


        df = pd.read_csv(file_path)
        df[category] = df[category].astype(object)
        dfs.append(df)

    return dfs, numericals, categories