import os, shutil, sys, base64, json
import pandas as pd


def get_input() -> dict:    
    input_ = sys.argv[1] # get parameter
    input_ = base64.b64decode(input_).decode("utf-8") # decode base64
    input_ = json.loads(input_) # Convert string to json format

    return input_


def get_path(input_: dict, config: dict):
    file_names = input_["fileNames"] 
    file_paths, parser_paths = [], []
    for file_name in file_names:
        # training & testing data
        file_path = os.path.join(config["GROUP_PROJECT_FOLDER"], input_["groupId"], input_["projectId"], "Tabular", "AfterLabelMerge", file_name)
        file_paths.append(file_path)

        # the information of features of training & testing data
        parser_path = os.path.join(config["GROUP_PROJECT_FOLDER"], input_["groupId"], input_["projectId"], "Tabular", "AfterLabelMerge", "ParserResult", file_name.replace(".csv", ".json"))
        parser_paths.append(parser_path)
    
    return file_names, file_paths, parser_paths


def create_folder(input_: dict, config: dict) -> str:
    chart_path = os.path.join(config["GROUP_PROJECT_FOLDER"], input_["groupId"], input_["projectId"], "ChartPresent")
    charts = ["missing_value", "heatmap", "count", "box", "kde", "kde_dataset", "adversarial"] # folders that need to be created
    for chart in charts:
        path = os.path.join(chart_path, chart) # folder path
        
        if os.path.exists(path): # delete old folder
            shutil.rmtree(path)
        
        os.makedirs(path, exist_ok = True) # create folder
    
    return chart_path


def get_df_feat(file_paths: list, parser_paths: list):
    dfs, numericals, categories = [], [], []
    for file_path, parser_path in zip(file_paths, parser_paths):
        # get parser data
        with open(parser_path) as f:
            feat_info = json.load(f)

        # get all numerical and category
        numerical, category = [], []
        for info in feat_info.values():
            if info["advise_dtype"] == "numerical":
                numerical.append(info["column_name"])
            else:
                category.append(info["column_name"])

        numericals.append(numerical)
        categories.append(category)

        # get all df
        df = pd.read_csv(file_path)
        df[category] = df[category].astype(object)
        dfs.append(df)

    return dfs, numericals, categories