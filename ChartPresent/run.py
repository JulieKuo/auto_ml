import pandas as pd
from plot import *
from traceback import format_exc
from log_config import Log
import os, json, sys, base64



# 取得根目錄
current_path = os.path.abspath(__file__)
root    = os.path.dirname(current_path)

log = Log()
log_path = os.path.join(root, "logs")
os.makedirs(log_path, exist_ok = True)
logging = log.set_log(filepath = os.path.join(log_path, "log.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "log")
logging.info("-"*200)


try:
    # get parameters
    input_ = sys.argv[1]
    input_ = base64.b64decode(input_).decode('utf-8')
    input_ = json.loads(input_)
    logging.info(f"input = {input_}")


    # get config data
    with open(os.path.join(root, "config.json")) as f:
        config = json.load(f)
    

    # get files path and parsers path
    file_names = input_["fileNames"]
    file_paths, parser_paths = [], []
    for file_name in file_names:
        file_path = os.path.join(config['GROUP_PROJECT_FOLDER'], input_['groupId'], input_['projectId'], "Tabular", "AfterLabelMerge", file_name)
        file_paths.append(file_path)

        parser_path = os.path.join(config['GROUP_PROJECT_FOLDER'], input_['groupId'], input_['projectId'], "Tabular", "AfterLabelMerge", "ParserResult", file_name.replace(".csv", ".json"))
        parser_paths.append(parser_path)
    

    # create folders to save charts
    chart_path = os.path.join(config['GROUP_PROJECT_FOLDER'], input_['groupId'], input_['projectId'], "ChartPresent", input_['job_id'])
    charts = ["missing_value", "heatmap", "count", "box", "kde", "kde_dataset", "adversarial"]
    for chart in charts:
        os.makedirs(os.path.join(chart_path, chart), exist_ok = True)    
    logging.info(f'save charts to {chart_path}')
    

    # get datas and features
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


    # TODO: delete
    feats = ["1_Turbine_Negative_Pressure.1", "2_Turbine_Negative_Pressure.1", "3_Vacuum_Pump_Motor_Side_Vibration", "4_Turbine_Negative_Pressure.1"]
    for feat in feats:
        numericals[0].remove(feat)
        numericals[1].remove(feat)
        categories[0].append(feat)
        categories[1].append(feat)
    dfs[0][feats] = dfs[0][feats].astype(object)
    dfs[1][feats] = dfs[1][feats].astype(object)


    # create charts
    logging.info('create charts...')
    top = 30
    for file_name, df, numerical, category in zip(file_names, dfs, numericals, categories):
        missing_value(file_name, df, top, chart_path)
        heatmap(file_name, df, numerical, top, chart_path)
        count(file_name, df, category, top, chart_path)
        box(file_name, df, numerical, chart_path)
        kde(file_name, df, numerical, chart_path)

    if (numericals[0] == numericals[1]) and (categories[0] == categories[1]): # 特徵相同
        kde_dataset(file_names, dfs, numericals[0], chart_path)
        if (len(dfs[0]) != len(dfs[1])) or (not (dfs[0] == dfs[1]).all().all()): # 資料集不相同
            adversarial(dfs, categories[0], chart_path)

    
    result = {
        "status": "success",
        "chart_path": chart_path
        }



except:
    logging.error(format_exc())
    result = {
        "status": "fail",
        "chart_path": chart_path,
        "reason": format_exc()
        }



finally:    
    result_json = os.path.join(root, "result.json")
    logging.info(f'Save result to {result_json}')
    with open(result_json, 'w') as file:
        json.dump(result, file, indent = 4)    
    
    log.shutdown()