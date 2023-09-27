from plot import *
from tools import *
from traceback import format_exc
from log_config import Log
from datetime import datetime
import os, json, sys, base64, redis



# get root path
current = os.path.abspath(__file__)
root    = os.path.dirname(current)

log = Log()
log_path = os.path.join(root, "logs")
os.makedirs(log_path, exist_ok = True)
date = datetime.now().strftime("%Y%m%d")
logging = log.set_log(filepath = os.path.join(log_path, f"{date}.log"), level = 2, freq = "D", interval = 1, backup = 0, name = "log")
logging.info("-" * 200)


try:
    # get parameters
    input_ = sys.argv[1]
    input_ = base64.b64decode(input_).decode("utf-8")
    input_ = json.loads(input_)
    logging.info(f"input = {input_}")


    # get config data
    with open(os.path.join(root, "config.json")) as f:
        config = json.load(f)


    # set redis to pass progress
    r = redis.Redis(host = config["REDIS_CONFIG"]["host"], port = config["REDIS_CONFIG"]["port"])
    

    # get files path and parsers path
    file_names, file_paths, parser_paths = get_path(input_, config)
    

    # create folders to save charts
    chart_path = create_folder(input_, config)
    logging.info(f"save charts to {chart_path}")
    

    # get datas and features
    dfs, numericals, categories = get_df_feat(file_paths, parser_paths)
    

    if (len(file_names) == 2):
        assert ((numericals[0] == numericals[1]) and (categories[0] == categories[1])), "The features of the two datasets are different."


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
    logging.info("create charts...")
    target = input_["target"]
    top = 30
    progress = 0
    progress_gap = (1 / 13) if (len(file_names) == 2) else (1 / 8)
    for file_name, df, numerical, category in zip(file_names, dfs, numericals, categories):
        missing_value(file_name, df, top, chart_path)
        progress += progress_gap
        r.set("ChartPresent_percent", round(progress, 2))

        heatmap(file_name, df, numerical, top, chart_path, target)
        progress += progress_gap
        r.set("ChartPresent_percent", round(progress, 2))

        count(file_name, df, category, top, chart_path)
        progress += progress_gap
        r.set("ChartPresent_percent", round(progress, 2))
        
        box(file_name, df, numerical, chart_path)
        progress += progress_gap
        r.set("ChartPresent_percent", round(progress, 2))

        kde(file_name, df, numerical, chart_path)
        progress += progress_gap
        r.set("ChartPresent_percent", round(progress, 2))

    if (len(file_names) == 2):
        kde_dataset(file_names, dfs, numericals[0], chart_path)
        progress += progress_gap
        r.set("ChartPresent_percent", round(progress, 2))

        if (len(dfs[0]) != len(dfs[1])) or (not (dfs[0] == dfs[1]).all().all()): # different datasets
            adversarial(dfs, categories[0], chart_path)
            progress += progress_gap
            r.set("ChartPresent_percent", round(progress, 2))


    
    result = {
        "status":     "success",
        "chart_path": chart_path
        }



except AssertionError as e:
    logging.error(e)
    result = {
        "status":     "fail",
        "chart_path": chart_path,
        "reason":     e.args[0]
        }



except:
    logging.error(format_exc())
    result = {
        "status":     "fail",
        "chart_path": chart_path,
        "reason":     format_exc()
        }



finally:
    result_json = os.path.join(root, "result.json")
    logging.info(f"Save result to {result_json}")
    with open(result_json, "w") as file:
        json.dump(result, file, indent = 4)
        
    r.set("ChartPresent_percent", 1)
    
    log.shutdown()