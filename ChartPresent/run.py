from plot import *
from tool import *
from traceback import format_exc
from log_config import Log
from datetime import datetime
import os, json, sys, redis



class visualize():
    def __init__(self):        
        # get root path
        current   = os.path.abspath(__file__)
        self.root = os.path.dirname(current)


        # initialize log
        self.log = Log()

        # create log folder
        log_path = os.path.join(self.root, "logs")
        os.makedirs(log_path, exist_ok = True)

        # set log
        date = datetime.now().strftime("%Y%m%d")
        self.logging = self.log.set_log(filepath = os.path.join(log_path, f"{date}.log"), level = 2, freq = "D", interval = 1, backup = 0, name = "log")



    def get_basic_info(self):
        # get parameters
        assert (len(sys.argv) == 2), "Input parameter missing." # Check if there are parameters
        self.input_ = get_input()
        self.logging.info(f"input = {self.input_}")


        # get config data
        with open(os.path.join(self.root, "config.json")) as f:
            config = json.load(f)


        # set redis to pass progress
        self.r = redis.Redis(host = config["REDIS_CONFIG"]["host"], port = config["REDIS_CONFIG"]["port"])
        

        # get files path and parsers path
        file_names, file_paths, parser_paths = get_path(self.input_, config)
        

        # create folders to save charts
        chart_path = create_folder(self.input_, config)
        self.logging.info(f"save charts to {chart_path}")


        return file_names, file_paths, parser_paths, chart_path
    
    

    def main(self, redis_key = "ChartPresent_percent"):
        try:
            self.logging.info("-" * 200)

            # get basic information
            file_names, file_paths, parser_paths, chart_path = self.get_basic_info()
            

            # get datas and features
            dfs, numericals, categories = get_df_feat(file_paths, parser_paths)
            

            # check whether the features of two datasets are the same
            if (len(file_names) == 2):
                assert ((numericals[0] == numericals[1]) and (categories[0] == categories[1])), "The features of two datasets are different."


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
            self.logging.info("create charts...")
            target = self.input_["target"]
            top = 30
            progress = 0
            progress_gap = (1 / 13) if (len(file_names) == 2) else (1 / 6) # how many "progress_gaps" need to be added to "progress"
            for file_name, df, numerical, category in zip(file_names, dfs, numericals, categories):
                missing_value(file_name, df, top, chart_path) # For all features.
                progress += progress_gap
                self.r.set(redis_key, round(progress, 2))

                heatmap(file_name, df, numerical, top, chart_path, target) # For numerical features. Used to know feature correlation.
                progress += progress_gap
                self.r.set(redis_key, round(progress, 2))

                count(file_name, df, category, top, chart_path) # For categorical features
                progress += progress_gap
                self.r.set(redis_key, round(progress, 2))
                
                box(file_name, df, numerical, chart_path) # For numerical features. Used to detect outliers.
                progress += progress_gap
                self.r.set(redis_key, round(progress, 2))

                kde(file_name, df, numerical, chart_path) # For numerical features. Used to detect skewness.
                progress += progress_gap
                self.r.set(redis_key, round(progress, 2))

            # Compare two datasets
            if (len(file_names) == 2):
                kde_dataset(file_names, dfs, numericals[0], chart_path) # For numerical features. Used to know feature distribution of two datasets.
                progress += progress_gap
                self.r.set(redis_key, round(progress, 2))

                # if these two datasets are different
                if (len(dfs[0]) != len(dfs[1])) or (not (dfs[0] == dfs[1]).all().all()):
                    adversarial(dfs, categories[0], chart_path) # For all features. Used to know feature distribution of two datasets.
                    progress += progress_gap
                    self.r.set(redis_key, round(progress, 2))

            
            result = {
                "status":     "success",
                "chart_path": chart_path
                }


        except AssertionError as e:
            self.logging.error(e)
            result = {
                "status":     "fail",
                "chart_path": chart_path,
                "reason":     e.args[0]
                }


        except:
            self.logging.error(format_exc())
            result = {
                "status":     "fail",
                "chart_path": chart_path,
                "reason":     format_exc()
                }


        finally:
            result_json = os.path.join(self.root, "result.json")
            self.logging.info(f"Save result to {result_json}")
            with open(result_json, "w", encoding = 'utf-8') as file:
                json.dump(result, file, indent = 4, ensure_ascii = False)
                
            self.r.set(redis_key, 1) 
            self.log.shutdown()



if __name__ == '__main__':
    visualize().main()