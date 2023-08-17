from AutoMLInterface import AutoMLInterface
from customized_exception import raiseCustomizedException
from dataframe_to_image import save_dataframe_as_image
from datetime import datetime
import glob
import h2o
from h2o.automl import H2OAutoML
import json
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import os
import pandas as pd
from pathlib import Path
from shutil import copyfile
import sys
from chart import save_explain_plots
# import sklearn.metrics
# from os import path, makedirs, environ
# import dataframe_image as dfi


FLAG_DEBUG_MODE = False
# FLAG_SAVE_MODEL = True

MODEL_LIST_FILENAME = 'model_list.txt'
TRAIN_INFO_FILENAME = 'train_info.json'

TRAINING_PARAM_FILEPATH = '/home/stadmin/AIPlatform/ExecutiveFile/data/H2O_train_param.json'
# TRAINING_PARAM_FILEPATH = './H2O_train_param.json'

# TODO
# run classification on numeric target
# unsupervised
# generate more graph

class stdout_handler():
    def __init__(self, platform_msg_handler, key):
        self._platform_msg_handler = platform_msg_handler
        self._key = key
        self.reset()

    def write(self, txt):
        if len(txt):
            self._count += txt.count('\u2588')
            # <1> update redis
            # if '100%' in txt:
            #     self._flag = 'Complete'
            #     self._platform_msg_handler.set_value('20201104_percent', 1.0)
            # else:
            self._platform_msg_handler.set_value(self._key, self._count/100)
            # <2> write to file
            if FLAG_DEBUG_MODE:
                with open('./redirect_stdout.txt', 'a') as f:
                    if not FLAG_DEBUG_MODE:
                        f.write(txt)
                    else:
                        f.write('{}|{}|{}\n'.format(type(txt), len(txt), txt))
    
    def reset(self):
        self._flag = True
        self._count = 0

    def flush(self):
        pass

def get_current_time():
    return datetime.now().strftime("%Y/%m/%d, %H:%M:%S")

def ranking_zfill(ranking):
    return str(ranking).zfill(2)

def create_folder(folderpath):
    Path(folderpath).mkdir(parents=True, exist_ok=True)
    return

def check_filepath(filepath_list):
    for filepath in filepath_list:
        if not Path(filepath).exists():
            raiseCustomizedException('9199@File '+filepath+' does not exist')

class H2O_AutoMLInterface(AutoMLInterface):
    def __init__(self, platform_msg_handler, redisKey, my_logger):
        self._platform_msg_handler = platform_msg_handler
        self._redisKey_percent = redisKey + '_percent'
        self._redisKey_model_type = redisKey + '_model_type'
        self._redisKey_accuracy = redisKey + '_accuracy'
        self._my_logger = my_logger
        self._flag_regression = False
        # jar_path = environ['APPDATA'] + r'\Python\h2o_jar'
        # if not path.exists(jar_path):
        #     makedirs(jar_path)
        #     copyfile('h2o.jar', jar_path + r'\h2o.jar')
        h2o.init()
        self._my_logger.write_info('h2o initialized.')
        if FLAG_DEBUG_MODE:
            print('\nRedisKey: {}\n'.format(redisKey))

    def set_percent(self, value):
        self._platform_msg_handler.set_value(self._redisKey_percent, value)
        
    def set_model_type(self, value):
        self._platform_msg_handler.set_value(self._redisKey_model_type, value)

    def set_accuracy(self, value):
        self._platform_msg_handler.set_value(self._redisKey_accuracy, value)
    
    @staticmethod
    def read_with_col_types(filepath):
        # filepath = filepath.replace('\\', '/')
        # filename = filepath.split('/')[-1]
        # AF_job_id = filename.split('_')[0]
        
        # AF_file_pattern = os.path.join('/'.join(filepath.split('/')[:-3]), 'FileHeaderAndType', AF_job_id).replace('\\', '/') + '_*'
        # result = glob.glob(AF_file_pattern)
        
        # col_types = None
        # if len(result):
        #     with open(result[0], 'r') as f:
        #         conf = json.load(f)
        #     col_types = conf['col_types']

        # df = h2o.import_file(filepath, col_types=col_types)
        df = h2o.import_file(filepath, header=1)
        return df

    def train(self, filepath, target, models_folderpath, max_runtime_secs=120):
        self._start_time = get_current_time()
        self.setup_before_train(filepath, target, models_folderpath, max_runtime_secs)

        # start training
        ## redirect stdout
        self._my_logger.write_info('start training')
        backup_stdout = sys.stdout
        new_out = stdout_handler(self._platform_msg_handler, self._redisKey_percent)
        sys.stdout = new_out
        # self._aml = H2OAutoML(max_runtime_secs=self._MAX_RUNTIME_SECS, max_models=self._MAX_MODELS, seed=self._SEED)
        self._aml = H2OAutoML(**self._param)
        self._aml.train(x=self._features,
                        y=self._target,
                        # validation_frame=self._valid,
                        training_frame=self._train)
        self._model = self._aml.leader
        self._my_logger.write_info('auto find end')
        sys.stdout = backup_stdout
        self._my_logger.write_info('restore console')
        self.set_percent(0.8)
        self._my_logger.write_info('training completed')
        
        # set then save information needed
        self.setup_after_train()
        self.set_percent(0.9)

        self.save_after_train()
        self.set_percent(1.0)
    
    def predict(self, filepath, models_folderpath, output_folderpath, flag_save_data_as_well=False):
        start_time = get_current_time()
        self._models_folderpath = models_folderpath    # for setup_model_info()
        # num_of_model, target = self.get_info_needed()
        self.setup_before_predict()
        # test = h2o.import_file(filepath)
        test = self.read_with_col_types(filepath)
        flag_data_with_answer = self._target in test.columns
        
        if FLAG_DEBUG_MODE:
            print('num of model:', self._total_model)
            print('test_data.shape:', test.shape)
            print('flag_data_with_answer:', flag_data_with_answer)

        if flag_data_with_answer:
            run_model = self._total_model
        else:
            run_model = 1    # run prediction with the leader model only
        
        prediction_steps = 3
        each_step_progress = 1.0 / (2 + prediction_steps * run_model)
        
        best_accuracy = -1
        best_model_ranking = None
        best_model_performance = None
        best_prediction = None
        best_metrics = None

        self.set_percent(each_step_progress)
        self._my_logger.write_info('start running prediction on {} model'.format(run_model))
        accuracy_str_list = []

        for ranking in range(1, 1+run_model):
            # 1.1. setup model_path, type (accuracy or r2 in get_metrics()) and num_of_class, then load model
            # self.setup_model_info(ranking)
            model_path = self.get_model_path(models_folderpath, ranking)
            saved_model = h2o.load_model(model_path)
            self.set_percent((1 + prediction_steps*(ranking-1) + 1) * each_step_progress)
                
            # 2. make prediction
            y_pred_aml = saved_model.predict(test)
            self.set_percent((1 + prediction_steps*(ranking-1) + 2) * each_step_progress)

            # 3. get metrics, keep the best result
            if flag_data_with_answer:
                if self._class_count == 2:
                    perf = saved_model.model_performance(test)
                    accuracy = (test[self._target] == y_pred_aml['predict']).sum() / len(test)
                    metrics = 'accuracy'
                else:
                    perf = saved_model.model_performance(test)
                    metrics, accuracy = self.get_metrics(perf)
                # self.save_confusion_matrix(perf, output_folderpath)
                # accuracy = metrics['accuracy'] if 'accuracy' in metrics.keys() else metrics['r2']
                if FLAG_DEBUG_MODE:
                    print('ranking: {}\taccuracy: {}'.format(ranking, accuracy))
                if accuracy > best_accuracy:
                    best_model_ranking = ranking
                    best_model_performance = perf
                    best_prediction = y_pred_aml[0]
                    best_accuracy = accuracy
                    best_metrics = metrics
                    best_model_path = model_path
                accuracy_str_list.append(str(round(accuracy,3)))
            else:
                best_metrics = {'accuracy': '---', 'r2': '---'}
                best_accuracy = '---'
                best_prediction = y_pred_aml[0]
                best_model_ranking = ranking
                best_model_path = model_path
                # # run prediction with the leader model only
                # break
            self.set_percent((1 + prediction_steps*(ranking-1) + 3) * each_step_progress)

        self._my_logger.write_info('{}\nFinished. best model on this test data is #{}'.format(','.join(accuracy_str_list), best_model_ranking))
        # save the best prediction
        create_folder(output_folderpath)
        if flag_save_data_as_well:
            result = best_prediction.as_data_frame()
        else:
            result = best_prediction.as_data_frame()
        result = pd.concat([test.as_data_frame(), result], axis=1)
        result.to_csv(output_folderpath+'prediction.csv', index=False)

        if flag_data_with_answer:
            self.save_confusion_matrix(best_model_performance, output_folderpath)
            self.save_model_performance(best_model_performance, output_folderpath)
            self.set_accuracy(best_accuracy)    # don't add this redis key if there is no corresponding target column.
        self.save_prediction_info({'test_filepath': filepath, 'test_data_shape': test.shape, 'start_time': start_time, 'end_time': get_current_time(), 'ranking': best_model_ranking, 'metrics': best_metrics, 'model_path': best_model_path}, output_folderpath)
        # complete.
        self.set_percent(1.0)

    def setup_before_train(self, filepath, target, models_folderpath, max_runtime_secs):
        self._models_folderpath = models_folderpath
        self._model_list_filepath = models_folderpath + MODEL_LIST_FILENAME
        self._train_filepath = filepath

        # setup
        check_filepath([TRAINING_PARAM_FILEPATH, filepath])
        create_folder(models_folderpath)
        
        with open(TRAINING_PARAM_FILEPATH, 'r') as f:
            self._param = json.load(f)
        self._param['max_runtime_secs'] = max_runtime_secs
        
        # self._train = h2o.import_file(filepath)
        self._train = self.read_with_col_types(filepath)
        # self._train = h2o.H2OFrame(self._train.as_data_frame().sample(frac=1, random_state=246))
        # self._train, self._valid = self._train.split_frame(ratios=[0.8], seed=913)
        target = target.strip()
        
        # self._train[target] = self._train[target].asfactor()
        col_ls = self._train.columns
        if target not in col_ls:
            raiseCustomizedException('1101@{}'.format(target))
        col_ls.remove(target)
        self._features = col_ls
        self._target = target

        if self._train[target].types[target] == 'enum':
            self._class_count = self._train[target].unique().shape[0]
            self.set_model_type('classification')
        else:
            self._flag_regression = True
            self._class_count = self._train[target].unique().shape[0]
            self.set_model_type('regression')

        if FLAG_DEBUG_MODE:
            print('training_data.shape', self._train.shape)
            print('num of features:', len(self._features))
            print('feature:', self._features)
            print('target:', self._target)
    
    def setup_after_train(self):
        self.set_leaderboard_df()
        self.set_model_list()
        self.set_training_info()
        metrics, value = self.get_metrics(self._aml.leader.model_performance())
        self.set_accuracy(value)
        # self.set_accuracy(self.get_accuracy(self._aml.leader.model_performance()))

    def save_after_train(self):
        self.save_leaderboard()
        self.save_model_list()
        self.save_model_info()
        self.save_all_model()
        self.save_explain()

    def get_aml(self):
        return self._aml
    
    def get_leader(self):
        return self._aml.leader

    def get_leaderboard(self):
        # Leaderboard is ranked by xval metrics
        return self._aml.leaderboard.head(rows=self._aml.leaderboard.nrows)
    
    def set_leaderboard_df(self):
        self._leaderboard_df = self._aml.leaderboard.head(rows=self._aml.leaderboard.nrows).as_data_frame()
    
    def set_model_list(self):
        self._model_list = self._leaderboard_df['model_id'].tolist()
        self._my_logger.write_info('num of model: {}'.format(len(self._model_list)))

    def save_leaderboard(self):
        filepath = self._models_folderpath + 'leaderboard.csv'
        if FLAG_DEBUG_MODE:
            print('save leaderboard. shape:', self._leaderboard_df.shape)
        self._leaderboard_df.to_csv(filepath, index=False)
        self._my_logger.write_info('save leaderboard: '+filepath)
       
    def save_model_info(self):
        model_info_filepath = self._models_folderpath + TRAIN_INFO_FILENAME
        with open(model_info_filepath, 'w') as f:
            json.dump(self._train_info, f, indent=4)

    def save_model_list(self):
        # self._model_list = self._aml.leaderboard.head(rows=self._aml.leaderboard.nrows).as_data_frame()['model_id'].tolist()
        with open(self._model_list_filepath, 'w') as f:
            f.write('\n'.join(self._model_list))
        self._my_logger.write_info('save model list: ' + self._model_list_filepath)
        if FLAG_DEBUG_MODE:
            print('type(model_list): {}'.format(type(self._model_list)))

    # def read_model_list(self):
    #     with open(self._model_list_filepath, 'r') as f:
    #         ls = f.read().split('\n')
    #     self._model_list = ls
    #     return ls

    def save_all_model(self):
        # SAVE EVERY MODEL TRAINED
        num_save_model = len(self._model_list)
        # num_save_model = min(len(self._model_list), 5)
        for index in range(0, num_save_model):
            # get model, ranking and folderpath
            model_name = self._model_list[index]
            model = h2o.get_model(model_name)
            ranking = 1 + index
            folderpath = '{}/{}/'.format(self._models_folderpath, ranking_zfill(ranking))

            # save model
            ## note that h2o.save_model() will make the dir if not exist
            path_model = h2o.save_model(model=model, path=folderpath, force=True)
            
            # save feature importance chart and csv
            self.save_feature_importance(model, folderpath)
            
            # get metrics then save in training info
            perf = model.model_performance()
            metrics, accuracy = self.get_metrics(perf)
            self.save_training_info({'ranking': ranking, 'model_id': model_name, 'metrics': metrics, 'model_path':path_model}, folderpath)

            # save confusion_matrix
            self.save_confusion_matrix(perf, folderpath)
            self.save_model_performance(perf, folderpath)
            
        self._my_logger.write_info('save top {} model to: {}'.format(num_save_model, self._models_folderpath))
            
    def save_explain(self):
        # save_explain_plots(self._aml, self._train, "aml", self._models_folderpath) # all model
        save_explain_plots(self._model, self._train, "model", self._models_folderpath) # leader model
        self._my_logger.write_info('save model explainability to: {}'.format(os.path.join(self._models_folderpath, "chart")))
        

    def save_feature_importance(self, model, folderpath):
        if isinstance(model, h2o.estimators.stackedensemble.H2OStackedEnsembleEstimator):
            if FLAG_DEBUG_MODE:
                print('.varimp is not available for Stacked Ensembles')
            with open(folderpath+'feature_importance.txt', 'w') as f:
                f.write('varimp is not available for Stacked Ensembles\n')
            self._my_logger.write_info('varimp is not available for Stacked Ensembles: '+model.model_id)
        else:
            # dataframe -> csv
            varimp_df = model.varimp(use_pandas=True)
            if isinstance(varimp_df, pd.DataFrame):
                top_n_feature = min(10, varimp_df.shape[0])
                varimp_df = varimp_df.iloc[:top_n_feature,:]
                varimp_df.to_csv(folderpath+'feature_importance.csv', index=False)
                # plt -> png
                x = varimp_df['scaled_importance']
                x.index=varimp_df['variable']
                fig = plt.figure(figsize=(8, 6), dpi=100)
                x.sort_values().plot(kind='barh', title='Feature Importance')
                fig.savefig(folderpath+'feature_importance.jpg', dpi=100, bbox_inches='tight')
                # self._my_logger.write_info('save feature importance at: '+folderpath)
        return
        
    def save_model_performance(self, perf, folderpath):
            # dataframe -> csv
        if not self._flag_regression:
            pass
        else:
            metric_df = pd.DataFrame([perf.r2(), perf.mse(), perf.rmse(), perf.mae(), perf.rmsle()],
             columns=['Performance'], index=['R^2', 'MSE', 'RMSE', 'MAE', 'RMSLE']).round(3)
            if isinstance(metric_df, pd.DataFrame):
                metric_df.to_csv(folderpath+'confusion_matrix.csv', index=False)
                save_dataframe_as_image(metric_df, folderpath+"confusion_matrix.jpg")
                # self._my_logger.write_info('save feature importance at: '+folderpath)
        return
        
    def save_confusion_matrix(self, perf, folderpath):
        if self._flag_regression:
            pass
        else:
            # classification problem
            if self._class_count > 2:
                confusion_matrix = perf.confusion_matrix().as_data_frame()
                new_index = confusion_matrix.columns[:self._class_count].to_list()
                new_index.append('Total')
                confusion_matrix.index = new_index
            else:
                # confusion_matrix = self._aml.leader.confusion_matrix().table.as_data_frame()
                confusion_matrix = perf.confusion_matrix().table.as_data_frame()
                confusion_matrix.set_index('', inplace=True)

            # confusion_matrix.set_index('', inplace=True)
            confusion_matrix.drop(['Rate'], axis=1, inplace=True)
            # label_list = confusion_matrix.index[:-1]
            confusion_matrix = confusion_matrix.astype({label: "int" for label in confusion_matrix.index[:-1]})
            is_str = confusion_matrix.dtypes['Error'] == 'object'
            confusion_matrix['Error'] = confusion_matrix['Error'].apply(lambda x: '{:.1f} %'.format(100*eval(x) if is_str else 100*x))

            # confusion_matrix.to_csv(folderpath+'confusion_matrix.csv', index=False)
            confusion_matrix.to_csv(folderpath+'confusion_matrix.csv')
            # dfi.export(confusion_matrix, folderpath+"confusion_matrix.jpg", table_conversion='matplotlib')
            save_dataframe_as_image(confusion_matrix, folderpath+"confusion_matrix.jpg")
        return

    def get_metrics(self, perf):
        # returns metrics and value to be compared
        value = None
        if self._flag_regression:
            # regression problem
            mse = perf.mse()
            r2 = perf.r2()
            metrics = {'mse': mse, 'r2': r2}
            value = r2
        else:
            # classification problem
            if self._class_count > 2:
                confusion_matrix = perf.confusion_matrix().as_data_frame()
                error_rate = confusion_matrix['Error'].iloc[-1]
                accuracy = 1 - error_rate
                # confusion_matrix = perf.confusion_matrix().as_data_frame()
                # accuracy = 1-eval(confusion_matrix.iloc[-1,-1])
            else:
                threshold_accuracy = perf.accuracy()[0]
                threshold = threshold_accuracy[0]
                accuracy = threshold_accuracy[1]
            metrics = {'accuracy': accuracy}
            value = accuracy
        return metrics, value

    def set_training_info(self):
        if self._flag_regression:
            model_type = 'regression'
        else:
            model_type = 'classification'
            # 'max_runtime_secs': self._MAX_RUNTIME_SECS, \
            #                 'max_models': self._MAX_MODELS, \
            #                 'seed': self._SEED, \
        self._train_info = {'train_parameters': self._param, \
                            'train_filepath': self._train_filepath, \
                            'target': self._target, \
                            'type': model_type, \
                            'num_of_class': self._class_count, \
                            'start_time': self._start_time, \
                            'end_time': get_current_time(), \
                            'total_model': len(self._model_list), \
                            'leader_model_id': self._aml.leader.model_id}
        
    def save_training_info(self, model_info, folderpath):
        # model_id, ranking, metrics, model_path
        self._train_info.update(model_info)
        with open(folderpath+TRAIN_INFO_FILENAME, 'w') as f:
            json.dump(self._train_info, f, indent=4)
    
    def save_prediction_info(self, pred_info, folderpath):
        # test_filepath, test_data_shape, start_time, end_time, metrics
        self._info_dict.update(pred_info)
        with open(folderpath+'pred_info.json', 'w') as f:
            json.dump(self._info_dict, f, indent=4)

    def setup_model_info(self, ranking):
        with open('{}/{}/{}'.format(self._models_folderpath, ranking_zfill(ranking), TRAIN_INFO_FILENAME), 'r') as f:
            conf = json.load(f)
        self._model_path = conf['model_path']
        if conf['type'][0] == 'r':
            self._flag_regression = True
        else:
            self._flag_regression = False
        self._class_count = conf['num_of_class']
        # self._info_dict = {'model_path': conf['model_path'], 'type': conf['type'], 'target': conf['target'], 'train_filepath': conf['train_filepath']}
        self._info_dict = {key: conf[key] for key in ['model_path', 'type', 'target', 'train_filepath']}
    
    @staticmethod
    def get_model_path(models_folderpath, ranking):
        with open('{}/{}/{}'.format(models_folderpath, ranking_zfill(ranking), TRAIN_INFO_FILENAME), 'r') as f:
            conf = json.load(f)
        return conf['model_path']

    def setup_before_predict(self):
        model_info_filepath = self._models_folderpath + TRAIN_INFO_FILENAME
        with open(model_info_filepath, 'r') as f:
            conf = json.load(f)
        self._total_model = conf['total_model']
        self._target = conf['target']
        if conf['type'][0] == 'r':
            self._flag_regression = True
        else:
            self._flag_regression = False
        self._class_count = conf['num_of_class']
        # self._info_dict = {'model_path': conf['model_path'], 'type': conf['type'], 'target': conf['target'], 'train_filepath': conf['train_filepath']}
        self._info_dict = {key: conf[key] for key in ['type', 'target', 'train_filepath']}
        return
