import sys, os, json, logging, time, traceback
from configparser import ConfigParser
from ai_platform_interface.AIPlatformMessageInterface import AIPlatformMessageInterface
from feature_fission.feature_engineering import FeatureEngineering
from feature_fission.feature_transform import FeatureTransformers
from train_model import *
from utils import loading_data
from utils.util_tools import base64_decoder


FORMAT = "%(asctime)s %(levelname)s %(message)s"
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.FileHandler("/home/stadmin/AIPlatform/ExecutiveFile/formula/FormulaAdvisor/logs/formula_train.log", 'a', 'utf-8')
# handler = logging.FileHandler(r'C:\Users\tzuli\Documents\python\1_AI\formula\FormulaAdvisor\logs\formula_train.log', 'a', 'utf-8')
handler.setFormatter(logging.Formatter(FORMAT))
root_logger.addHandler(handler)

ami = None

try:
    input_string = sys.argv[1]
    
    input_arguments = base64_decoder(input_string)
    print(f"input_arguments = {input_arguments}")
    file_name = input_arguments['fileName']
    label = input_arguments['label']
    group_id = input_arguments['groupId']
    user_id = input_arguments['userId']
    project_id = input_arguments['projectId']
    job_id = input_arguments['job_id']
    model_name = input_arguments['modelName']
    root_logger.info(f'input = {input_arguments}')


    config = ConfigParser()
    config.read("/home/stadmin/AIPlatform/ExecutiveFile/formula/FormulaAdvisor/config.ini", encoding='utf8')
    # config.read(r"C:\Users\tzuli\Documents\python\1_AI\formula\FormulaAdvisor\config.ini", encoding='utf8')
    program_user = config['Interface'].getint('PROGRAM_USER')
    redis_config = eval(config['Interface'].get('REDIS_CONFIG'))
    database_config = eval(config['Interface'].get('DATABASE_CONFIG'))
    root = config['Interface'].get('ROOT')
    model_counts = config.getint('Model Search', 'model_counts')
    valid_size = config.getfloat('Model Search', 'valid_size')
    epoch = config.getint('Model Search', 'epoch')


    ami = AIPlatformMessageInterface(job_id, user_id, group_id, project_id, database_config, redis_config, user=program_user)


    model_path = os.path.join(root, group_id, project_id, 'Tabular', 'model', model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    metadata_path = os.path.join(root, group_id, project_id, 'Tabular', 'model', model_name, 'metadata')
    data_filepath = os.path.join(root, group_id, project_id, 'Tabular', 'AfterLabelMerge', file_name)
    parser_filepath = os.path.join(root, group_id, project_id, 'Tabular', 'AfterLabelMerge', 'ParserResult')
    parser_filename = '.'.join([os.path.splitext(file_name)[0], 'json'])
    root_logger.info(f'data_filepath = {data_filepath}')
    df = loading_data(data_filepath)
    print(f"Data shape: {df.shape}")
    root_logger.info('file loaded')
    feat_engine = FeatureEngineering(metadata_path)
    feat_trans = FeatureTransformers(metadata_path)


    ami.start_job()
    root_logger.info('Training...')
    root_logger.info('FeatureEngineering in progressing...')
    df_fe = feat_engine.fit_transform(df, label)
    df_ft = feat_trans.fit_transform(df_fe)
    ms_begin_time = time.time()
    ami.set_value(f"formula_train_{group_id}_{project_id}_{model_name}_status", "Training")
    ami.set_value(f"formula_train_{group_id}_{project_id}_{model_name}_percent", 0.3)
    
    root_logger.info(f'Searching best model by Autokeras...')
    keras_model_path = os.path.join(model_path, "keras")
    if not os.path.exists(keras_model_path):
        os.makedirs(keras_model_path)
    ak = ModelAutokeras(model_name, keras_model_path)
    ak_model, ak_train_mse, ak_test_mse, ak_train_r2, ak_test_r2 = ak.search(df_ft, 'label', model_counts, epoch, valid_size)
    root_logger.info(f"Autokeras Done. trina_mse={ak_train_mse}, test_mse={ak_test_mse}, train_r2={ak_train_r2}, test_r2={ak_test_r2}")

    root_logger.info(f'Train MLP model...')
    mlp_model_path = os.path.join(model_path, "mlp")
    if not os.path.exists(mlp_model_path):
        os.makedirs(mlp_model_path)
    mlp = ModelMLP(mlp_model_path)
    mlp_model, mlp_train_mse, mlp_test_mse, mlp_train_r2, mlp_test_r2 = mlp.search(df_ft, 'label', valid_size)
    root_logger.info(f"MLP Done. trina_mse={mlp_train_mse}, test_mse={mlp_test_mse}, train_r2={mlp_train_r2}, test_r2={mlp_test_r2}")

    ami.set_value(f"formula_train_{group_id}_{project_id}_{model_name}_status", "Complete")
    ami.set_value(f"formula_train_{group_id}_{project_id}_{model_name}_percent", 1)
    ami.set_value(f"formula_train_{group_id}_{project_id}_{model_name}_loss", min(round(ak_test_mse, 2), round(mlp_test_mse, 2)))
    ami.set_value(f"formula_train_{group_id}_{project_id}_{model_name}_accuracy", max(round(ak_test_r2, 2), round(mlp_test_r2, 2)))


    with open(os.path.join(parser_filepath, parser_filename), 'r') as pr:
        parser_result = json.load(pr)

    parser_result.update({'label': label})
    with open(os.path.join(model_path, f"{model_name}.json"), 'w') as outfile:
        json.dump(parser_result, outfile, indent=4)


    with open(metadata_path, 'r') as mt:
        metadata = json.load(mt)

    metadata['model_info'] = [dict()]
    metadata['model_info'][0].update({'train_data': data_filepath})
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)


    ms_end_time = time.time()
    ms_dur_time = ms_end_time - ms_begin_time
    ami.complete_job()

except Exception as err:
    exstr = traceback.format_exc()
    print(exstr)
    root_logger.error(err)
    root_logger.error(exstr)

    if isinstance(ami, AIPlatformMessageInterface):
        ami.handle_error(err)
    print('error')
