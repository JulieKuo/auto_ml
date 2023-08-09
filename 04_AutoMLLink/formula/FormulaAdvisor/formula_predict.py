import sys, os, json, logging, traceback
from configparser import ConfigParser
from ai_platform_interface.AIPlatformMessageInterface import AIPlatformMessageInterface
from feature_fission.feature_engineering import FeatureEngineering
from feature_fission.feature_transform import FeatureTransformers
from utils import loading_data, contract_data_range
from utils.util_tools import base64_decoder
import pandas as pd
from advisor import *


FORMAT = "%(asctime)s %(levelname)s %(message)s"
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
handler = logging.FileHandler("/home/stadmin/AIPlatform/ExecutiveFile/formula/FormulaAdvisor/logs/formula_predict.log", 'a', 'utf-8')
# handler = logging.FileHandler(r'C:\Users\tzuli\Documents\python\1_AI\formula\FormulaAdvisor\logs\formula_predict.log', 'a', 'utf-8')
handler.setFormatter(logging.Formatter(FORMAT))
root_logger.addHandler(handler)

ami = None


try:
    input_string = sys.argv[1]
    
    input_arguments = base64_decoder(input_string)
    group_id = input_arguments['groupId']
    user_id = input_arguments['userId']
    project_id = input_arguments['projectId']
    job_id = input_arguments['job_id']
    model_name = input_arguments['modelName']
    target = input_arguments['target']
    file_id = input_arguments['fileId']
    root_logger.info(f'input = {input_arguments}')


    config = ConfigParser()
    config.read("/home/stadmin/AIPlatform/ExecutiveFile/formula/FormulaAdvisor/config.ini", encoding='utf8')
    # config.read(r"C:\Users\tzuli\Documents\python\1_AI\formula\FormulaAdvisor\config.ini", encoding='utf8')
    program_user = config['Interface'].getint('PROGRAM_USER')
    redis_config = eval(config['Interface'].get('REDIS_CONFIG'))
    database_config = eval(config['Interface'].get('DATABASE_CONFIG'))
    root = config['Interface'].get('ROOT')
    FI_EPOCHS = config.getint('Formula Inference', 'epochs')
    FI_FIT_RATE = config.getfloat('Formula Inference', 'fitting_rate')
    time_limit = config.getfloat('Formula Inference', "per_advisor_time_limit")


    ami = AIPlatformMessageInterface(job_id, user_id, group_id, project_id, database_config, redis_config, user=program_user)


    model_path = os.path.join(root, group_id, project_id, 'Tabular', 'model', model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    metadata_path = os.path.join(root, group_id, project_id, 'Tabular', 'model', model_name, 'metadata')
    parser_filepath = os.path.join(root, group_id, project_id, 'Tabular', 'AfterLabelMerge', 'ParserResult')
    root_logger.info('file loaded')
    feat_engine = FeatureEngineering(metadata_path)
    feat_trans = FeatureTransformers(metadata_path)
    with open(metadata_path, 'r') as mt:
        metadata = json.load(mt)
    root_logger.info('Predicting...')
    data_filepath = metadata['model_info'][0]['train_data']

    with open(os.path.join(model_path, f"{model_name}.json")) as pf:
        formula_constrain = json.load(pf)
        formula_constrain.pop('label')
    constrain_df = pd.DataFrame.from_dict(formula_constrain).T
    dtype = constrain_df[constrain_df["advise_dtype"] == "numerical"].set_index("column_name")["dtype_for_process"].to_dict()
    constrain_df = constrain_df[['column_name', 'describe']].set_index('column_name', drop=True)
    constrain_df[['count', 'mean', 'stdv', 'min', 'max', 'num_unique']] = pd.DataFrame(constrain_df['describe'].to_list(), index=constrain_df.index)
    constrain_df.drop(columns=['describe', 'count', 'mean', 'stdv', 'num_unique'], inplace=True)
    constrain_df = constrain_df.T.astype(dtype)


    trained_df = loading_data(data_filepath)
    root_logger.info('File loaded')
    root_logger.info('FeatureEngineering in progressing...')
    trained_df = feat_engine.transform(trained_df)
    trained_df = feat_trans.transform(trained_df)
    constrain_df.columns = trained_df.columns
    constrain_df = feat_engine.transform(constrain_df)
    constrain_df = feat_trans.transform(constrain_df)
    constrain_df.pop('label')


    ami.set_value(f"formula_pred_{group_id}_{project_id}_{model_name}_status", "Training")
    ami.set_value(f"formula_pred_{group_id}_{project_id}_{model_name}_percent", 0.3)
    clip_trained_df = contract_data_range(trained_df, 'label', target, 0.1)
    clip_trained_df_wo_label = pd.DataFrame(clip_trained_df.drop(columns='label', axis=1))
    
    
    
    kde = FormulaAdvisorKde(model_path, clip_trained_df, constrain_df)
    adam = FormulaAdvisorAdam(model_path, constrain_df)

    df_results = pd.DataFrame()
    for idx, quant in enumerate([0.25, 0.375, 0.5, 0.625, 0.75]):
        for advisor in ["kde", "adam"]:
            initial_formula = pd.DataFrame(clip_trained_df_wo_label.quantile(q=quant)).T
            root_logger.info(f'Finding acceptable formula by quant ({quant}), advisor({advisor})...')

            if advisor == "kde":
                inference_formula, inference_target = kde.train(target, initial_formula, epochs=FI_EPOCHS, fitting_rate=0.03, time_limit=time_limit)
            elif advisor == "adam":
                inference_formula, inference_target = adam.train(target, initial_formula, time_limit = time_limit)        
            
            # root_logger.info(f'Inverse transform formula')
            formula_inverse_trans = feat_trans.inverse_transform(inference_formula)
            df_result = feat_engine.inverse_transform(formula_inverse_trans)
            df_result.insert(0, "Target", inference_target)
            df_result.insert(0, 'deviation', inference_target - target)
            df_result.insert(0, "advisor", advisor)
            df_results = pd.concat([df_results, df_result], ignore_index=True)


    df_results.sort_values(by='deviation', key=lambda x: abs(x), inplace=True)
    df_results.reset_index(drop=True, inplace=True)
    root_logger.info(f'Results written to: {model_path}/predict_result.csv')
    df_results.to_csv(f"{model_path}/predict_result.csv")


    ami.set_value(f"formula_pred_{group_id}_{project_id}_{file_id}_{model_name}_percent", 1)
    ami.set_value(f"formula_pred_{group_id}_{project_id}_{file_id}_{model_name}_status", "Complete")
    ami.set_value(f"formula_pred_{group_id}_{project_id}_{file_id}_{model_name}_deviation", str(df_results.loc[0, "deviation"]))
    ami.complete_job()

except Exception as err:
    exstr = traceback.format_exc()
    print(exstr)
    root_logger.error(err)
    root_logger.error(exstr)

    if isinstance(ami, AIPlatformMessageInterface):
        ami.handle_error(err)
    print('error')
