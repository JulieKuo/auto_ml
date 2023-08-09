from AIPlatformMessageInterface import AIPlatformMessageInterface
from customized_exception import CustomizedException, raiseCustomizedException
from check_license import check_license
from ExecutionLog import ExecutionLog
from file_io import load_json
from H2O_AutoMLInterface import H2O_AutoMLInterface
from h2o.exceptions import H2OError, H2OJobCancelled
from pathlib import Path
import sys


job_name = 'data_predict'
default_log_folder = '/home/stadmin/AIPlatform/ExecutiveFile/data/log/'
license_config_filepath = '/home/stadmin/AIPlatform/ExecutiveFile/data/check_license.json'

config_filepath = '/home/stadmin/AIPlatform/ExecutiveFile/data/data_predict_config.json'


if __name__ == "__main__":

    # setup vairables needed
    try:
        if not Path(config_filepath).exists():
            log_obj = ExecutionLog(default_log_folder)
            log_obj.start_job(job_name)
            raiseCustomizedException(f'Config file does not exists. \"{config_filepath}\"')

        program_config = load_json(config_filepath)
        log_filepath = program_config['LOG_FOLDER'] if 'LOG_FOLDER' in program_config.keys() else default_log_folder

        log_obj = ExecutionLog(log_filepath)
        log_obj.start_job(job_name)

        if len(sys.argv) < 2:
            raiseCustomizedException('arguments missing')

        arg_dict = AIPlatformMessageInterface.get_arguments(sys.argv[1])
    
        filename = arg_dict['fileName']
        filepath = '{}/{}/{}/Tabular/AfterLabelMerge/{}'.format(program_config['GROUP_PROJECT_FOLDER'], arg_dict['groupId'], arg_dict['projectId'], arg_dict['fileName'])
        models_folderpath = '{}/{}/{}/Tabular/model/{}/'.format(program_config['GROUP_PROJECT_FOLDER'], arg_dict['groupId'], arg_dict['projectId'], arg_dict['modelName'])
        output_folderpath = '{}/{}/{}/Tabular/prediction/{}_{}/'.format(program_config['GROUP_PROJECT_FOLDER'], arg_dict['groupId'], arg_dict['projectId'], filename, arg_dict['modelName'])
        
        # redisKey
        redisKey = 'pred_{}_{}_{}_{}'.format(arg_dict['groupId'], arg_dict['projectId'], filename, arg_dict['modelName'])
        redisKey_status = redisKey + '_status'

        # connect to AIPlatform
        platform_msg_handler = AIPlatformMessageInterface(arg_dict['job_id'], arg_dict['userId'], arg_dict['groupId'], arg_dict['projectId'], program_config['DATABASE_CONFIG'], program_config['REDIS_CONFIG'], program_config['PROGRAM_USER'])
        
    except Exception as e:
        log_obj.write_exception('Exception caught - set up step')

    else:
        log_obj.write_info(arg_dict)

        # start
        try:
            # check license
            try:
                if not check_license(license_config_filepath):
                    raiseCustomizedException('9101@License check failed')
            except FileNotFoundError as fe:
                raiseCustomizedException(f'9102@file missing: {str(fe)}')

            # start job
            platform_msg_handler.start_job()
            platform_msg_handler.set_value(redisKey_status, 'Running')
            
            # predict
            h2o_obj = H2O_AutoMLInterface(platform_msg_handler, redisKey, log_obj)
            h2o_obj.predict(filepath, models_folderpath, output_folderpath)
        
        except H2OJobCancelled as e:
            log_obj.write_error(f'H2OJobCancelled - {str(e)}')
            platform_msg_handler.set_value(redisKey_status, 'Error')
            platform_msg_handler.handle_error(CustomizedException("9199@H2OJobCancelled"))

        except CustomizedException as e:
            log_obj.write_error(str(e))
            platform_msg_handler.set_value(redisKey_status, 'Error')
            platform_msg_handler.handle_error(e)
            
        except Exception as e:
            log_obj.write_error(f'9199@{str(type(e))} {str(e)}')
            platform_msg_handler.set_value(redisKey_status, 'Error')
            print(e, type(e))
            if not platform_msg_handler.handle_error(e, 9199):
                raise
        else:
            platform_msg_handler.set_value(redisKey_status, 'Complete')
            platform_msg_handler.complete_job()
            log_obj.complete_job()
