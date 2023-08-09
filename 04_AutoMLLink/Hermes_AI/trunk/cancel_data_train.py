import sys, os 
from ExecutionLog import ExecutionLog
from AIPlatformMessageInterface import AIPlatformMessageInterface
from customized_exception import raiseCustomizedException
from file_io import load_json
from pathlib import Path


DEBUG_MODE = True
job_name = 'cancel_data_train'
default_log_folder = '/home/stadmin/AIPlatform/ExecutiveFile/data/log/'
config_filepath = '/home/stadmin/AIPlatform/ExecutiveFile/data/data_train_config.json'

if __name__ == "__main__":

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
        platform_msg_handler = AIPlatformMessageInterface(arg_dict['job_id'], arg_dict["userId"], arg_dict['groupId'], arg_dict['projectId'], program_config['DATABASE_CONFIG'], program_config['REDIS_CONFIG'], program_config['PROGRAM_USER'])
        
    except Exception as e:
        log_obj.write_error('step1: get arguments\n  '+str(e))
    else:
        log_obj.write_info(arg_dict)
    
    try:
        platform_msg_handler.start_job()

        # redisKey_pid = 'train_{groupId}_{projectId}_{modelName}_pid'.format(groupId=arg_dict['groupId'], projectId=arg_dict['projectId'], modelName=arg_dict['modelName'])
        redisKey = 'train_{}_{}_{}'.format(arg_dict['groupId'], arg_dict['projectId'], arg_dict['modelName'])

        cancel_job_pid = platform_msg_handler.get_value(redisKey+'_pid')
        command = 'kill -9 {}'.format(cancel_job_pid)
        # os.system("cmd /k "+command)
        os.system(command)
        platform_msg_handler.kill_process(arg_dict['cancel_job_id'], arg_dict['groupId'], arg_dict['projectId'], arg_dict['userId'])
        platform_msg_handler.set_value(redisKey+'_status', 'Cancel')

    except Exception as e:
        if DEBUG_MODE:
            print(e, type(e))
        log_obj.write_error(str(e))
        platform_msg_handler.handle_error(e, 9199)

    else:
        log_obj.complete_job()
        platform_msg_handler.complete_job()
