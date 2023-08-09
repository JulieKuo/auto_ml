import abc
import os
import json
import logging
import base64
import traceback
import redis
from pathlib import Path
from datetime import datetime
from .AIPlatformMessageInterface import AIPlatformMessageInterface

VERSION = '2.0.3'


class AbstractMusesAiFrame(metaclass=abc.ABCMeta):
    def __init__(self):
        self.log_msg = None
        self.ami = None
        self.log_folder = None
        self.config_path = None
        self.program_user = None
        self.redis_config = None
        self.database_config = None
        self.user_id = None
        self.group_id = None
        self.project_id = None
        self.job_id = None
        self.root = None
        self.redis_head = None

    def preprocess(self, sys_argv):
        log_msg = self.activate_log()
        params = self.decode_argv(sys_argv)
        config = self.read_config()
        ami = self.activate_ami()
        return log_msg, ami, config, params

    def activate_log(self):
        FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        log_filename = datetime.today().strftime("%Y%m%d")
        if not Path(self.log_folder).exists():
            os.makedirs(self.log_folder)
        handler = logging.FileHandler(f"{self.log_folder}/{log_filename}.log", 'a', 'utf-8')
        handler.setFormatter(logging.Formatter(FORMAT))
        root_logger.addHandler(handler)
        return root_logger

    def decode_argv(self, argv):
        params_base64 = str(argv)
        base64_bytes = params_base64.encode("UTF-8")
        string_bytes = base64.b64decode(base64_bytes)
        string = string_bytes.decode("UTF-8")
        params = string.replace('\\\\', '#').replace('\\', '').replace('#', '\\')
        params = eval(params)
        self.user_id = params['userId']
        self.group_id = params['groupId']
        self.project_id = params['projectId']
        self.job_id = params['job_id']
        self.log_msg.info('Request received, got arguments: {}'.format(params))
        return params

    @staticmethod
    def encode_argv(argv):
        base64_argv = base64.b64encode(argv)
        return base64_argv

    def read_config(self):
        self.log_msg.info('Reading Config')
        with open(self.config_path, 'r') as cf:
            config = json.load(cf)
        self.program_user = config['PROGRAM_USER']
        self.redis_config = config['REDIS_CONFIG']
        self.database_config = config['DATABASE_CONFIG']
        self.root = config['ROOT']
        return config

    def activate_ami(self):
        ami = AIPlatformMessageInterface(self.job_id,
                                         self.user_id,
                                         self.group_id,
                                         self.project_id,
                                         self.database_config,
                                         self.redis_config,
                                         self.program_user)
        return ami

    def report_progress(self, status=None, percent=None):
        """
        status: 'Uploading', 'Completed' or 'Error', status=None if no need to be change.
        percent: Number of percent between 0 and 1, percent=None if no need to be change.
        """
        assert self.ami, 'activate_ami must be called first'
        assert self.redis_head, 'redis_head is required'
        if percent:
            assert 1 >= percent >= 0, 'progress percent must between 0 and 1'
            self.ami.set_value('_'.join(self.redis_head + ['percent']), percent)
        if status:
            self.ami.set_value('_'.join(self.redis_head + ['status']), status)

    def exception(version):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                class_name = type(self).__name__
                try:
                    print("Copyright (c) 2021 Servtech Co., Ltd. All Rights Reserved.\n")
                    print(f"Running {class_name}-{version}")
                    func(self, args, **kwargs)
                    print("Completed.")
                except Exception as err:
                    print("Error occurred", err)
                    exc_str = traceback.format_exc()
                    print(exc_str)
                    self.log_msg.error(exc_str)
                    if isinstance(self.ami, AIPlatformMessageInterface):
                        if self.redis_head:
                            self.ami.set_value('_'.join(self.redis_head + ['status']), 'Error')
                            self.log_msg.info('set redis status as error')
                        self.ami.handle_error(err)
                        self.log_msg.info(f'set database error_msg as {err}')
                return
            return wrapper
        return decorator

    def _test_connection(self):
        self.log_msg = self.activate_log()
        print("Checking config.json... ", end="")
        if Path(self.config_path).exists():
            print("OK")
            config = self.read_config()
        else:
            print("NG")
            return

        print("Checking Database connection... ", end="")
        try:
            ami = AIPlatformMessageInterface(19990913202109040,
                                             'OzonoMomoko',
                                             'nogizaka',
                                             'p00046',
                                             self.database_config,
                                             self.redis_config)
            ami.close_connection()
            print("OK")
        except Exception as e:
            print("NG")
            raise e

        print("Checking Redis connection... ", end="")
        try:
            r = redis.Redis(**self.redis_config)
            r.set("connection_test", "To the moon")
            print("OK")
        except ConnectionError as e:
            print("NG")
            raise e

    @staticmethod
    def obtain_size(path):
        # get size
        size = 0
        try:
            for file in os.scandir(path):
                size += os.path.getsize(file)
        except NotADirectoryError:
            size = os.path.getsize(path)
        return size

    @abc.abstractmethod
    def execution(self):
        return NotImplemented
