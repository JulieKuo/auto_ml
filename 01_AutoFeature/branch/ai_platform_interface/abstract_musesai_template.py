import abc
import os
import sys
import json
import logging
import base64
import zipfile
import traceback
import redis
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from .AIPlatformMessageInterface import AIPlatformMessageInterface

VERSION = '2.0.5-afe'


class AbstractMusesAiFrame(metaclass=abc.ABCMeta):
    def __init__(self):
        self.log_msg = None
        self.ami = None
        self.log_folder = None
        self.log_name = None
        self.config_path = None
        self.program_user = None
        self.redis_config = None
        self.database_config = None
        self.user_id = None
        self.group_id = None
        self.project_id = None
        self.job_id = None
        self.mold_job_id = None
        self.root = None
        self.redis_head = None

    def preprocess(self, sys_argv):
        log_msg = self.activate_log()
        params = self.decode_argv(sys_argv)
        config = self.read_config()
        ami = self.activate_ami()
        return log_msg, ami, config, params

    def activate_log(self):
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        FORMAT = '%(asctime)s %(levelname)s %(message)s'
        handler = TimedRotatingFileHandler(
            f"{self.log_folder}/{self.log_name}", when="D", interval=1, backupCount=10,
            encoding="UTF-8", delay=False, utc=True)

        formatter = logging.Formatter(FORMAT)
        handler.setFormatter(formatter)

        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        return logger

    def decode_argv(self, argv):
        params_base64 = str(argv)
        base64_bytes = params_base64.encode("UTF-8")
        string_bytes = base64.b64decode(base64_bytes)
        string = string_bytes.decode("UTF-8")
        params = string.replace('\\\\', '#').replace('\\', '').replace('#', '\\')
        params = eval(params)
        self.log_msg.info('Request received, got arguments: {}'.format(params))
        try:
            self.user_id = params['userId']
            self.group_id = params['groupId']
            self.project_id = params['projectId']
            self.job_id = params['job_id']
        except KeyError:
            self.mold_job_id, self.group_id, self.project_id = params['trained_file'].split('_')[: 3]
            self.job_id = params['job_id']
        return params

    @staticmethod
    def encode_argv(argv):
        b = argv.encode("UTF-8")
        base64_argv = base64.b64encode(b)
        base64_argv = base64_argv.decode("UTF-8")
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

    @staticmethod
    def activate_custom_ami(job_id,user_id, group_id, project_id, database_config, redis_config, program_user):
        ami = AIPlatformMessageInterface(job_id, user_id, group_id, project_id, database_config, redis_config, program_user)
        return ami

    def report_progress(self, func, status=None, percent=None, **kwargs):
        """
        status: 'Uploading', 'Completed' or 'Error', status=None if no need to be change.
        percent: Number of percent between 0 and 1, percent=None if no need to be change.
        """
        assert self.ami, 'activate_ami must be called first'
        assert self.redis_head, 'redis_head is required'
        if percent:

            percent = percent / 2 if func == 'synthesis' \
                else percent / 3 + 0.5 if func == 'statistic' \
                else percent / 5 + 0.8 if func == 'transform' \
                else percent
            # sys.stdout.write(f"\r{func}:|" + "#" * int(percent * 40) + " " * int((1 - percent) * 40) + "|" + f"{percent * 100:.1f}%")
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

    @staticmethod
    def file_zipper(filepath, zipname):
        zf = zipfile.ZipFile(f"{filepath}/{zipname}.zip", mode='w')
        for root, _, files in os.walk(filepath):
            for file in files:
                if file != zipname + '.zip':
                    file_to_zip = file
                    zf.write(os.path.join(root, file_to_zip), arcname=file_to_zip)
        zf.close()

    @abc.abstractmethod
    def execution(self):
        return NotImplemented
