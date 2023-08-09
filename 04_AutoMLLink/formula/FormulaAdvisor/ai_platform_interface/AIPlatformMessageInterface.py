# -*- coding: utf-8 -*

import os
import base64
from datetime import datetime
import json
import mysql.connector
import redis
import requests.exceptions
from .MessageInterface import MessageInterface
from .customized_exception import raiseCustomizedException

VERSION = '2.0.2a'

STATUS_WAITING = 0    # 等待執行
STATUS_RUNNING = 1    # 執行中
STATUS_COMPLETE = 9    # 成功執行完
STATUS_CONNECTION_ERROR = 98    # 有執行但連線失敗
STATUS_OTHER_ERROR = 99    # 有執行但未正常結束
STATUS_STOP = 999    # 手動取消


class AIPlatformMessageInterface(MessageInterface):
    def __init__(self, job_id, user_id, group_id, project_id, database_config={}, redis_config={}, user=None):
        self._job_id = job_id
        self._user_id = user_id
        self._group_id = group_id
        self._project_id = project_id
        self._database_args = {'host': '127.0.0.1', 'port': 53306, 'user': 'root', 'password': 'servtechpwd', 'database': 'servcloud'}
        self._redis_args = {'host': '127.0.0.1', 'port': 6379, 'decode_responses': True}
        
        if user is not None:
            if user == 1:
                self._database_args.update({'user': 'hermesai', 'password': '1qaz@WSX3edc'})
                self._redis_args['password'] = '87893939'
            elif user == 2:
                self._database_args.update({'user': 'servtechadmin@musesaidb', 'password': 'MusesAi@0624'})
                self._redis_args['password'] = 'FaVyOvkiwXwiYB4hW0UGNVWOdXyllKk0xfUGwQh8NJ4='
            elif user == 78:
                pass
        
        # self._database_args.update({'host': conf['DATABASE_IP'], 'port': conf['DATABASE_PORT']})
        # self._redis_args.update({'host': conf['REDIS_IP'], 'port': conf['REDIS_PORT']})
        self._database_args.update(database_config)
        self._redis_args.update(redis_config)

        self._r = None
        self._connection = None
        self._cursor = None
        self.build_connection()

    @staticmethod
    def get_arguments(params_base64):
        # s = eval(s)
        # <Cliff> s = s.encode('utf-8')
        try:
            string_bytes = base64.b64decode(params_base64)
        except base64.binascii.Error:
            raiseCustomizedException('9299@Incorrect argument')
        s2 = string_bytes.decode('utf-8')
        arg_dict = json.loads(s2.replace('\\\"', '\"').replace('\\', '/'))
        return arg_dict
        
    def get_value(self, key):
        # print('AIPlatformMessageInterface.get_value')
        return self._r.get(key)

    def set_value(self, key, value):
        # print('AIPlatformMessageInterface.set_value')
        self._r.set(key, value)
    
    def get_list_value(self, key, start=0, end=-1):
        # print('AIPlatformMessageInterface.get_list_value')
        return self._r.lrange(key, start, end)
    
    def add_list_value(self, key, value):
        # print('AIPlatformMessageInterface.add_list_value')
        self._r.rpush(key, value)

    def delete_value(self, key, count, value):
        self._r.lrem(key, count, value)

    def build_connection(self):
        # print('AIPlatformMessageInterface.build_connection')
        try:
            r = redis.Redis(**self._redis_args)
            connection = mysql.connector.connect(**self._database_args)
            if connection.is_connected():
                # # 顯示資料庫版本
                db_Info = connection.get_server_info()
                # print("資料庫版本：", db_Info)

                # # 顯示目前使用的資料庫
                cursor = connection.cursor()
                cursor.execute("SELECT DATABASE();")
                record = cursor.fetchone()
                # print("目前使用的資料庫：", record)
            else:
                # 連接失敗
                pass
        except Exception as e:
            # 連接失敗
            print("資料庫連接失敗：", e)
            raise

        self._r = r
        self._connection = connection
        self._cursor = cursor
        return

    def close_connection(self):
        # print('AIPlatformMessageInterface.close_connection')
        if (self._connection.is_connected()):
            self._cursor.close()
            self._connection.close()
            print("資料庫連線已關閉")
        return
    
    def check_connection(self):
        # print('AIPlatformMessageInterface.check_connection')
        if not (self._connection.is_connected()):
            self.build_connection()
        else:
            pass
            #print('Is connected.')
    
    def start_job(self):
        # print('AIPlatformMessageInterface.start_job')
        self.check_connection()
        # # Update DB status (to "1") and exec_times ( x += 1)
        select = "SELECT exec_times FROM a_application_call_exe_log WHERE job_id = %s AND group_id = %s AND project_id = %s"
        ids = (self._job_id, self._group_id, self._project_id)
        self._cursor.execute(select, ids)
        
        exec_times = 1 + self._cursor.fetchone()[0]
        update = "UPDATE a_application_call_exe_log SET status = %s, exec_times = %s WHERE job_id = %s AND group_id = %s AND project_id = %s"
        val = (STATUS_RUNNING, exec_times, self._job_id, self._group_id, self._project_id)
        self._cursor.execute(update, val)
        self._connection.commit()
        return
        
    def activate_job(self):
        # print('AIPlatformMessageInterface.activate_job')
        self.check_connection()
        sql = "UPDATE a_application_call_exe_log SET status = %s WHERE job_id = %s AND group_id = %s AND project_id = %s"
        val = (STATUS_WAITING, self._job_id, self._group_id, self._project_id)
        self._cursor.execute(sql, val)
        self._connection.commit()
        return

    def complete_job(self):
        # print('AIPlatformMessageInterface.complete_job')
        self.check_connection()
        sql = "UPDATE a_application_call_exe_log SET status = %s, modify_time = %s WHERE job_id = %s AND group_id = %s AND project_id = %s"
        val = (STATUS_COMPLETE, self.get_current_time(), self._job_id, self._group_id, self._project_id)
        self._cursor.execute(sql, val)
        self._connection.commit()
        return

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

    def record_volume(self, exe_name, file_name, file_path, file_size):
        self.check_connection()
        sql = "INSERT INTO a_application_job_use_capacity (job_id, exe_name, file_name, file_path, file_size, user_id, group_id, project_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (self._job_id, exe_name, file_name, file_path, file_size, self._user_id, self._group_id, self._project_id)
        self._cursor.execute(sql, val)
        self._connection.commit()
        return

    def handle_error(self, e, error_code=None, ignore_most_exceptions=False):
        # # ignore mostly == True and it's not request exception, then continue job
        # # ignore mostly == False and IT IS request exception, then stop doing this job.
        # if not handle_error(e, True/False):
        #     raise

        # print('AIPlatformMessageInterface.handle_error\t',type(e), str(e))
        self.check_connection()
        
        if isinstance(e, requests.exceptions.RequestException):
            # connection error
            sql = "UPDATE a_application_call_exe_log SET status = %s, modify_time = %s WHERE job_id = %s AND group_id = %s AND project_id = %s"
            val = (STATUS_CONNECTION_ERROR, self.get_current_time(), self._job_id, self._group_id, self._project_id)
            self._cursor.execute(sql, val)
            self._connection.commit()
            flag_continue_job = False
        else:
            # other error
            if error_code:
                error_msg = '{}@{}'.format(error_code, str(e))
            else:
                error_msg = str(e)

            self.check_connection()
            sql = "UPDATE a_application_call_exe_log SET status = %s, error_msg = %s, modify_time = %s WHERE job_id = %s AND group_id = %s AND project_id = %s"
            val = (STATUS_OTHER_ERROR, error_msg, self.get_current_time(), self._job_id, self._group_id, self._project_id)
            self._cursor.execute(sql, val)
            self._connection.commit()
            flag_continue_job = True
            print(val)
        # print(type(e), ignore_most_exceptions, flag_continue_job)
        return (ignore_most_exceptions and flag_continue_job)
    
    @staticmethod
    def get_current_time():
        datefmt='%Y-%m-%d %H:%M:%S'
        return datetime.utcnow().strftime(datefmt)

    def kill_process(self, job_id, group_id, project_id, user_id):
        self.check_connection()
        sql = "UPDATE a_application_call_exe_log SET status = %s, modify_by = %s, modify_time = %s WHERE job_id = %s AND group_id = %s AND project_id = %s"
        val = (STATUS_STOP, user_id, self.get_current_time(), job_id, group_id, project_id)
        self._cursor.execute(sql, val)
        self._connection.commit()
        return
