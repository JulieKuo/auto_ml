import os
import sys
import shutil
import json
import pandas as pd
# import sweetviz as sv
from ai_platform_interface.abstract_musesai_template import AbstractMusesAiFrame
from task.utils.data_type_parser import DataTypeParser


NAME = 'parser_tool.exe'
MAJOR = 1
MINOR = 2
PATCH = '20220223'
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"

ROOT_CLOUD = "/aiplatform/ExecutiveFile/ParserTool"
ROOT_GROUND = "."
ROOT = ROOT_GROUND


class ParserTool(AbstractMusesAiFrame):
    def __init__(self, sys_argv):
        super().__init__()
        self.log_folder = f"{ROOT}/log"
        self.config_path = f"{ROOT}/config.json"
        self.execution(sys_argv)

    @AbstractMusesAiFrame.exception(VERSION)
    def execution(self, sys_argv):
        self.log_msg = self.activate_log()
        params = self.decode_argv(sys_argv)
        config = self.read_config()
        self.ami = self.activate_ami()
        file_path = os.path.join(self.root, self.group_id, self.project_id, 'FileUploadArea', 'Tabular')
        file_name = params['fileName']
        self.redis_head = [self.group_id, self.project_id, file_name]

        self.ami.start_job()
        self.report_progress(status='Analyzing', percent=0.32)
        self.log_msg.info(f'Parser request received, filename={file_name}')
        print("Loading data and check head format...", end=" ")
        dtypes_file_name = f"{os.path.splitext(file_name)[0]}.json"
        dtypes_path = os.path.join(self.root, self.group_id, self.project_id, 'FileHeaderAndType')
        # profiling_html_path = os.path.join(self.root, 'UserHTMLGroup', self.group_id, self.project_id, 'AF_parser_result')
        dtype_file = os.path.join(dtypes_path, dtypes_file_name)
        if not os.path.exists(dtypes_path):
            os.makedirs(dtypes_path)
        # if not os.path.exists(profiling_html_path):
        #     os.makedirs(profiling_html_path)
        try:
            train = pd.read_csv(os.path.join(file_path, 'temp', file_name), low_memory=False, header=0)
            train_noheader = pd.read_csv(os.path.join(file_path, 'temp', file_name), low_memory=False, header=None)
        except pd.errors.ParserError as e:
            raise Exception('1001')

        if self._check_header(train, train_noheader):
            print("OK")
            self.log_msg.info(f'Assert dataset header: {list(train.columns)}')
            del train_noheader
            self.report_progress(percent=0.33)

            # Remove irregular samples
            to_keep_index, impurity_list = self._centrifugal(train)
            if len(impurity_list) > 0:
                self.log_msg.info(f'  Impurity list has been removed: {impurity_list}')
            train = train.loc[to_keep_index].reset_index(drop=True)

            # Save dataset to 'FileUploadArea/Tabular/'
            train.to_csv(os.path.join(file_path, file_name), index=False)
            os.remove(os.path.join(file_path, 'temp', file_name))
            self.report_progress(percent=0.35)

            # Get and save dtype_info of parser result
            dtp = DataTypeParser(self.report_progress)
            dtype_info = dtp.get_feature_types(train)
            self.report_progress(percent=0.99)
            print(f"\nSaving {file_name} and {dtypes_file_name}")
            dtype_info['length'] = train.shape[1]
            dtype_info['impurity_value'] = impurity_list
            self.log_msg.info(f"save {dtypes_file_name} to {dtype_file}")
            with open(dtype_file, 'w') as dp:
                json.dump(dtype_info, dp, indent=4)
            self.report_progress('Complete', 1)

            # Write size usage to DB
            csv_size = self.obtain_size(os.path.join(file_path, file_name))
            json_size = self.obtain_size(os.path.join(dtypes_path, dtypes_file_name))
            self.ami.record_volume(exe_name=NAME, file_name=file_name, file_path='-1', file_size=csv_size)
            self.ami.record_volume(exe_name=NAME, file_name=dtypes_file_name, file_path='-1', file_size=json_size)
            self.log_msg.info(f"{file_name} usage={csv_size}(bytes)")
            self.log_msg.info(f"{dtypes_file_name} usage={json_size}(bytes)")

            self.log_msg.info('Parsing successfully completed')
            print("Parsing successfully completed")
            self.ami.complete_job()

            # Call profiling
            # self._eda(train, file_name)
        else:
            print("NG")
            self.log_msg.error('Detected the data does not have header')
            os.remove(os.path.join(file_path, 'temp', file_name))
            self.log_msg.warning('{} has been removed from temp folder'.format(file_name))
            self.ami.delete_value('_'.join([self.group_id, self.project_id, 'fileUpload']), 0, file_name)
            self.log_msg.warning('Expelled {} from list of redis, key: fileUpload'.format(file_name))
            self.report_progress(status='Error')
            raise Exception('1002')
        return

    @staticmethod
    def _check_header(df, df_noheader):
        df_nunique = df.nunique().reset_index(drop=True)
        df_nohead_nunique = df_noheader.nunique().reset_index(drop=True)
        if df_nohead_nunique.subtract(df_nunique).sum() >= len(df.columns) * 0.5:
            return True
        return False

    @staticmethod
    def _centrifugal(df, max_try=5, unique_threshold=256):
        dtype_dict = {}
        impurity_list = []
        to_keep_index = df.index
        obj_cols = df.select_dtypes(include=['object'])
        for name, column in obj_cols.items():
            if column.nunique() > unique_threshold:
                output = ""
                impurity_temp = []
                column = column.loc[to_keep_index]
                curr_try = max_try
                while curr_try > 0:
                    try:
                        column.astype(float)
                        if curr_try == max_try:
                            break
                        to_keep_index = keep_index_temp
                        impurity_list.extend(impurity_temp)
                        dtype_dict.update({name: float})
                        print(output)
                        break
                    except ValueError as e:
                        impurity = str(e).split('could not convert string to float: ')[-1].strip("'")
                        keep_index_temp = column[~(column == impurity)].index
                        impurity_temp.append(impurity)
                        output += f"remove {name} contains {impurity} columns: {column.index.difference(keep_index_temp).values}\n"
                        column = column.loc[keep_index_temp]
                    curr_try -= 1
        impurity_list = list(set(impurity_list))
        return to_keep_index, impurity_list

    # def _eda(self, df, file_name):
    #     try:
    #         profiling_html_path = os.path.join(self.root, 'UserHTMLGroup', self.group_id, self.project_id, 'AF_parser_result')
    #         if not os.path.exists(profiling_html_path):
    #             os.makedirs(profiling_html_path)
    #         self.ami.set_value('_'.join(self.redis_head + ['profiling']), 'Profiling')
    #         report_train = sv.analyze([df, file_name], pairwise_analysis='off')
    #         report_train.show_html(filepath=os.path.join(profiling_html_path, '_'.join([file_name, 'profiling.html'])), open_browser=False)
    #         self.ami.set_value('_'.join(self.redis_head + ['profiling']), 'Completed')
    #     except Exception as e:
    #         print("Profiling Error", e)
    #         self.log_msg.warning('Profiling Error')
    #         self.log_msg.error(e)
    #         self.ami.set_value('_'.join(self.redis_head + ['profiling']), 'Error')

if __name__ == '__main__':
    # argv = "e1wiZmlsZU5hbWVcIjpcIk5CX1NBTjVfMjAxOTExXzIwMjEwM18yMDIxMDYzMC5jc3ZcIixcImpvYl9pZFwiOlwiMjAyMTA3MjcwMTE1MDczNTBcIixcImZpbGVQYXRoXCI6XCJDOlxcU2VydnRlY2hcXFNlcnZvbHV0aW9uXFxQbGF0Zm9ybVxcQUlQbGF0Zm9ybS9Hcm91cFByb2plY3QvVzA4UVBUNFBORzJCNlAvcDAwMDA1L0ZpbGVVcGxvYWRBcmVhL1RhYnVsYXJcIixcImdyb3VwSWRcIjpcIlcwOFFQVDRQTkcyQjZQXCIsXCJ1c2VySWRcIjpcIlcwOFFQVDRQTkcyQjZQX2ZjZmNtdXNlYWlAc2VydnRlY2guY29tLnR3XCIsXCJwcm9qZWN0SWRcIjpcInAwMDAwNVwifQ=="
    # ParserTool(argv)
    ParserTool(sys.argv[1])
