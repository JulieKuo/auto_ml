import os
import sys
import shutil
import json
import pandas as pd
import sweetviz as sv
from ai_platform_interface.abstract_musesai_template import AbstractMusesAiFrame

NAME = 'parser_tool.exe'
VERSION = '2.1a'


class ParserTool(AbstractMusesAiFrame):
    def __init__(self, sys_argv):
        super().__init__()
        self.log_folder = '/aiplatform/ExecutiveFile/ParserTool/log'
        self.config_path = '/aiplatform/ExecutiveFile/ParserTool/config.json'
        self.execution(sys_argv)

    @AbstractMusesAiFrame.exception(VERSION)
    def execution(self, sys_argv):
        self.log_msg = self.activate_log()
        params = self.decode_argv(sys_argv)
        config = self.read_config()
        self.ami = self.activate_ami()
        file_path = params['filePath']
        file_name = params['fileName']
        self.redis_head = [self.group_id, self.project_id, file_name]

        self.ami.set_value('_'.join(self.redis_head + ['percent']), 0.32)
        dtypes_file_name = '.'.join([file_name.split('.')[0], 'json'])
        dtypes_path = os.path.join(self.root, 'GroupProject', self.group_id, self.project_id, 'FileHeaderAndType')
        profiling_html_path = os.path.join(self.root, 'UserHTMLGroup', self.group_id, self.project_id, 'AF_parser_result')
        dtype_file = os.path.join(dtypes_path, dtypes_file_name)
        if not os.path.exists(dtypes_path):
            os.makedirs(dtypes_path)
        if not os.path.exists(profiling_html_path):
            os.makedirs(profiling_html_path)
        train = pd.read_csv(os.path.join(file_path, 'temp', file_name), low_memory=False, header=0)

        # Check if header exist
        train_dummy = pd.read_csv(os.path.join(file_path, 'temp', file_name), low_memory=False, header=None)
        train_nunique = train.nunique().reset_index(drop=True)
        train_dummy_nunique = train_dummy.nunique().reset_index(drop=True)
        if train_dummy_nunique.subtract(train_nunique).sum() >= len(train.columns) * 0.5:
            # Move file to datapool from temp
            shutil.move(os.path.join(file_path, 'temp', file_name), os.path.join(file_path, file_name))
            # Delete dummy dataset which build for header check.
            del train_dummy, train_dummy_nunique
            self.ami.set_value('_'.join(self.redis_head + ['percent']), 0.5)
            self.ami.set_value('_'.join(self.redis_head + ['status']), 'Uploading')
            from task.utils.data_type_parser import DataTypeParser

            dtp = DataTypeParser()
            dtype_info = dtp.get_feature_types(train)
            self.ami.set_value('_'.join(self.redis_head + ['percent']), 0.9)
            dtype_info['length'] = train.shape[1]
            with open(dtype_file, 'w') as dp:
                json.dump(dtype_info, dp, indent=4)
            self.ami.set_value('_'.join(self.redis_head + ['status']), 'Complete')
            self.ami.set_value('_'.join(self.redis_head + ['percent']), 1)
            file_size = self.obtain_size(os.path.join(file_path, file_name))
            self.ami.record_volume(exe_name=NAME, file_name=file_name, file_path='-1', file_size=file_size)
            self.log_msg.info('Parsing successfully completed')
        else:
            self.log_msg.error('Detected the data does not have header')
            os.remove(os.path.join(file_path, 'temp', file_name))
            self.log_msg.warning('{} has been removed from temp folder'.format(file_name))
            self.ami.delete_value('_'.join([self.group_id, self.project_id, 'fileUpload']), 0, file_name)
            self.log_msg.warning('Expelled {} from list of redis, key: fileUpload'.format(file_name))
            self.ami.set_value('_'.join(self.redis_head + ['status']), 'Error')
            self.eda(train, file_name)
        return

    def eda(self, df, file_name):
        try:
            profiling_html_path = os.path.join(self.root, 'UserHTMLGroup', self.group_id, self.project_id, 'AF_parser_result')
            if not os.path.exists(profiling_html_path):
                os.makedirs(profiling_html_path)
            self.ami.set_value('_'.join(self.redis_head + ['profiling']), 'Profiling')
            report_train = sv.analyze([df, file_name], pairwise_analysis='off')
            report_train.show_html(filepath=os.path.join(profiling_html_path, '_'.join([file_name, 'profiling.html'])), open_browser=False)
            self.ami.set_value('_'.join(self.redis_head + ['profiling']), 'Completed')
        except Exception as e:
            print("Profiling Error", e)
            self.log_msg.warning('Profiling Error')
            self.log_msg.error(e)
            self.ami.set_value('_'.join(self.redis_head + ['profiling']), 'Error')


if __name__ == '__main__':
    # argv = "e1wiZmlsZU5hbWVcIjpcIk5CX1NBTjVfMjAxOTExXzIwMjEwM18yMDIxMDYzMC5jc3ZcIixcImpvYl9pZFwiOlwiMjAyMTA3MjcwMTE1MDczNTBcIixcImZpbGVQYXRoXCI6XCIvYWlwbGF0Zm9ybS9Hcm91cFByb2plY3QvVzA4UVBUNFBORzJCNlAvcDAwMDA1L0ZpbGVVcGxvYWRBcmVhL1RhYnVsYXIvTkJfU0FONV8yMDE5MTFfMjAyMTAzXzIwMjEwNjMwLmNzdlwiLFwiZ3JvdXBJZFwiOlwiVzA4UVBUNFBORzJCNlBcIixcInVzZXJJZFwiOlwiVzA4UVBUNFBORzJCNlBfZmNmY211c2VhaUBzZXJ2dGVjaC5jb20udHdcIixcInByb2plY3RJZFwiOlwicDAwMDA1XCJ9="
    ParserTool(sys.argv[1])
