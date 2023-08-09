import os
import sys
from traceback import format_exc
import pandas as pd
from utils.parser_tool import DataTypeParser
from ai_platform_interface.abstract_musesai_template import AbstractMusesAiFrame
from operate.combine import combine, label_combine, split, data_leakage

NAME = 'data_reconstruct.exe'
MAJOR = 1
MINOR = 0
PATCH = '20220223'
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"


class DataReconstruct(AbstractMusesAiFrame):
    def __init__(self, sys_argv):
        super().__init__()
        self.log_folder = '/aiplatform/ExecutiveFile/DataReconstruct/log'
        self.config_path = '/aiplatform/ExecutiveFile/DataReconstruct/config.json'
        self.execution(sys_argv)

    @AbstractMusesAiFrame.exception(VERSION)
    def execution(self, sys_argv):
        try:
            self.log_msg = self.activate_log()
            params = self.decode_argv(sys_argv)
            config = self.read_config()
            self.log_msg.info('connect to db and redis')
            self.ami = self.activate_ami()
            mode = params['data']['mode']
            files_name = params['data']['file_name']
            new_name = params['data']['new_name']
            file_dir = os.path.join(self.root, self.group_id, self.project_id, 'Tabular', 'AfterLabelMerge')

            # TODO: This line is irregular solution to prevent job vanish in platform list, should be correct in th future
            self.log_msg.info(f"create dummy csv to '{file_dir}'")
            pd.DataFrame().to_csv(f'{file_dir}/{new_name[0]}', index=False)
            self.log_msg.info('instance DataTypeParser')
            dtp = DataTypeParser()

            self.log_msg.info(f'start reconstruct process, mode: {mode}')
            if mode == 'split':
                split_size = params['data']['split_size']
                shuffle = params['data']['shuffle']
                stratify = params['data'].get("stratify", []) # params['data']['stratify']
                df_raw = pd.read_csv(os.path.join(file_dir, files_name[0]))
                df_1, df_2 = split(df_raw, split_size=split_size, shuffle=shuffle, stratify = stratify)
                df_1.to_csv(f'{file_dir}/{new_name[0]}', index=False)
                df_2.to_csv(f'{file_dir}/{new_name[1]}', index=False)
                dtp.save_to_json(df_1, file_dir, new_name[0])
                dtp.save_to_json(df_2, file_dir, new_name[1])
                self.record_usage(file_dir, new_name[0])
                self.record_usage(file_dir, new_name[1])
                self.record_usage(os.path.join(file_dir, 'ParserResult'), new_name[0].replace('.csv', '.json'))
                self.record_usage(os.path.join(file_dir, 'ParserResult'), new_name[1].replace('.csv', '.json'))

            elif mode == 'combine':
                df_1 = pd.read_csv(os.path.join(file_dir, files_name[0]))
                df_2 = pd.read_csv(os.path.join(file_dir, files_name[1]))
                df_combined = combine(df_1, df_2)
                df_combined.to_csv(f'{file_dir}/{new_name[0]}', index=False)
                dtp.save_to_json(df_combined, file_dir, new_name[0])
                self.record_usage(file_dir, new_name[0])
                self.record_usage(os.path.join(file_dir, 'ParserResult'), new_name[0].replace('.csv', '.json'))

            elif mode == 'label_combine':
                file_dir = os.path.join(self.root, self.group_id, self.project_id, 'FileUploadArea', 'Tabular')
                df_raw = pd.read_csv(os.path.join(file_dir, files_name[0]))
                file_dir = file_dir.replace("FileUploadArea/Tabular", "Tabular/AfterLabelMerge")
                if len(files_name) == 2:
                    df_label = pd.read_csv(os.path.join(file_dir, 'label', files_name[1]))
                    df_label_combined = label_combine(df_raw, df_label)
                    df_label_combined.to_csv(f'{file_dir}/{new_name[0]}', index=False)
                    dtp.save_to_json(df_label_combined, file_dir, new_name[0])
                else:
                    df_raw.to_csv(f'{file_dir}/{new_name[0]}', index=False)
                    dtp.save_to_json(df_raw, file_dir, new_name[0])
                    self.record_usage(file_dir, new_name[0])
                    self.record_usage(os.path.join(file_dir, 'ParserResult'), new_name[0].replace('.csv', '.json'))

            elif mode == 'column_filter':
                column = params['data']['column_remain']
                df_raw = pd.read_csv(os.path.join(file_dir, files_name[0]))
                df_filtered = df_raw[column]
                df_filtered.to_csv(f'{file_dir}/{new_name[0]}', index=False)
                dtp.save_to_json(df_filtered, file_dir, new_name[0])
                self.record_usage(file_dir, new_name[0])
                self.record_usage(os.path.join(file_dir, 'ParserResult'), new_name[0].replace('.csv', '.json'))

            elif mode == 'data_leakage':
                column_sort = params["data"]["column_sort"]
                remove_quantile = params["data"]["remove_quantile"]
                df_raw = pd.read_csv(os.path.join(file_dir, files_name[0]))
                dtype = dtp.get_feature_types(df_raw)
                df_filtered = data_leakage(df_raw, column_sort, remove_quantile, dtype)
                df_filtered.to_csv(f'{file_dir}/{new_name[0]}', index = False)            
                dtp.save_to_json(df_filtered, file_dir, new_name[0])
                self.record_usage(file_dir, new_name[0])
                self.record_usage(os.path.join(file_dir, "ParserResult"), new_name[0].replace(".csv", ".json"))

            else:
                raise AssertionError("mode error")

            self.log_msg.info('data reconstruct completed successfully.')
            self.ami.complete_job()
            self.log_msg.info('DB - job completed.')
            return

        except Exception as err:
            err_msg = format_exc()
            raise err_msg

    def record_usage(self, file_dir, filename):
        file_size = self.obtain_size(os.path.join(file_dir, filename))
        self.ami.record_volume(exe_name='data_reconstruct.exe',
                               file_name=filename,
                               file_path='-1',
                               file_size=file_size)
        self.log_msg.info(f"{filename} usage={file_size}(bytes)")


if __name__ == '__main__':
    DataReconstruct(sys.argv[1])
