import os
import sys
import glob
import json
import shutil
import argparse
import subprocess
import traceback
from pathlib import Path

import pandas as pd

from ai_platform_interface.abstract_musesai_template import AbstractMusesAiFrame
from feature_flow.flow import Pipeline
from feature_flow.utils import LoadData

NAME = 'AutomaticFeatureExtraction.exe'
MAJOR = 1
MINOR = 1
PATCH = '20220725'
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", help="undergo system tests", action="store_true")
parser.add_argument("task", help="determine which task would be predict", nargs='?', type=str,)
args = parser.parse_args()
# args = parser.parse_args(args = [])


class AutomaticFeatureExtraction(AbstractMusesAiFrame):
    def __init__(self):
        super().__init__()
        self.log_folder = './log'
        self.log_name = 'auto_feats_extract.log'
        self.config_path = './config.json'
        self.dtypes_dict = None
        if args.test:
            print("Running undergo system tests...")
            self._undergo_system_tests()
        else:
            assert args.task, 'missing 1 required argument'
            self.execution(args.task)

    @AbstractMusesAiFrame.exception(VERSION)
    def execution(self, sys_argv):
        self.log_msg = self.activate_log()
        arguments = self.decode_argv(sys_argv)
        config = self.read_config()
        # Check if old version
        self.root_dir = Path(self.root) / self.group_id / self.project_id
        if 'modelName' in arguments:
            search_pattern = os.path.join(self.root_dir, 'AutoFeatureEngineering', f"*{self.mold_job_id}", '*mold.mo')
            relative_mold_path = glob.glob(search_pattern)
            if not relative_mold_path:
                subprocess.Popen(["Autofs_AIP_previous.exe", sys_argv[0]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print('Reassign job to old version...')
                return

        self.ami = self.activate_ami()
        self.ami.start_job()

        try:
            filename = arguments['fileName'].split('|')
        except KeyError:
            filename = ["NONE", arguments['predict_file']]

        self.redis_head = [self.group_id, self.project_id, 'AF', filename[0] + '|' + filename[1], self.job_id]
        self.file_head = '_'.join([self.job_id, self.group_id, self.project_id])

        file_dir = self.root_dir / 'Tabular' / 'AfterLabelMerge'

        ld = LoadData()
        train = ld.load(file_dir / filename[0]) if filename[0] != 'NONE' else None
        test = ld.load(file_dir / filename[1]) if filename[1] != 'NONE' else None

        ff = Pipeline(logger=self.log_msg, callback=self.report_progress)
        if config['CUSTOM_PARAMETER']:
            params = config['CUSTOM_PARAMETER']
        else:
            params = arguments.get('params', {})
        ff.setup(**params)

        save_dir = self.root_dir / 'AutoFeatureEngineering' / '_'.join(filename + [self.job_id])
        save_dir.mkdir(parents=True, exist_ok=True)
        report_file = f"{self.file_head}_report.txt"
        sys.stdout = open(save_dir / report_file, 'w')

        if 'fileName' in arguments:
            dtypes_path = self.root_dir / 'FileHeaderAndType' / f"{filename[0]}_{self.job_id}.json"
            dtypes_dict = self.read_dtypes(dtypes_path)

            label = arguments['label']
            train_ff = ff.cast(train, label, test, col_type=dtypes_dict)
            ff.save_mold(save_dir / '_'.join([self.file_head, 'mold.mo']))
            self.write_to_csv(train_ff, filename[0], save_dir)
            self.save_dtypes(train_ff, filename[0], label)

            if test is not None:
                test_ff = ff.shape(test)
                self.write_to_csv(test_ff, filename[1], save_dir)
                self.save_dtypes(test_ff, filename[1], label)

        elif 'modelName' in arguments:
            predict_job_id = arguments['predict_job_id']
            ami_predict = self.activate_custom_ami(predict_job_id, self.user_id, self.group_id, self.project_id, self.database_config, self.redis_config, self.program_user)

            # Find mold of previous FF job
            search_pattern = os.path.join(self.root_dir, 'AutoFeatureEngineering', f"*{self.mold_job_id}", '*mold.mo')
            relative_mold_path = glob.glob(search_pattern)
            if not relative_mold_path:
                raise ValueError(f'1005@{self.mold_job_id}')
            mold_path = relative_mold_path[0]
            ff.load_mold(mold_path)
            label = ff._meta['synthesis']['label']

            # Shape data by mold info
            shaped = ff.shape(test)
            self.file_head = '_'.join([self.mold_job_id, self.group_id, self.project_id])  # Temp. should remove this line
            self.write_to_csv(shaped, filename[1], save_dir)
            self.save_dtypes(shaped, filename[1], label)
            ami_predict.activate_job()

        sys.stdout.close()
        zipfile_name = '_'.join(filename)
        self.file_zipper(save_dir, zipfile_name)
        self.record_usage(save_dir)
        self.log_msg.info('AutomaticFeatureExtraction completed successfully.')
        self.ami.complete_job()
        self.log_msg.info('DB - job completed.')
        return

    def record_usage(self, file_dir):
        folder_size = self.obtain_size(file_dir)
        self.ami.record_volume(exe_name='Autofs_AIP.exe',
                               file_name="-1",
                               file_path=file_dir.stem,
                               file_size=folder_size)

    def write_to_csv(self, x, file_name, save_dir):
        self.log_msg.info("Writing dataset...")
        save_name = '_'.join([self.file_head, 'AF', file_name])
        x.to_csv((save_dir / save_name), index=False)
        shutil.copyfile((save_dir / save_name), (self.root_dir / 'Tabular' / 'AfterLabelMerge' / save_name))
        print('Completed. {} shape: {}'.format(file_name, x.shape))

    def read_dtypes(self, dtypes_path):
        assert os.path.exists(dtypes_path), AssertionError(f"3501@")
        with open(dtypes_path, 'r', encoding="UTF-8") as md:
            datatype = json.load(md)
        dtypes_dict = dict()
        for d in datatype.values():
            for_process = 'datetime' if d['dtype_for_process'][0] == "%" else d['dtype_for_process']
            dtypes_dict[d['column_name']] = for_process
        self.log_msg.info('Received FileHeaderAndType')
        return dtypes_dict

    def save_dtypes(self, x, file_name, label):
        save_name = '_'.join([self.file_head, 'AF', file_name])
        dtypes_path = self.root_dir / 'FileHeaderAndType' / save_name.replace('.csv', '.json')
        dtypes_dict = dict()
        dtypes_dict[0] = {"column_name": label}
        dtypes_dict['length'] = 1
        dtypes_dict['col_types'] = {col: 'enum' if x[col].dtype.name == 'category' else 'numeric' for col in x}
        with open(dtypes_path, 'w') as dp:
            json.dump(dtypes_dict, dp, indent=4)

    def _undergo_system_tests(self):
        try:
            self._static_fire()
            # os.system('cls')
            print("Checking AF... OK")
        except:
            print("Checking AF... NG")
            pass
        finally:
            self._test_connection()

    def _static_fire(self):
        try:
            self.root = './test'
            self.config_path = './test/config.json'
            self.log_msg = self.activate_log()
            dtypes_dict = self.read_dtypes('./test/type.json')
            config = self.read_config()
            params = config['CUSTOM_PARAMETER']
            train = pd.read_csv('./test/mf_train.csv')
            test = pd.read_csv('./test/mf_test.csv')
            ff = Pipeline()
            print("Checking Pipeline... ")
            print(" setup... ", end="")
            ff.setup(**params)
            print("OK")

            print(" cast... ", end="")

            ff.cast(train, label='Fail_tomorrow', col_type=dtypes_dict)
            print("OK")

            print(" save... ", end="")
            ff.save_mold('./test/test.mo')
            print("OK")

            print(" init... ", end="")
            ff.init()
            print("OK")

            print(" load... ", end="")
            ff.load_mold('./test/test.mo')
            print("OK")

            print(" shape... ", end="")
            ff.shape(test)
            print("OK")

        except Exception as e:
            exc_str = traceback.format_exc()
            print(exc_str)
            raise e

    @staticmethod
    def _main_engine_cut_off():
        print("Main Engine Cut Off")


if __name__ == '__main__':
    AutomaticFeatureExtraction()
