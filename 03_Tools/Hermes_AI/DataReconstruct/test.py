import sys
import os
import json
import pandas as pd
from utils.parser_tool import DataTypeParser
from utils.tools import base64_decoder
from operate.combine import combine, label_combine, split
from ai_platform_interface.AIPlatformMessageInterface import AIPlatformMessageInterface


mode = 'combine'
file_dir = "C:\\Users\\admin\\Downloads\\temp"
files_name = ["sk.del.csv", "ml.del.csv"]
split_size = 0.2
shuffle = True
new_name = 'skmldata.csv'
column_remain = []

# ami = AIPlatformMessageInterface(job_id, group_id, project_id)
# ami.start_job()

dtp = DataTypeParser()

# try:
if mode == 'split':
    split_size = split_size
    shuffle = shuffle
    df_raw = pd.read_csv(os.path.join(file_dir, files_name[0]))
    df_1, df_2 = split(df_raw, split_size=split_size, shuffle=shuffle)
    df_1.to_csv(f'{file_dir}/{new_name[0]}', index=False)
    df_2.to_csv(f'{file_dir}/{new_name[1]}', index=False)
    dtp.save_to_json(df_1, file_dir, new_name[0])
    dtp.save_to_json(df_2, file_dir, new_name[1])
elif mode == 'combine':
    df_1 = pd.read_csv(os.path.join(file_dir, files_name[0]))
    df_2 = pd.read_csv(os.path.join(file_dir, files_name[1]))
    df_combined = combine(df_1, df_2)
    df_combined.to_csv(f'{file_dir}/{new_name[0]}', index=False)
    dtp.save_to_json(df_combined, file_dir, new_name[0])
elif mode == 'label_combine':
    df_raw = pd.read_csv(os.path.join(file_dir, files_name[0]))
    df_label = pd.read_csv(os.path.join(file_dir, files_name[1]))
    df_label_combined = label_combine(df_raw, df_label)
    df_label_combined.to_csv(f'{file_dir}/{new_name[0]}', index=False)
    dtp.save_to_json(df_label_combined, file_dir, new_name[0])
elif mode == 'column_filter':
    column = column_remain
    df_raw = pd.read_csv(os.path.join(file_dir, files_name[0]))
    df_filtered = df_raw[column]
    df_filtered.to_csv(f'{file_dir}/{new_name[0]}', index=False)
    dtp.save_to_json(df_filtered, file_dir, new_name[0])
else:
    raise AssertionError("mode errror")

    # ami.complete_job()

# except Exception as err:
#     # ami.handle_error(err)
#     print('error')