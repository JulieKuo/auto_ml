import json
from customized_exception import raiseCustomizedException

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            conf = json.load(f)
    except FileNotFoundError:
        raiseCustomizedException(f'File not found at {filepath}')
    return conf

def dump_json(filepath, data):
    try:
        with open(filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=4, ensure_ascii=False)
    except Exception as e:
        raiseCustomizedException(f'Failed to dump data into json file. ({type(e)})')
    return
