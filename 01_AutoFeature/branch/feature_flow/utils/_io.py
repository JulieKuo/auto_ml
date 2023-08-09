import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

__all__ = [
    'JsonTool',
    'Logger',
]


class JsonTool(object):
    """docstring for JsonTool"""

    def __init__(self, filepath):
        super(JsonTool, self).__init__()

    @staticmethod
    def load(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return d

    def read_file(self):
        try:
            self._config_dict = self.load_json(self._filepath)
        except FileNotFoundError:
            self.initialize_config_dict()
            self.load_conf_folder()
            self.update_file()

    @staticmethod
    def save(content, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=4)

    def update_file(self):
        with open(self._filepath, 'w', encoding='utf-8') as f:
            json.dump(self._config_dict, f, indent=4)


class Logger(object):
    """docstring for LogWriter"""
    def __init__(self, path, name):
        super(Logger, self).__init__()

        self.log_path = path
        self.log_name = name

        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        FORMAT = '%(asctime)s %(levelname)s %(message)s'
        handler = TimedRotatingFileHandler(
            f"{self.log_path}/{self.log_name}", when="midnight", interval=1, backupCount=10,
            encoding="UTF-8", delay=False, utc=True)

        formatter = logging.Formatter(FORMAT)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
