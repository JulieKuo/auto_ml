import logging
import base64
import re

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, filename='advisor.log', filemode='w', format=FORMAT)


def write_to_log(message, level):
    if level == 'info':
        logging.info(message)
    elif level == 'error':
        logging.error(message)


def base64_decoder(base64_str):
    base64_bytes = base64_str.encode("UTF-8")
    string_bytes = base64.b64decode(base64_bytes)
    string = string_bytes.decode("UTF-8")
    arguments = string.replace('\\\\', '#').replace('\\', '').replace('#', '\\')
    # logging.info('Arguments received: {}'.format(arguments))
    arguments = eval(arguments)
    return arguments


def not_character(col_name):
    for t in col_name:
        # \u4e00-\u9fa5 chinese
        # \u3040-\u309f japanese hiragana
        # \u30a0-\u30ff japanese katakana
        s = re.match('[\u4e00-\u9fa5\u30a0-\u30ff\u3040-\u309f]', t)
        if s:
            return True
    return False