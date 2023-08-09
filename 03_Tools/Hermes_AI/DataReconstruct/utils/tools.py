import base64
import json
import os


def base64_decoder(base64_str):
    base64_bytes = base64_str.encode("UTF-8")
    string_bytes = base64.b64decode(base64_bytes)
    string = string_bytes.decode("UTF-8")
    arguments = string.replace('\\\\', '#').replace('\\', '').replace('#', '\\')
    arguments = eval(arguments)
    return arguments


def check_size(path):
    # get size
    size = 0
    try:
        for ele in os.scandir(path):
            size += os.path.getsize(ele)
    except NotADirectoryError:
        size = os.path.getsize(path)

    return size
