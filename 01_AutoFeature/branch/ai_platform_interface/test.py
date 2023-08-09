from AIPlatformMessageInterface import AIPlatformMessageInterface, raiseCustomException
import requests.exceptions
import sys
from time import sleep


# job_id = '20200526134229870'
job_id = '20200527101213392'
userId = 'userstd'


def test_raise():
    # raise ValueError
    raise requests.exceptions.RequestException


try:
    temp = AIPlatformMessageInterface(job_id, userId)

    temp.close_connection()
    temp.start_job()
    sleep(5)
    temp.complete_job()

    key = 'test-0617'
    print(key, temp.get_value(key))

    key = 'test-0928'
    temp.set_value(key, '0953')
    print(key, temp.get_value(key))

    temp.set_value(key, '0953')
    print(key, temp.get_value(key))

    key = 'test-0928-ls'
    temp.add_list_value(key, '0953')
    print(key, temp.get_list_value(key))

    # test_raise()
    raiseCustomException("write error message")

except Exception as e:
    print(e, type(e))
    # print('caught!')
    if not temp.handle_error(e):
        # raise
        pass

print('--------------------')

temp.complete_job()
