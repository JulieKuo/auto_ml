class CustomizedException(Exception):
    pass

def raiseCustomizedException(err_message):
    raise CustomizedException(err_message)
