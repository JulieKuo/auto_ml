from tensorflow.keras.models import load_model
from autokeras import CUSTOM_OBJECTS


def loading_model(filepath):
    model = load_model(filepath, custom_objects=CUSTOM_OBJECTS)
    return model
