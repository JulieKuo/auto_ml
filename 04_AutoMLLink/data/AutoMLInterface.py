import random  
from abc import ABCMeta, abstractmethod


class AutoMLInterface(metaclass=ABCMeta):
    @abstractmethod
    def train(self, filepath, target):
        pass

    @abstractmethod
    def predict(self, filepath):
        pass
