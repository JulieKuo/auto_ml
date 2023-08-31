import random  
from abc import ABCMeta, abstractmethod  
  

class MessageInterface(metaclass=ABCMeta):
    @abstractmethod
    def get_value(self, key):
        pass

    @abstractmethod
    def set_value(self, key, value):
        pass
        
    @abstractmethod
    def get_list_value(self, key, start=0, end=-1):
        pass
        
    @abstractmethod
    def add_list_value(self, key, value):
        pass
        
    @abstractmethod
    def build_connection(self):
        pass
    
    @abstractmethod
    def close_connection(self):
        pass
    
    @abstractmethod
    def check_connection(self):
        pass

    @abstractmethod
    def start_job(self):
        pass
        
    @abstractmethod
    def complete_job(self):
        pass
    
    @abstractmethod
    def handle_error(self, e, ignore_most_exceptions):
        pass
