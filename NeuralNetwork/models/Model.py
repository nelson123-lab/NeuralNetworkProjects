from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
