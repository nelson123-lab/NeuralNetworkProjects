from abc import ABC, abstractmethod

class Layer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize_weights(self, units_of_layer_before):
        pass

    @abstractmethod
    def initialize_biases(self):
        pass

    @abstractmethod
    def compute(self):
        pass
