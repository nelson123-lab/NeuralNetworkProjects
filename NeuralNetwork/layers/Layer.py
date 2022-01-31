from abc import ABC, abstractmethod
from NeuralNetworkProjects.NeuralNetwork.ActivationFunctions import *


class Layer(ABC):

    @abstractmethod
    def __init__(self):
        self.activation = None
        self.activation_prime = None
        self.activation_name = None

        self.input_dim = None

    @abstractmethod
    def initialize_weights(self, units_of_layer_before):
        pass

    @abstractmethod
    def initialize_biases(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    def set_activation(self, activation):
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_prime = sigmoid_derivative
        if activation == "relu":
            self.activation = relu
            self.activation_prime = relu_derivative
        elif activation == "leaky_relu":
            self.activation = leaky_relu
            self.activation_prime = leaky_relu_derivative
        elif activation == "softmax":
            self.activation = softmax

        if activation not in ["sigmoid", "relu", "leaky_relu", "softmax"]:
            raise ValueError("ERROR: the given activation function is not available.")

        self.activation_name = activation

