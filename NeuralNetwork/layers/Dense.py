from NeuralNetworkProjects.NeuralNetwork.layers.Layer import *
from NeuralNetworkProjects.NeuralNetwork.Math_AI import *

import numpy as np


class Dense(Layer):
    def __init__(self, units, activation="sigmoid", amount_of_input_neurons=0):
        self.values = []
        self.weights = np.array([])
        self.biases = np.array([])

        self.units = units
        self.activation = activation
        self.amount_of_input_neurons = amount_of_input_neurons

    def initialize_weights(self, units_of_layer_before):
        self.weights = 2 * np.random.random((units_of_layer_before, self.units)) - 1

    def initialize_biases(self):
        self.biases = 2 * np.random.random((1, self.units)) - 1

    def compute(self, values_of_layer_before):
        if self.activation == "sigmoid":
            self.values = (Math_AI.sigmoid(np.dot(values_of_layer_before, self.weights) + self.biases))
        if self.activation == "relu":
            pass