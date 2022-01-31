from NeuralNetworkProjects.NeuralNetwork.layers.Layer import *
from NeuralNetworkProjects.NeuralNetwork.Utils import *

import numpy as np

"""
As in Keras, Dense is a simple fully connected layer.
"""


class Dense(Layer):
    def __init__(self, units, activation, input_dim=0):
        """
        :param units: the amount of neurons in this layer
        :param activation: the activation function for this layer
        :param input_dim: the amount of input neurons; should only be set for the first layer
        """

        self.s = np.array([])  # preactivation values
        self.a = np.array([])  # activation values
        self.W = np.array([])  # weights of layer
        self.b = np.array([])  # bias of layer

        self.dZ = 0.0
        self.delta = 0.0

        self.units = units
        self.input_dim = input_dim
        self.is_output_layer = False

        self.set_activation(activation)

    def initialize_weights(self, units_of_layer_before):
        """
        :param units_of_layer_before: amount of neurons of the layer before

        For the RELU activation function we need smaller weights.
        We use the "He Weight Initialization" (* sqrt(2/n) where n is the amount of neurons of the layer before)
        """

        if self.activation_name in ["relu", "leaky_relu"]:
            self.W = (2 * np.random.random((units_of_layer_before, self.units)) - 1) \
                           * np.sqrt(2/units_of_layer_before)
        else:
            self.W = (2 * np.random.random((units_of_layer_before, self.units)) - 1)
            
    def initialize_biases(self):
        """
        For the RELU activation function we need smaller biases, too.
        """

        if self.activation_name in ["relu", "leaky_relu"]:
            self.b = (2 * np.random.random((1, self.units)) - 1) * 0.01
        else:
            self.b = (2 * np.random.random((1, self.units)) - 1)

    def forward(self, values_of_layer_before):
        """
        :param values_of_layer_before: the values of the layer before which we need to calculate the new values

        First we calculate the linear part: s = weights.T * values_of_layer_before + biases
        Then we calculate the values of the neurons with the activation function of the layer and the linear values.
        """
        s = np.dot(values_of_layer_before, self.W) + self.b

        self.a = self.activation(s)

        if not self.is_output_layer:
            self.dZ = self.activation_prime(s)

        return self.a

    def backward_(self, y, right_layer):
        if self.is_output_layer:
            error = self.a - y
            self.delta = error / y.shape[0]
        else:
            self.delta = np.atleast_2d(np.dot(right_layer.delta, right_layer.W.T) * self.dZ)

        return self.delta

    def backward(self, y_or_output_gradient, right_layer):
        if self.is_output_layer:
            output_gradient = self.a - y_or_output_gradient
            self.delta = output_gradient / y_or_output_gradient.shape[0]
        else:
            self.delta = np.atleast_2d(np.dot(right_layer.delta, right_layer.W.T) * self.dZ)

        return self.delta

    def update(self, learning_rate, left_a):
        a = np.atleast_2d(left_a)
        d = np.atleast_2d(self.delta)
        ad = a.T.dot(d)
        self.W -= learning_rate * ad
        self.b -= learning_rate * np.sum(self.delta, axis=0)
