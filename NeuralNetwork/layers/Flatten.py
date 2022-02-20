from NeuralNetworkProjects.NeuralNetwork.layers.Layer import *
import numpy as np

"""
A layer to flatten the output of an convolutional layer.
"""


class Flatten(Layer):
    def __init__(self):
        self.a = np.asarray([])
        self.activation = ""
        self.activation_prime = ""
        self.units = -1
        self.W = np.asarray([])
        self.b = np.asarray([])

        self.delta = np.asarray([])
        self.dZ = np.asarray([])

        self.layer_type = "Flatten"

        self.original_shape = ()

    def initialize_weights(self, units_of_layer_before):
        pass

    def initialize_biases(self):
        pass

    def forward(self, values_of_layer_before):
        self.original_shape = (values_of_layer_before.shape[0], values_of_layer_before.shape[1],
                               values_of_layer_before.shape[2], values_of_layer_before.shape[3])

        self.a = values_of_layer_before.reshape(
            (values_of_layer_before.shape[0],
             values_of_layer_before.shape[1] * values_of_layer_before.shape[2] * values_of_layer_before.shape[3]))

        self.units = values_of_layer_before.shape[1] * \
                     values_of_layer_before.shape[2] * values_of_layer_before.shape[3]

        return self.a

    def backward(self, delta, right_layer):
        self.delta = delta.reshape(self.original_shape)
        return self.delta

    def update(self, learning_rate, left_a):
        pass