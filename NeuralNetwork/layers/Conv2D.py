from NeuralNetworkProjects.NeuralNetwork.layers.Layer import *

import numpy as np
import math
from scipy import signal

"""
A convolutional layer for 2D input values
"""


class Conv2D(Layer):
    def __init__(self, input_channels, nr_of_filters, kernel_size=(2, 2), strides=(1, 1),
                 padding="same", activation="relu", input_shape=(28, 28, 1)):
        """
        :param units: the amount of units in this layer
        :param activation: the activation function for this layer
        :param input_dim: the amount of input units; should only be set for the first layer
        """

        self.a = np.array([])  # = activation/output values
        self.W = np.array([])  # = filters
        self.b = np.array([])  # = biases

        self.x = np.array([])  # = input values
        self.filter_gradients = np.array([])
        self.input_gradients = np.array([])
        self.delta = np.array([])
        self.bias_delta = np.array([])

        self.number_of_filters = nr_of_filters
        self.kernel_size = kernel_size
        self.strides = strides
        # for now, only option "same" works (for kernel_sizes where the difference between x and y is at most 1)
        self.padding = padding

        self.input_channels = input_channels
        self.input_shape = input_shape
        self.input_dim = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        self.units = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * self.number_of_filters)

        self.pad_x = (math.ceil((self.kernel_size[0] - 1) / 2), math.floor(((self.kernel_size[0]) - 1) / 2))
        self.pad_y = (math.ceil((self.kernel_size[1] - 1) / 2), math.floor(((self.kernel_size[1]) - 1) / 2))
        self.pad_width = ((0, 0), self.pad_y, self.pad_x)
        self.input_shape_with_padding = list(self.input_shape)
        self.input_shape_with_padding[-2] += (self.pad_x[0] + self.pad_x[1])
        self.input_shape_with_padding[-3] += (self.pad_y[0] + self.pad_y[1])

        self.set_activation(activation)

    def initialize_weights(self, units_of_layer_before):
        """
        :param units_of_layer_before: amount of neurons of the layer before

        For the RELU activation function we need smaller weights, in order to avoid exploding gradients.
        We use the "He initialization".
        """

        if self.activation_name in ["relu", "leaky_relu"]:
            filters = (2 * np.random.random(
                size=(self.number_of_filters, self.input_channels, self.kernel_size[0], self.kernel_size[1])) - 1) \
                * np.sqrt(2 / units_of_layer_before)
        else:
            filters = (2 * np.random.random(
                size=(self.number_of_filters, self.input_channels, self.kernel_size[0], self.kernel_size[1])) - 1)

        self.W = np.array(filters)

    def initialize_biases(self):
        """
        For the RELU activation function we need smaller biases, too.
        """

        shape = (self.number_of_filters, self.input_shape_with_padding[-2] - self.kernel_size[0] + 1,
                 self.input_shape_with_padding[-3] - self.kernel_size[1] + 1)

        if self.activation_name in ["relu", "leaky_relu"]:
            self.b = (2 * np.random.random(shape) - 1) * 0.01
        else:
            self.b = (2 * np.random.random(shape) - 1)

    def forward(self, values_of_layer_before):

        if values_of_layer_before.shape[-1] == 1:
            values_of_layer_before = values_of_layer_before.reshape(values_of_layer_before.shape[:-1])

        if len(values_of_layer_before.shape) <= 2:
            values_of_layer_before = values_of_layer_before\
                .reshape((values_of_layer_before.shape[0], self.input_shape[0], self.input_shape[1]))

        shape = list(values_of_layer_before.shape)

        if len(values_of_layer_before.shape) < 4:
            shape.insert(1, self.number_of_filters)
            values_with_padding = \
                np.pad(values_of_layer_before, pad_width=self.pad_width, mode='constant', constant_values=0.0)

            self.x = values_with_padding

            self.a = np.zeros(shape=shape)
            self.a[:] = np.copy(self.b)

            for batch_sample_nr in range(values_of_layer_before.shape[0]):
                for i in range(self.number_of_filters):
                    for j in range(self.input_channels):
                        self.a[batch_sample_nr, i] += signal.correlate2d(self.x[batch_sample_nr], self.W[i, j], "valid")
        else:
            pad_width = self.pad_width
            pad_width = (pad_width[0], (0, 0), pad_width[1], pad_width[2])
            values_with_padding = \
                np.pad(values_of_layer_before, pad_width=pad_width, mode='constant', constant_values=0.0)

            self.x = values_with_padding

            self.a = np.zeros(shape=shape)
            self.a[:] = np.copy(self.b)

            for batch_sample_nr in range(values_of_layer_before.shape[0]):
                for i in range(self.number_of_filters):
                    for j in range(self.input_channels):
                        self.a[batch_sample_nr, i] += signal.correlate2d(self.x[batch_sample_nr, j], self.W[i, j], "valid")

        self.a = self.activation(self.a)
        return self.a

    def backward(self, delta, right_layer, learning_rate=0.08):
        """
                :param y: the output
                :param right_layer: the layer we get the gradient from / the layer to the right
                :return: self.delta, which in this case consists of the gradient of the loss with respect to the filters
                         and the gradients of the loss with respect to the inputs which has to be backpropagated.

                We have to calculate 2 gradients:
                    1. The gradient of the loss with respect to the filters
                    2. The gradient of the loss with respect to the input

                For both gradients we need the loss gradient from the previous layer (right_layer)
                which will then be multiplied by the local gradient (= Chain rule).
                This means we need two different local gradients:
                    1.  The gradient of the output with respect to the filters, which equals the values of the inputs/x
                        which the filter got multiplied by.

                        For example:
                        Consider a (forward) convolution with a 2x2 filter F.
                        The result of applying this filter on input matrix X just once is the output O_11:
                            O_11 = X_11 * F_11 + X_12 * F_12 + X_21 * F_21 + X_22 * F_22

                        Then the gradient of the output O_11 with respect to the filter equals
                        a matrix consisting of X_11, X_12, X_21, X_22 due to:

                            dO_11 / dF_11 = X_11
                            dO_12 / dF_12 = X_12
                            dO_21 / dF_21 = X_21
                            dO_22 / dF_22 = X_22

                    2.  The gradient of the output with respect to the inputs, which equals the values of the filters the
                        inputs get multiplied by.

                        Let's consider example from 1. and now let's take the gradient with respect to the inputs X.
                        Then:

                            dO_11 / dX_11 = F_11
                            dO_12 / dX_12 = F_12
                            dO_21 / dX_21 = F_21
                            dO_22 / dX_22 = F_22

                The total gradients then can be calculated by:
                    1. a convolution between input X and the loss gradient with respect to the output.
                    2. a convolution between the 180° rotated filter matrix and the loss gradient with respect to the output.

                """

        # TODO: generalize this code such that you can have more than one Conv layer in your network.

        shape = [self.x.shape[0]] + list(self.W.shape)
        kernels_gradient = np.zeros(shape)
        input_gradient = np.zeros((self.x.shape[0], self.input_channels, delta.shape[-1]+4, delta.shape[-2]+4))

        x_shape = (self.input_shape[0], self.input_shape[1])

        output_gradient = right_layer.delta
        nr_elements = len(output_gradient.reshape(-1))

        if nr_elements / (self.number_of_filters * self.x.shape[0]) != (x_shape[0] * x_shape[1]):
            x_shape = (self.input_shape_with_padding[0], self.input_shape_with_padding[1])

        output_gradient = right_layer.delta.reshape(
            (self.number_of_filters, self.x.shape[0], x_shape[0], x_shape[1])
        )

        for batch_sample_nr in range(output_gradient.shape[1]):
            for i in range(self.number_of_filters):
                for j in range(self.input_channels):
                    if len(self.x.shape) < 4:
                        kernels_gradient[batch_sample_nr, i, j] = signal.correlate2d(
                            self.x[batch_sample_nr], output_gradient[i, batch_sample_nr], "valid")
                    else:
                        kernels_gradient[batch_sample_nr, i, j] = signal.correlate2d(
                            self.x[batch_sample_nr, j], output_gradient[i, batch_sample_nr], "valid")

                    tz = signal.convolve2d(output_gradient[i, batch_sample_nr], self.W[i, j], "full")

                    input_gradient[batch_sample_nr, j] += tz

        kernels_gradient = kernels_gradient.sum(axis=0)

        self.delta = self.activation_prime(input_gradient)
        self.W -= learning_rate * self.activation_prime(kernels_gradient)
        self.b -= learning_rate * np.sum(output_gradient, axis=1)
        return self.delta

    def update(self, learning_rate, left_a):
        pass