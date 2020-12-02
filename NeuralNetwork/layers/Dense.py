from NeuralNetworkProjects.NeuralNetwork.layers.Layer import *
from NeuralNetworkProjects.NeuralNetwork.ActivationFunctions import *
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
        

        self.values = np.array([])
        self.weights = np.array([])
        self.biases = np.array([])

        self.units = units
        self.activation = activation
        self.derivative_activation = ""
        self.input_dim = input_dim


        
    def initialize_weights(self, units_of_layer_before):
        """
        :param units_of_layer_before: amount of neurons of the layer before

        For the RELU activation function we need smaller weights, in order to avoid exploding gradients.
        We use the "He initialization" (specifically +/- sqrt(2/n) where n is the amount of neurons of the layer before)
        """
        

        if self.activation == "relu":
            self.weights = (2 * np.random.random((units_of_layer_before, self.units)) - 1) \
                           * np.sqrt(2/units_of_layer_before)
        else:
            self.weights = (2 * np.random.random((units_of_layer_before, self.units)) - 1)


            
    def initialize_biases(self):
        """
        For the RELU activation function we need smaller biases, too.
        """

        
        if self.activation == "relu":
            self.biases = (2 * np.random.random((1, self.units)) - 1) * 0.01
        else:
            self.biases = (2 * np.random.random((1, self.units)) - 1)

            

    def compute(self, values_of_layer_before):
        """
        :param values_of_layer_before: the values of the layer before which we need to calculate the new values

        First we calculate linear: values = values_of_layer_before * weights + biases.
        Then we calculate the values of the neurons with the activation function of the layer and the linear values.
        """

        
        if self.activation == "sigmoid":
            self.values = sigmoid(multiply(values_of_layer_before, self.weights) + self.biases)
        if self.activation == "relu":
            self.values = relu(multiply(values_of_layer_before, self.weights) + self.biases)
        if self.activation == "softmax":
            self.values = softmax(multiply(values_of_layer_before, self.weights) + self.biases)

            
        """
        np.isnan(np.max(x)) is a good way to check if there is any number which is np.nan in the array. 
        This is important to know because than you most likely have exploding gradients. 
        
        This means that calculated numbers are so high that they cannot be stored as this datatype anymore and 
        then turn into np.nan which also effects the accuracy of the neural network. 
        Since the activation function sigmoid minimizes every value and the softmax activation function 
        just calculates the probabilities this problem only concerns layers with the RELU activation function. 
        The RELU activation function and also leaky RELU do not minimize the numbers greater than zero, 
        so you need to normalize the input values and set the weights and biases to a small value 
        in order to keep the numbers as small as possible.
        """

        
        if np.isnan(np.max(self.values)) or np.isnan(np.max(self.weights)) or np.isnan(np.max(self.biases)):
            print("Exploding Gradients WARNING: if this warning shows up, it means that you have dead neurons in your "
                  "neural network due to high input values, weights or biases. "
                  "This can affect the accuracy of the neural network.")

        if self.activation != "sigmoid" and self.activation != "relu" and self.activation != "softmax":
            raise ValueError("ERROR: the given activation function is not avialable.")
