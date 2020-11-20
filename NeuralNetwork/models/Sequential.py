from NeuralNetworkProjects.NeuralNetwork.layers.Dense import *
from NeuralNetworkProjects.NeuralNetwork.models.Model import *
from NeuralNetworkProjects.NeuralNetwork.Math_AI import *

import numpy as np
from prettytable import PrettyTable
import random
import copy


class Sequential(Model):
    def __init__(self):
        self.layers = []
        self.input_values = []
        self.expected_output_values = []
        self.mini_batch_size = 0
        self.learning_rate = 0

    def create(self, mini_batch_size=32, learning_rate=0.9):
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate

    def add(self, layer: Dense):
        if len(self.layers) == 0:
            layer.initialize_weights(layer.amount_of_input_neurons)
        else:
            layer.initialize_weights(self.layers[-1].units)
        layer.initialize_biases()
        self.layers.append(layer)

    def summary(self):
        table = PrettyTable()
        table.field_names = ["index", "layer type", "layer units", "weights shape", "biases shape", "param #"]

        sum_of_params = 0
        for index, layer in enumerate(self.layers):

            if len(layer.weights.shape) >= 2:
                nr_of_params_weights = layer.weights.shape[0] * layer.weights.shape[1]
            else:
                nr_of_params_weights = layer.weights.shape[0]

            if len(layer.biases.shape) >= 2:
                nr_of_params_bias = layer.biases.shape[0] * layer.biases.shape[1]
            else:
                nr_of_params_bias = layer.biases.shape[0]

            nr_of_params = nr_of_params_weights + nr_of_params_bias
            sum_of_params += nr_of_params

            # type(layer).__name__
            table.add_row([index, "Dense", layer.units, layer.weights.shape, layer.biases.shape, nr_of_params])

        print(table)
        print("Total params: " + str(sum_of_params))


    # fuctions for training the neural network

    def calc_adjustments_for_last_layer(self, exp_output_data):
        actual_output = self.layers[-1].values

        error = exp_output_data - actual_output
        adjustments = error * Math_AI.sigmoid_derivative(actual_output)
        return adjustments

    def calc_adjustments_for_hidden_layer(self, adjustments, layer_index_reverse):
        error = np.dot(adjustments, self.layers[len(self.layers)-layer_index_reverse].weights.T) # 0
        adjustments_new = error * Math_AI.sigmoid_derivative(self.layers[len(self.layers)-layer_index_reverse-1].values)
        return adjustments_new


    def calc_adjustment(self, exp_output_data, adjustments, layer_index_reversed):
        if layer_index_reversed == 0:
            return self.calc_adjustments_for_last_layer(exp_output_data)
        else:
            return self.calc_adjustments_for_hidden_layer(adjustments, layer_index_reversed)


    def backprop(self, input_data, exp_output_data):
        d_weights = []
        d_biases = []
        adjustments = []

        layers_with_input = copy.deepcopy(self.layers)
        layer = Dense(units=self.layers[0].amount_of_input_neurons, activation="sigmoid")
        layer.values = input_data
        layers_with_input.insert(0, layer)

        for iteration_nr, index in enumerate(range(len(layers_with_input)-2, -1, -1)):
            adjustments = self.calc_adjustment(exp_output_data, adjustments, iteration_nr)

            delta_weight = np.dot(layers_with_input[index].values.T, adjustments)
            delta_bias = np.sum(adjustments, axis=0)

            d_weights.append(delta_weight)
            d_biases.append(delta_bias)

        d_weights.reverse()
        d_biases.reverse()

        for index in range(len(self.layers)):
            self.layers[index].weights += ((self.learning_rate / self.mini_batch_size) * d_weights[index])
            self.layers[index].biases += ((self.learning_rate / self.mini_batch_size) * d_biases[index])


    def calc_feedforward(self, input_data):
        for layer_id in range(len(self.layers)-1):
            if layer_id == 0:
                self.layers[layer_id].compute(input_data)

            self.layers[layer_id+1].compute(self.layers[layer_id].values)



    def get_mini_batch_data(self, input_data, exp_output_data):
        input_data = copy.deepcopy(input_data)
        exp_output_data = copy.deepcopy(exp_output_data)

        random_nr = random.randint(0, 20) + 1
        random.seed(random_nr)
        random.shuffle(input_data)
        random.seed(random_nr)
        random.shuffle(exp_output_data)
        input_mini_batch = input_data[:self.mini_batch_size]
        exp_output_mini_batch = exp_output_data[:self.mini_batch_size]
        return input_mini_batch, exp_output_mini_batch


    def train(self, input_values, expected_output_values, epochs=10):
        self.input_values = copy.deepcopy(input_values)
        self.expected_output_values = copy.deepcopy(expected_output_values)

        input_data = copy.deepcopy(input_values)
        exp_output_data = copy.deepcopy(expected_output_values)

        batches = []

        # create batches
        for i in range(int(len(self.input_values)/self.mini_batch_size)):
            input_mini_batch, exp_output_mini_batch = self.get_mini_batch_data(input_data, exp_output_data)
            input_data = input_data[self.mini_batch_size:]
            exp_output_data = exp_output_data[self.mini_batch_size:]
            batches.append((input_mini_batch, exp_output_mini_batch))

        # train network with mini batches and backpropagation
        for epoch in range(epochs):
            for input_d, output_d in batches:
                self.calc_feedforward(input_d)
                self.backprop(input_d, output_d)


    # functions for testing the neural network

    def get_index_of_max_in_arr(self, arr):
        max = 0
        index = -1
        for i in range(len(arr)):
            if arr[i] > max:
                max = arr[i]
                index = i

        return index

    def get_index_of_number_one_in_arr(self, arr):
        for i in range(len(arr)):
            if arr[i] == 1:
                return i

        return -1


    def test(self, input_data, output_data):

        count = 0
        error = 0
        error2 = 0
        amount = len(input_data)

        for i in range(amount):
            self.calc_feedforward(input_data[i])
            output_layer = self.layers[-1].values.T
            exp_output = output_data[i]

            diff = exp_output - output_layer.T
            diff = np.absolute(diff.T)
            sum = np.sum(diff, axis=0)
            error += sum
            error2 += np.mean(np.power(diff, 2))

            max_id = self.get_index_of_max_in_arr(output_layer)
            exp_id = self.get_index_of_number_one_in_arr(exp_output)

            if max_id == exp_id:
                count += 1

        acc = count / amount
        error2 /= amount

        return error[0], error2, acc * 100