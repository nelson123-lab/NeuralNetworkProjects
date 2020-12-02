from NeuralNetworkProjects.NeuralNetwork.layers.Dense import *
from NeuralNetworkProjects.NeuralNetwork.models.Model import *
from NeuralNetworkProjects.NeuralNetwork.ActivationFunctions import *
from NeuralNetworkProjects.NeuralNetwork.LossFunctions import *
from NeuralNetworkProjects.NeuralNetwork.Utils import *

import sys
import numpy as np
from prettytable import PrettyTable


class Sequential(Model):
    def __init__(self):
        self.layers = []
        self.input_values = []
        self.expected_output_values = []
        self.mini_batch_size = 0
        self.learning_rate = 0
        self.loss_function = ""
        self.loss = 0

        
        
    def compile(self, loss, mini_batch_size=32, learning_rate=0.08):
        """
        :param loss: the loss function you want to use
        :param mini_batch_size: the mini-batch-size you want to use (optionally, standard=32)
        :param learning_rate: the learning_rate you want to use (optionally, standard=0.08)
        """
                

        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.loss_function = loss

        
        
    def add(self, layer: Layer):
        """
        :param layer: the layer to add to the model

        This function adds the layer to the model and initializes its weights and biases.
        Furthermore it sets the derivative-activation function of the previous layer to the activation function
        of the current layer. (This is useful for backpropagation because the weights and biases of layer x
        get calculated with the derivative-activation function of the layer x+1.)
        """
        

        if len(self.layers) == 0:
            layer.initialize_weights(layer.input_dim)
        else:
            layer.initialize_weights(self.layers[-1].units)
        layer.initialize_biases()

        if len(self.layers) > 0:
            self.layers[-1].derivative_activation = layer.activation
        self.layers.append(layer)


                
    def summary(self):
        """
        This function prints a summary of the model.
        """
        

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
            table.add_row([index, type(layer).__name__, layer.units, layer.weights.shape,
                           layer.biases.shape, nr_of_params])
        print(table)
        print("Total params: " + str(sum_of_params))


        
    def calc_adjustment(self, expected_output_values, adjustments, activation_function, reversed_layer_index):
        """
        :param expected_output_values: usually a mini-batch of the expected output values
        :param adjustments: if reversed_layer_index is not zero it contains the adjustments of the layer handled before.
        :param activation_function: the activation function for calculating the correct derivative
        :param reversed_layer_index: the layer index but reversed; reversed_layer_index = 0 = last layer of self.layers
        :return: the adjustments

        If the reversed_layer_index equals 0 we calculate the loss for the last layer (= mse, categorical_crossentropy),
        otherwise we calculate the loss for layer x with the adjustments of the layer handled before.
        Then we calculate the derivative and return the product of the loss and the derivative.
        """
        

        if reversed_layer_index == 0:
            predicted_output_values = self.layers[-1].values
            loss = calc_loss(predicted_output_values, expected_output_values, self.loss_function)
            self.loss = loss
            derivative = calc_derivative(predicted_output_values, expected_output_values, activation_function)
        else:
            loss = multiply(adjustments, self.layers[len(self.layers) - reversed_layer_index].weights.T)
            values_of_layer_x = self.layers[len(self.layers)-reversed_layer_index-1].values
            derivative = calc_derivative(values_of_layer_x, [], activation_function)

        adjustments = np.multiply(loss, derivative)
        return adjustments


    
    def backprop(self, input_values, expected_output_values):
        """
        :param input_values: usually a mini batch of all the input values
        :param expected_output_values: usually a mini batch of the expected output values belonging to the input values

        We create a new layer representing the input layer where the derivative-activation function is equal to 
        the activation function of the first layer.
        We set the values of the new layer to the input_values and insert the new layer at index 0 in a copied list 
        of self.layers.
        Then we reverse the list and go through every layer except the first one which is actually the output layer.
        In the loop we call calc_adjustment for getting the loss. Afterwards we multiply the adjustments with
        the values of the current layer to get delta_weight and sum up the adjustments to get delta_bias.
        Finally we calculate the new weights by subtracting the product of the delta_weights of layer x and
        the learning rate; and the new biases by subtracting the product of the delta_biases of layer x and
        the learning rate.
        """
        

        d_weights = []
        d_biases = []
        adjustments = []

        layers_with_input = self.layers[:]
        input_layer = Dense(units=self.layers[0].input_dim, activation="")
        input_layer.derivative_activation = self.layers[0].activation
        input_layer.values = input_values
        layers_with_input.insert(0, input_layer)
        layers_with_input.reverse()

        for iteration_nr, layer in enumerate(layers_with_input[1:]):
            adjustments = self.calc_adjustment(expected_output_values, adjustments,
                                               layer.derivative_activation, iteration_nr)

            delta_weight = multiply(layer.values, adjustments)
            delta_bias = np.sum(adjustments, axis=0)

            d_weights.append(delta_weight)
            d_biases.append(delta_bias)

        d_weights.reverse()
        d_biases.reverse()

        for index in range(len(self.layers)):
            self.layers[index].weights -= (self.learning_rate * d_weights[index])
            self.layers[index].biases -= (self.learning_rate * d_biases[index])


            
    def calc_feedforward(self, input_values):
        """
        :param input_values: usually a mini batch of all the input values

        For the first hidden layer we compute the values of the neurons with the weights, biases and the input_values.
        For the other hidden layers and the last layer we compute the values of the neurons with
        the weights, biases and the values of the neurons of the layer before.
        """

        
        for layer_id in range(len(self.layers)-1):
            if layer_id == 0:
                self.layers[layer_id].compute(input_values)

            self.layers[layer_id+1].compute(self.layers[layer_id].values)


            
    def get_mini_batch_data(self, input_values, expected_output_values):
        """
        :param input_values: the input values to train the model on
        :param expected_output_values: the expected output values of the input values
        :return: two small mini-batches, one containing input_values, the other expected_output_values
        """

        
        input_mini_batch = input_values[:self.mini_batch_size]
        exp_output_mini_batch = expected_output_values[:self.mini_batch_size]
        return input_mini_batch, exp_output_mini_batch

    

    def get_random_mini_batch(self, input_values, expected_output_values):
        """
        :param input_values: the input values to train the model on
        :param expected_output_values: the expected output values of the input values
        :return: two lists of the mini-batch-size, one containing random chosen input values and
        one the to the input values belonging expected output values
        """

        
        input_mini_batch, expected_output_mini_batch = \
            shuffle_two_arrays_same_order(input_values, expected_output_values)
        return input_mini_batch[:self.mini_batch_size], expected_output_mini_batch[:self.mini_batch_size]

    

    def create_mini_batches(self, input_values, expected_output_values):
        """
        :param input_values: all the input values
        :param expected_output_values: all the expected output values
        :return: a list of mini-batches of size self.mini_batch_size
        """
        
        
        batches = []
        for i in range(int(len(input_values)/self.mini_batch_size)):
            input_mini_batch, exp_output_mini_batch = self.get_mini_batch_data(input_values, expected_output_values)
            input_values = input_values[self.mini_batch_size:]
            expected_output_values = expected_output_values[self.mini_batch_size:]
            batches.append((input_mini_batch, exp_output_mini_batch))

        return batches

    

    def fit(self, input_values, expected_output_values, epochs=10, inner_epochs=1):
        """
        :param input_values: the input values to train the model on
        :param expected_output_values: the expected output values of the input values
        :param epochs: the amount of how often the network should be trained with the whole dataset
        :param inner_epochs: the amount of how often the network should be trained with one batch before training it
        with the next batch
        """
        

        self.input_values = input_values
        self.expected_output_values = expected_output_values

        """ train network with mini batches and backpropagation """
        print("---------------------------------------------------------------------------------")
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs}")
            sys.stdout.flush()
            nr_of_batches_done = 0

            input_values, expected_output_values = shuffle_two_arrays_same_order(input_values, expected_output_values)
            batches = self.create_mini_batches(input_values, expected_output_values)

            for repetition_nr in range(inner_epochs):
                for input_d, output_d in batches:
                    self.calc_feedforward(input_d)
                    self.backprop(input_d, output_d)

                    x_random, y_random = self.get_random_mini_batch(self.input_values, self.expected_output_values)
                    acc, loss = self.evaluate(x_random, y_random)

                    """
                    The following code of this function is just for printing the progress, loss and accuracy.
                    """
                    units = 2 * inner_epochs + 1
                    progress_output = ("=" * int(nr_of_batches_done/units)) + ">" + \
                                      ("." * int((len(batches)*inner_epochs)/units - int(nr_of_batches_done/units)))
                    print(f"\r [{progress_output}] - loss: {loss:.4f} - accuracy: {acc:.4f}", end='\r')
                    sys.stdout.flush()
                    nr_of_batches_done += 1

            progress_output = ("=" * int(nr_of_batches_done/units)) + "="
            print(f"\r [{progress_output}] - loss: {loss:.4f} - accuracy: {acc:.4f}", end='\r')
            sys.stdout.flush()
            print()


            
    # function for testing the neural network
    def evaluate(self, input_values, expected_output_values):
        """
        :param input_values: input values to test the accuracy on
        :param expected_output_values: expected output values of the input values
        :return: the accuracy of the neural network on this set and the loss.
        """
        

        amount = len(input_values)
        self.calc_feedforward(input_values)
        predicted_output_values = self.layers[-1].values
        one_dim_predicted_output = [np.argmax(layer) for layer in predicted_output_values]
        true_predicted_output = [x for idx, x in enumerate(one_dim_predicted_output)
                                 if x == np.argmax(expected_output_values[idx])]
        count = len(true_predicted_output)

        loss = calc_loss(predicted_output_values, expected_output_values, self.loss_function)
        acc = count / amount

        return acc, loss

    

    def get_predicted_data(self, input_values):
        """
        :param input_values: input values
        :return: the predicted output values of the input values
        """

        self.calc_feedforward(input_values)
        return self.layers[-1].values
