from NeuralNetworkProjects.NeuralNetwork.layers.Dense import *
from NeuralNetworkProjects.NeuralNetwork.models.Model import *
from NeuralNetworkProjects.NeuralNetwork.ActivationFunctions import *
from NeuralNetworkProjects.NeuralNetwork.LossFunctions import *
from NeuralNetworkProjects.NeuralNetwork.Utils import *
from NeuralNetworkProjects.NeuralNetwork.LossFunctions import *

import sys
import numpy as np
from prettytable import PrettyTable


class Sequential(Model):
    def __init__(self):
        self.layers = []
        self.x = [] # = input values
        self.y = [] # = expected output values
        self.mini_batch_size = 32
        self.learning_rate = 0.08
        self.loss_function = None
        self.loss = 0

    def compile(self, loss, mini_batch_size=32, learning_rate=0.08):
        """
        :param loss: the loss function you want to use
        :param mini_batch_size: the mini-batch-size you want to use (optionally, standard=32)
        :param learning_rate: the learning_rate you want to use (optionally, standard=0.08)
        """

        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate

        if loss == "categorical_crossentropy":
            self.loss_function = categorical_crossentropy_loss
        elif loss == "mse":
            self.loss_function = mse

        self.layers[-1].is_output_layer = True

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
            if self.layers[-1].units == -1:
                x = [x.units for x in self.layers if x.units != -1][-1]
                self.layers[-1].units = x

            layer.initialize_weights(self.layers[-1].units)
        layer.initialize_biases()

        self.layers.append(layer)

    def summary(self):
        """
        This function prints a summary of the model.
        """

        table = PrettyTable()
        table.field_names = ["index", "layer type", "layer units", "weights shape", "biases shape", "param #"]
        sum_of_params = 0
        for index, layer in enumerate(self.layers):

            if len(layer.W.shape) >= 2:
                nr_of_params_weights = layer.W.shape[0] * layer.W.shape[1]
            else:
                nr_of_params_weights = layer.W.shape[0]

            if len(layer.b.shape) >= 2:
                nr_of_params_bias = layer.b.shape[0] * layer.b.shape[1]
            else:
                nr_of_params_bias = layer.b.shape[0]

            nr_of_params = nr_of_params_weights + nr_of_params_bias
            sum_of_params += nr_of_params
            table.add_row([index, type(layer).__name__, layer.units, layer.W.shape,
                           layer.b.shape, nr_of_params])
        print(table)
        print("Total params: " + str(sum_of_params))

    def calc_feedforward(self, input_values):
        """
        :param input_values: usually a mini batch of all the input values

        For the first hidden layer we compute the values of the neurons with the weights, biases and the x.
        For the other hidden layers and the last layer we compute the values of the neurons with
        the weights, biases and the values of the neurons of the layer before.
        """

        for layer_id in range(len(self.layers)-1):
            if layer_id == 0:
                self.layers[layer_id].forward(input_values)

            self.layers[layer_id+1].forward(self.layers[layer_id].a)

    def get_mini_batch_data(self, input_values, expected_output_values):
        """
        :param input_values: the input values to train the model on
        :param expected_output_values: the expected output values of the input values
        :return: two small mini-batches, one containing x, the other y
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

        self.x = input_values
        self.y = expected_output_values

        """ train network with mini batches and backpropagation """
        print("---------------------------------------------------------------------------------")
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs}")
            sys.stdout.flush()
            nr_of_batches_done = 0
            units = 2 * inner_epochs + 1

            input_values, expected_output_values = shuffle_two_arrays_same_order(input_values, expected_output_values)
            batches = self.create_mini_batches(input_values, expected_output_values)

            epoch_losses, epoch_accs = [], []

            for repetition_nr in range(inner_epochs):
                for input_d, y in batches:

                    # compute the feed forward
                    a = input_d
                    for layer in self.layers:
                        a = layer.forward(a)

                    # compute the backward propagation
                    delta = self.layers[-1].backward(y, None)
                    for l in range(len(self.layers) - 2, -1, -1):
                        delta = self.layers[l].backward(delta, self.layers[l+1])

                    # update weights
                    a = input_d
                    for layer in self.layers:
                        layer.update(self.learning_rate, a)
                        a = layer.a

                    x_random, y_random = self.get_random_mini_batch(self.x, self.y)
                    acc, loss = self.evaluate(x_random, y_random)

                    epoch_accs.append(acc)
                    epoch_losses.append(loss)

                    """
                    The following code of this function is just for printing the progress, loss and accuracy.
                    """
                    # units = 2 * inner_epochs + 1
                    progress_output = ("=" * int(nr_of_batches_done/(units*2))) + ">" + \
                                      ("." * int((len(batches)*inner_epochs)/(units*2)
                                                 - int(nr_of_batches_done/(units*2))))
                    print(f"\r [{progress_output}] - loss: {loss:.4f} - accuracy: {acc:.4f}", end='\r')
                    sys.stdout.flush()
                    nr_of_batches_done += 1

            epoch_accs = np.array(epoch_accs)
            epoch_losses = np.array(epoch_losses)

            progress_output = ("=" * int(nr_of_batches_done/(units*2))) + "="
            print(f"\r [{progress_output}] - loss: {epoch_losses.mean():.4f} - accuracy: "
                  f"{epoch_accs.mean():.4f}", end='\r')
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
        predicted_output_values = self.layers[-1].a
        one_dim_predicted_output = [np.argmax(layer) for layer in predicted_output_values]
        true_predicted_output = [x for idx, x in enumerate(one_dim_predicted_output)
                                 if x == np.argmax(expected_output_values[idx])]
        count = len(true_predicted_output)
        loss = self.loss_function(predicted_output_values, expected_output_values)
        acc = count / amount

        return acc, loss

    def get_predicted_data(self, input_values):
        """
        :param input_values: input values
        :return: the predicted output values of the input values
        """

        self.calc_feedforward(input_values)
        return self.layers[-1].a
