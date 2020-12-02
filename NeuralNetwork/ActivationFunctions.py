from scipy.special import expit
import numpy as np


def sigmoid(x):
    return expit(x)



def sigmoid_derivative(x):
    return x * (1.0 - x)



def relu(x):
    """
    We actually use here the leaky RELU activation function.
    It is known as a more stable and better version of RELU.
    The difference is that we do not set the numbers smaller than 0 to 0
    but to x*alpha where alpha is a small number like 0.03
    """
    

    values = np.full_like(x, x)
    values[x <= 0] = values[x <= 0] * 0.03 # 0,01
    return values



def relu_derivative(x):
    """
    The derivative of the leaky RELU function is 1 if x is greater than 0 and alpha otherwise
    where alpha is the same alpha as the one used in the relu function.
    """

    
    derivative = np.ones_like(x)
    derivative[x < 0] = 0.03 # 0.01
    return derivative



def softmax(x):
    """
    :param x: the values calculated from the layers before
    :return: the probabilities of x
    """

    
    temp = x - x.max(axis=1).reshape(-1, 1)
    exponential_temp = np.exp(temp)
    return exponential_temp / exponential_temp.sum(axis=1).reshape(-1, 1)



def softmax_derivative(predicted, expected):
    """
    :param predicted: the predicted values
    :param expected: the expected values
    :return: the difference of the predicted and the expected values divided by the amount of expected/predicted values
    """
    

    return (predicted - expected) / len(expected)



def calc_derivative(predicted_output, expected_output, activation_function):
    """
    :param predicted_output: the current values of a layer x
    :param expected_output: the expected values of the last layer
    :param activation_function: the activation function of layer x
    :return: the derivative of the current values of a layer x
    """

    
    if activation_function == "sigmoid":
        return sigmoid_derivative(predicted_output)

    if activation_function == "relu":
        return relu_derivative(predicted_output)

    if activation_function == "softmax":
        return softmax_derivative(predicted_output, expected_output)

    raise ValueError("ERROR: the given activation function is not avialable.")
