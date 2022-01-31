import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    """
    We set values smaller equal than 0 to 0.
    """

    values = np.full_like(x, x)
    values[x <= 0] = 0.0
    return values


def relu_derivative(x):
    """
    The derivative of the RELU function is 1 if x is greater than 0 and 0 otherwise.
    """

    derivative = np.ones_like(x)
    derivative[x <= 0] = 0.0
    return derivative


def leaky_relu(x):
    """
    The difference to ReLU is that we do not set the numbers smaller than 0 to 0
    but to x*alpha where alpha is a small number like 0.03
    """

    values = np.full_like(x, x)
    values[x <= 0] = values[x <= 0] * 0.03
    return values


def leaky_relu_derivative(x):
    """
    The derivative of the leaky RELU function is 1 if x is greater than 0 and alpha otherwise
    where alpha is the same alpha as the one used in the relu function.
    """

    derivative = np.ones_like(x)
    derivative[x < 0] = 0.03
    return derivative


def softmax(x):
    """
    :param x: the values calculated from the layers before
    :return: the probabilities of x
    """

    temp = x - x.max(axis=1).reshape(-1, 1)
    exponential_temp = np.exp(temp)
    return exponential_temp / exponential_temp.sum(axis=1).reshape(-1, 1)