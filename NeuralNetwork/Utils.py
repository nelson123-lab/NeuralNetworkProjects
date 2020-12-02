import numpy as np


def to_categorical(input_arr, number_of_categories: int):
    """
    :param input_arr: array of numbers from 0 to number_of_categories
    :param number_of_categories: number of the possible categories
    :return: the new array only containing arrays of zeros and ones

    like the to_categorical function provided by Keras.

    Instead of an output x we want a list of size number_of_categories
    where the value at index x is 1 and all others are zero.
    example: input_arr=[5, 1], number_of_categories=10
    --> [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    """

    arr = np.zeros((len(input_arr), number_of_categories))
    for i in range(len(input_arr)):
        arr[i][int(input_arr[i])] = 1

    return arr


def multiply(one, two):
    """
    :param one: a numpy array
    :param two: a numpy array
    :return: the product of the two numpy arrays

    This function just tries to multiply the two arrays together and returns the product.
    """

    final_one = one
    final_two = two

    if len(one.shape) > 1:
        if one.shape[1] == two.shape[0]:
            final_one = one
            final_two = two
        elif len(two.shape) > 1:
            if two.shape[1] == one.shape[0]:
                final_two = two.T
                final_one = one.T
            elif one.shape[0] == two.shape[0]:
                final_one = one.T
                final_two = two
            elif one.shape[1] == two.shape[1]:
                final_one = one
                final_two = two.T
            else:
                raise ValueError(f"Error: incompatible types {one.shape}, {two.shape}")

    product = np.dot(final_one, final_two)
    return product


def shuffle_two_arrays_same_order(x, y):
    """
    :param x: a numpy array
    :param y: a numpy array
    :return: the same numpy arrays in a different order

    This function shuffles the two arrays given in the same order and returns the two shuffled arrays.
    """

    if len(x) == len(y):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        x_final = x[randomize]
        y_final = y[randomize]
        return x_final, y_final
    else:
        raise ValueError("Error: The two lists have to have the same length!")
