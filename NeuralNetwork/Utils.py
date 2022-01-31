import numpy as np
from matplotlib import pyplot as plt


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
    Works with up to 3 dimensional-arrays if only one is 3 dimensional and one is 2 dimensional
    """

    final_one = one
    final_two = two

    if len(one.shape) > 2 or len(two.shape) > 2:
        three_dim, two_dim = (one, two) if len(one.shape) > 2 else (two, one)

        if three_dim.shape[2] == two_dim.shape[0]:
            final_one, final_two = three_dim, two_dim
        elif three_dim.shape[2] == two_dim.shape[1]:
            final_one, final_two = three_dim, two_dim.T

    elif len(one.shape) > 1:
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


def visualize_img(predicted_output, x_test, y_test):
    """
    With the function get_predicted_data(x) we get the predicted values of the input values x.
    """

    """
    Now we just display the input images and the predicted result for visualization. 
    """
    testing_data_size = x_test.shape[0]

    tested_input = x_test.reshape(testing_data_size, 28, 28)
    tested_input = tested_input[:(testing_data_size - (testing_data_size % 9))]
    tested_input = tested_input.reshape(int(len(tested_input) / 9), 9, 28, 28)

    for idx, images in enumerate(tested_input):
        for i in range(9):
            index = idx * 9 + i
            predicted = np.argmax(predicted_output[index])
            expected = y_test[index]

            ax = plt.subplot(330 + 1 * i + 1)
            ax.axis('off')

            if expected == predicted:
                ax.set_title(predicted, color="green")
            else:
                ax.set_title(f"exp: {expected}, pred: {predicted}", color="red")

            plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        plt.show()
