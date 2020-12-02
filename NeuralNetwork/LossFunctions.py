from NeuralNetworkProjects.NeuralNetwork.Utils import *


def categorical_crossentropy_loss(predicted_output, expected_output):
    """
    :param predicted_output: the current predicted output of the model
    :param expected_output: the expected output
    :return: the categorical-crossentropy-loss

    This function calculates and return the categorical-crossentropy-loss.
    "+1e-15" is just for adding a very small number to avoid np.log(0)
    """
    

    loss = -np.sum(expected_output * (np.log(predicted_output+1e-15)))
    return loss / (len(expected_output))



def mse(predicted_output, expected_output):
    """
    :param predicted_output: the current predicted output of the model
    :param expected_output: the expected output
    :return: the mean-squared-error loss
    """
    
    
    return 1/len(expected_output) * np.sum((expected_output - predicted_output) ** 2)



def calc_loss(predicted_output, expected_output, loss_function):
    """
    :param predicted_output: the current predicted output of the model
    :param expected_output: the expected output
    :param loss_function: the loss function
    :return: the loss value
    """
    

    if loss_function == "categorical_crossentropy":
        return categorical_crossentropy_loss(predicted_output, expected_output)

    if loss_function == "mse":
        return mse(predicted_output, expected_output)

    raise ValueError("ERROR: the given loss function is not avialable.")
