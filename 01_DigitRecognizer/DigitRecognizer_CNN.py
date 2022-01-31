import sys
sys.path.append("/home/tobias/python/NeuralNetworkProjects/")
from NeuralNetworkProjects.NeuralNetwork.models.Sequential import *
sys.path.append("/home/tobias/python/NeuralNetworkProjects/")
from NeuralNetworkProjects.NeuralNetwork.layers.Dense import *
sys.path.append("/home/tobias/python/NeuralNetworkProjects/")
from NeuralNetworkProjects.NeuralNetwork.layers.Conv2D import *
sys.path.append("/home/tobias/python/NeuralNetworkProjects/")
from NeuralNetworkProjects.NeuralNetwork.layers.Flatten import *
sys.path.append("/home/tobias/python/NeuralNetworkProjects/")
from NeuralNetworkProjects.NeuralNetwork.Utils import *


"""
Just using Keras for getting the dataset. 
"""
from keras.datasets import mnist

"""
epochs: how often you want to train your model with the whole dataset
number_of_categories: the amount of possible outputs (numbers from 0-9 --> 10) = number of output neurons
"""
epochs = 20
number_of_categories = 10
training_data_size = 4096
testing_data_size = 1300

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train, y_train = shuffle_two_arrays_same_order(x_train, y_train)
x_train = x_train[:training_data_size]
y_train_not_categorical = y_train[:training_data_size]

x_test = x_test.reshape(10000, 784)
x_test, y_test = shuffle_two_arrays_same_order(x_test, y_test)
x_test = x_test[:testing_data_size]
y_test_not_categorical = y_test[:testing_data_size]


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

"""
Instead of an output like "5" we want a list of size 10 where the value at index 5 is 1 and all others are zero.
--> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
That is exactly what the to_categorical function does.
"""
y_train = to_categorical(y_train_not_categorical, number_of_categories)
y_test = to_categorical(y_test_not_categorical, number_of_categories)

"""
Now there are numbers between 0 and 255 in the arrays x_train and x_test. 
We want to normalize the numbers between 0 and 1. 
(That is very important when we are using RELU as activation function, 
so that the numbers of the weights and values do not explode.)
"""
x_train = x_train/255.0
x_test = x_test/255.0

"""
In the next steps we create our model and add layers with different activation functions to the model.
input_dim = the amount of input neurons (in this example 784 (= 28x28))
In the last layer we use the activation function Softmax which turns the calculated values into probabilities. 
"""
model = Sequential()
model.add(Conv2D(1, 4, (5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(4, 4, (5, 5), padding='Same', activation="relu"))
model.add(Flatten())
model.add(Dense(3136, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

"""
As in Keras, the summary function gives us an oversight of the structure of our model. 
Next we compile the model and set the loss function. 
Optionally we also could set the mini-batch-size or the learning-rate.
Finally we train our model with the function fit. 
"""
model.summary()
model.compile(loss="categorical_crossentropy")
model.fit(x_train, y_train, epochs=epochs)

"""
Next we test the model with the training data and the testing data. 
If the training accuracy is way higher than the testing accuracy, the model might be overfitted. 
"""
acc_train, loss_train = model.evaluate(x_train, y_train)
print("Training Accuracy: %.2f%%\n" % (acc_train*100))

acc_test, loss_test = model.evaluate(x_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (acc_test*100))

predicted_output = model.get_predicted_data(x_test)

visualize_img(predicted_output, x_test, y_test_not_categorical)
