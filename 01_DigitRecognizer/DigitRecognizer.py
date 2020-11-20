import sys
sys.path.append("/home/tobias/python/NeuralNetworkProjects/")
from NeuralNetworkProjects.NeuralNetwork.models.Sequential import *
sys.path.append("/home/tobias/python/NeuralNetworkProjects/")
from NeuralNetworkProjects.NeuralNetwork.layers.Dense import *

import numpy as np
import csv


amountData = 2048
epochs = 128


# read data
with open('data/train.csv','r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ',')
    data = [data for data in data_iter]

train_data = np.asarray(data, dtype=np.float64)
test_data = train_data[0:2048]


# create training data
train_data = train_data[2048:2048+(int(amountData))]
output = train_data[:,0]

output2dArr = np.zeros((len(output), 10))
for i in range(len(output)):
    output2dArr[i][int(output[i])] = 1

train = train_data[:,1:]


# create testing data
test_output = test_data[:,0]
test_output2dArr = np.zeros((len(test_output), 10))
for i in range(len(test_output)):
        test_output2dArr[i][int(test_output[i])] = 1

test = test_data[:,1:]


# creating model
model = Sequential()
model.add(Dense(256, activation="sigmoid", amount_of_input_neurons=784))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(32, activation="sigmoid"))
model.add(Dense(16, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))

model.summary()
model.create()
model.train(train, output2dArr, epochs=epochs)

error1, error2, acc = model.test(test, test_output2dArr)
print(error1, error2, acc)