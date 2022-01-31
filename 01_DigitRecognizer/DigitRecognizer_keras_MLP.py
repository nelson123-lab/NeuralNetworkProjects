from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)

randomize = np.arange(len(x_train))
np.random.shuffle(randomize)
x_train = x_train[randomize]
y_train = y_train[randomize]

x_train = x_train[:4096]
y_train = y_train[:4096]

x_test = x_test.reshape(10000, 784)

randomize = np.arange(len(x_test))
np.random.shuffle(randomize)
x_test = x_test[randomize]
y_test = y_test[randomize]

x_test = x_test[:1300]
y_test = y_test[:1300]

x_train = x_train/255.0
x_test = x_test/255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# creating model
model = Sequential()
model.add(Dense(256, activation="relu", input_dim=x_train.shape[1]))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy']) # mse
model.fit(x_train, y_train, epochs=epochs)

scores = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))