from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
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

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255.0
x_test = x_test/255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# creating model
model = Sequential()

model.add(Conv2D(32, (5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (5, 5), padding='Same', activation="relu"))

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='Same', activation="relu"))
model.add(Conv2D(64, (3, 3), padding='Same', activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs)

scores = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))