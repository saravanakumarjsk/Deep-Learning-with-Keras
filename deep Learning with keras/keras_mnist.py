import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
print(x_train.shape[0], 'training samples')
print(x_test.shape[0], 'testing samples')

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(input_dim = 784, output_dim = 500))
model.add(Activation('relu'))

model.add(Dense(output_dim = 500))
model.add(Activation('relu'))

model.add(Dense(output_dim = 500))
model.add(Activation('relu'))

model.add(Dense(output_dim = 10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy',
            optimizer = 'Adam',
            metrics = ['accuracy'])

history = model.fit(x_train, Y_train,
                    batch_size = 128,
                    nb_epoch = 2,
                    verbose=1,
                    validation_split = 0.2)

score = model.evaluate(x_test, Y_test, verbose = 1)

print('test score', score[0])
print('test accuracy', score[1])






