from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam, rmsprop

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense((500), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense((10), activation = 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy',
                optimizer = 'Adam',
                metrics = ['accuracy'])

model.fit(x_train, Y_train, batch_size = 128, epochs = 5, validation_split = 0.2, verbose = 1)

score = model.evaluate(x_test, Y_test, verbose = 0)

print("Test score:", score[0])
print('Test accuracy:', score[1])
