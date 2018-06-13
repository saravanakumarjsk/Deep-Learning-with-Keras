from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD,
from keras.datasets import mnist
from PIL import Image
import numpy as np
import argparse
import math

def generator_model():
    model = Sequential()

    model.add(Dense(input_dim = 100, output_dim = 1024))
    model.add(Dense((128*7*7), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Reshape((7, 7, 128), input_dim = (128*7*7)))

    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(64, (5,5), padding = 'same', activation = 'relu'))

    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(1, (5,5), padding = 'same', activation = 'relu'))
    return model

def discriminator_model():
    model = Sequential()

    model.add(Dense(64, (5, 5), padding = 'same', input_dim = (28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dense(128, (5, 5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model





















































