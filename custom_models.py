import numpy as np

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32"
os.environ["KERAS_BACKEND"] = "theano"
import theano
import keras
from keras.models import Sequential

from keras.utils import np_utils

from keras.layers import Dense, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

# define the model
def basic_cnn(input_shape, optimizer="adam", loss="mse"):

    model = Sequential()

    #INPUT LAYER
    model.add(Convolution3D(32, (5,5,5), activation='relu', input_shape=input_shape, data_format = 'channels_first'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Convolution3D(64, (2,2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Convolution3D(64, (2,2,2), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.2))
    # OUTPUT LAYER
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def basic_nn(input_shape, optimizer="adam", loss="mse"):

    model = Sequential()

    #INPUT LAYER
    model.add(Dense(10000, input_shape=input_shape, activation="relu"))
    model.add(Dense(5000, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(50, activation="relu"))
    # OUTPUT LAYER
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)
    return model
