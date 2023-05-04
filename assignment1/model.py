from typing import *
import os
import tensorflow as tf
import keras
from keras import Model
from keras.layers import Layer, Conv2D, MaxPool2D, Dropout, BatchNormalization, Dropout, Add, Flatten, Dense, Input, MaxPooling2D, AveragePooling2D

from data import get_image_dimensions

class ResidualConvolutional2D(Layer):
    def __init__(self, 
                 filters : int,
                 kernel_size : Tuple[int, int], 
                 dropout : float, 
                 activation : str):
        super().__init__()
        self._conv2d  = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation=activation)
        self._dropout = Dropout(dropout)
        self._bn      = BatchNormalization()

    def call(self, inputs):    # 
        x = self._conv2d(inputs)
        x = self._dropout(x)
        x = self._bn(x)

        y = x + inputs

        return y

class MyModel(Model):
    def __init__(self):
        super().__init__(name='StureDigitNet')

        self._layers = []
        
        # Initial layers
        self._layers.append(Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'))
        self._layers.append(Dropout(0.2))
        self._layers.append(Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'))
        self._layers.append(Dropout(0.2))
        
        # Residual layers
        res_filters = 32
        self._layers.append(Conv2D(res_filters, kernel_size=1, activation='relu', name='ResidualPrepare'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))
        self._layers.append(ResidualConvolutional2D(filters=res_filters, kernel_size=3, dropout=0.2, activation='relu'))

        # # Size reduction layers
        self._layers.append(Conv2D(filters=16, kernel_size=(9, 9), strides=(1, 1), padding='valid', activation='relu'))
        self._layers.append(Dropout(0.2))
        self._layers.append(Conv2D(filters=12, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'))
        self._layers.append(Dropout(0.2))
        
        # Reduce channels to ideal
        self._layers.append(Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'))
        
        # Flatten and map down to number of classes
        self._layers.append(Flatten())
        self._layers.append(Dense(10, activation='softmax'))

    def call(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x
    
    # Work-around to get .summary() to properly work and display output shapes with subclassed Models
    def summary(self):
        image_dims = get_image_dimensions()
        x = Input(shape=(image_dims[0], image_dims[1], 1))
        Model(inputs=[x], outputs=self.call(x)).summary()