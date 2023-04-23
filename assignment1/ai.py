from typing import *
import os
import tensorflow as tf
import keras
from keras import Model
from keras.layers import Layer, Conv2D, MaxPool2D, Dropout, BatchNormalization, Dropout, Add, Flatten, Dense

class ResidualConvolutional2D(Layer):
    def __init__(self, filters : int, kernel_size : Tuple[int, int], dropout : float, activation : str):
        super().__init__()
        self._conv2d  = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation=activation)
        self._dropout = Dropout(dropout)
        self._bn      = BatchNormalization()

    def call(self, inputs):
        x = self._conv2d(inputs)
        x = self._dropout(x)
        x = self._bn(x)

        y = x + inputs

        return y

class MyModel(Model):
    def __init__(self, model_filters : int, residual_filters : int, kernel_size : Tuple[int, int], dropout : float, ff_dim : int, NUM_CLASSES : int = 10):
        super().__init__()

        self._layers = [
            Conv2D(filters=model_filters, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'),
            BatchNormalization(),
            Dropout(dropout),
            Conv2D(filters=model_filters, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu'),  
            BatchNormalization(),
            Dropout(dropout),          
            Conv2D(filters=residual_filters, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation='relu', name='ResidualPrepare'),  
            ResidualConvolutional2D(filters=residual_filters, kernel_size=kernel_size, dropout=dropout, activation='relu'),
            ResidualConvolutional2D(filters=residual_filters, kernel_size=kernel_size, dropout=dropout, activation='relu'),
            ResidualConvolutional2D(filters=residual_filters, kernel_size=kernel_size, dropout=dropout, activation='relu'),
            ResidualConvolutional2D(filters=residual_filters, kernel_size=kernel_size, dropout=dropout, activation='relu'),
            Conv2D(filters=model_filters, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'),
            Conv2D(filters=model_filters, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'),
            Flatten(),
            Dense(ff_dim, activation='relu'),
            Dropout(dropout),
            Dense(NUM_CLASSES, activation='softmax')
        ]


    def call(self, inputs):
        x = inputs

        for layer in self._layers:
            x = layer(x)
        
        return x