from typing import *
import os
import tensorflow as tf
import keras
from keras import Model
from keras.layers import Layer, Conv2D, MaxPool2D, Dropout, BatchNormalization, Dropout, Add, Flatten, Dense, Input, MaxPooling2D, AveragePooling2D

from data import get_image_resize_dimensions

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

    def call(self, inputs):
        x = self._conv2d(inputs)
        x = self._dropout(x)
        x = self._bn(x)

        y = x + inputs

        return y

class MyModel(Model):
    def __init__(self, 
                 model_filters : int, 
                 residual_layers : int, 
                 residual_filters : int, 
                 residual_kernel_size : Tuple[int, int], 
                 dropout : float, 
                 ff_dim : int, 
                 NUM_CLASSES : int = 10):
        
        super().__init__(name='StureDigitNet')

        self._layers = []
        self._layers.append(Conv2D(filters=residual_filters, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu', name='ResidualPrepare'))
        self._layers.append(Dropout(dropout))
        
        for _ in range(residual_layers):
            self._layers.append(ResidualConvolutional2D(filters=residual_filters, kernel_size=residual_kernel_size, dropout=dropout, activation='relu'))

        self._layers.append(Conv2D(filters=model_filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self._layers.append(MaxPooling2D(pool_size=2))
        
        self._layers.append(Conv2D(filters=model_filters, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self._layers.append(MaxPooling2D(pool_size=2))
        
        self._layers.append(Flatten())
        
        self._layers.append(Dense(ff_dim, activation='relu'))
        self._layers.append(Dense(NUM_CLASSES, activation='softmax'))
        

    def call(self, inputs):
        x = inputs

        for layer in self._layers:
            x = layer(x)
        
        return x
    
    # Work-around to get .summary() to properly work and display output shapes with subclassed Models
    def summary(self):
        image_dims = get_image_resize_dimensions()
        x = Input(shape=(image_dims[0], image_dims[1], 1))
        Model(inputs=[x], outputs=self.call(x)).summary()