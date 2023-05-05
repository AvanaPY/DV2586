from __future__ import annotations
from typing import *
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, CuDNNLSTM, Dense, Input, TimeDistributed, RepeatVector, Dropout

from data.data import Config as DataConfig

class ModelConfig:
    LEARNING_RATE = 1e-3
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss      = tf.keras.losses.MeanAbsoluteError()
    metrics   = []
    
    MODEL_INPUT_VALUES = 4
    MODEL_OUTPUT_VALUES = 4
    MODEL_SEQUENCE_LENGTH = DataConfig.SEQUENCE_LENGTH
        
class ModelWrapper:
    def __init__(self, model : Model):
        self._model = model
        self._reconstruction_error = 0
        
    @property
    def model(self) -> Model:
        return self._model
    
    def calculate_reconstruction_error(self, train_ds : tf.data.Dataset) -> None:
        h = self._model.evaluate(train_ds)
        
        mse = h
        self._reconstruction_error = mse
        
        return self._reconstruction_error
    
    def plot_anomalies(dataset : tf.data.Dataset) -> None:
        pass
    
    def __call__(self, *args, **kwargs) -> Any:
        return self._model(*args, **kwargs)
    
    def fit(self, *args, **kwargs) -> Any:
        return self._model.fit(*args, **kwargs)        
    
    def summary(self) -> None:
        return self._model.summary()

def build_encoder_decoder_lstm() -> ModelWrapper:
    model = Sequential()
    # Encoder
    model.add(LSTM(128, activation='tanh', name='EncoderStart'))
    model.add(Dropout(0.2))
    model.add(RepeatVector(ModelConfig.MODEL_SEQUENCE_LENGTH))
    # Decoder
    model.add(LSTM(128, activation='tanh', return_sequences=True, name='DecoderStart'))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(ModelConfig.MODEL_OUTPUT_VALUES)))
    
    model.compile(
        optimizer=ModelConfig.optimizer,
        loss=ModelConfig.loss,
        metrics=ModelConfig.metrics
    )
    model.build((None, ModelConfig.MODEL_SEQUENCE_LENGTH, ModelConfig.MODEL_INPUT_VALUES))
    
    
    return ModelWrapper(model)