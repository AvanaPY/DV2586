from __future__ import annotations
from typing import *
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed, RepeatVector, BatchNormalization

from data.data import DataConfig as DataConfig

class ModelConfig:
    LEARNING_RATE = 1e-4
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss      = tf.keras.losses.MeanAbsoluteError()
    metrics   = []
    
    MODEL_INPUT_VALUES = 4
    MODEL_OUTPUT_VALUES = 4
    MODEL_SEQUENCE_LENGTH = DataConfig.SEQUENCE_LENGTH
    
def build_encoder_decoder_lstm() -> Model:
    model = Sequential(name='SomeCoolAutoEncoderMaybe')
    # Encoder
    model.add(LSTM(32, activation='tanh', return_sequences=True, name='encoderstart'))
    model.add(BatchNormalization())
    model.add(LSTM(16, activation='tanh'))
    model.add(BatchNormalization())
    
    # Decoder
    model.add(RepeatVector(ModelConfig.MODEL_SEQUENCE_LENGTH))
    model.add(LSTM(32, activation='tanh', return_sequences=True, name='decoderstart'))
    model.add(BatchNormalization())
    model.add(LSTM(128, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(ModelConfig.MODEL_OUTPUT_VALUES, activation='sigmoid')))
    
    model.compile(
        optimizer=ModelConfig.optimizer,
        loss=ModelConfig.loss,
        metrics=ModelConfig.metrics
    )
    model.build((None, ModelConfig.MODEL_SEQUENCE_LENGTH, ModelConfig.MODEL_INPUT_VALUES))

    return model