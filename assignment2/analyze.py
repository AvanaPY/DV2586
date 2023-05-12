import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from data import get_dataset2_df
from data.data import DataConfig, create_sequence
from model import build_encoder_decoder_lstm, Analyzer

model = build_encoder_decoder_lstm()
model.summary()

MODEL_WEIGHTS_PATH = './checkpoints/run27/ckpt'
try:
    print(f'Loading weights from "{MODEL_WEIGHTS_PATH}"')
    model.load_weights(MODEL_WEIGHTS_PATH)
except:
    raise Exception(f'Failed to load weights')

df, val, data_cols, scalers = get_dataset2_df()

analyzer = Analyzer(model)
analyzer.calculate_reconstruction_errors(create_sequence(df, data_cols, DataConfig.SEQUENCE_LENGTH))
print(f'Reconstruction errors: {analyzer.reconstruction_errors}')

df_analyze = pd.concat([df, val]).reset_index()
sequences = create_sequence(df_analyze, data_cols, DataConfig.SEQUENCE_LENGTH)
dates = mdates.datestr2num(df_analyze['timestamp'])
analyzer.plot_anomalies(sequences, dates, data_cols, scalers=scalers)