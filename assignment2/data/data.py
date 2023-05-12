from typing import *
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

class DataConfig:
    DATASET_PATH = './data/dataset2.csv'
    SEQUENCE_LENGTH = 32
    BATCH_SIZE = 4
    SHUFFLE_BUFFER_SIZE = 128
    CUTOFF_DATE = '2004-02-16 00:00:00'
        
def standardize_column(df : pd.DataFrame, column : str) -> None:
    scaler = MinMaxScaler()
    df[[column]] = scaler.fit_transform(df[[column]])
    return scaler
    
def get_dataset2_df() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(DataConfig.DATASET_PATH)
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    data_cols = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
    scalers = [standardize_column(df, col) for col in data_cols]

    train_data = df[df['timestamp'] < DataConfig.CUTOFF_DATE]
    val_data = df[df['timestamp'] >= DataConfig.CUTOFF_DATE].reset_index()
    return (train_data, val_data, data_cols, scalers)
    
def create_sequence(df : pd.DataFrame, data_columns : List[str], sequence_length : int) -> Any:
    df_length = len(df.index)
    xs = []
    for i in range(df_length-sequence_length+1):
        data_point = [df[data_col].iloc[i:i+sequence_length].to_list() for data_col in data_columns]
        data_point = [d for d in zip(*data_point)]
        xs.append(data_point)
    return np.array(xs)

def get_data(return_scalers : bool = False) -> Any:
    df, val_data, data_columns, scalers = get_dataset2_df()

    df_length = len(df.index)
    sl = DataConfig.SEQUENCE_LENGTH
    
    # Get data from dataframe, shape (3, SEQUENCE_LENGTH)
    data = create_sequence(df, data_columns, DataConfig.SEQUENCE_LENGTH)
    test = create_sequence(val_data, data_columns, DataConfig.SEQUENCE_LENGTH)
    
    ret = [(data, data), (test, test)]
    
    if return_scalers:
        ret.append(scalers)
    return tuple(ret)