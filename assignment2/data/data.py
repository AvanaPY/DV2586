from typing import *
import pandas as pd
import tensorflow as tf

class Config:
    DATASET_PATH = './data/dataset2.csv'
    VALIDATION_SPLIT = 0.2
    SEQUENCE_LENGTH = 16
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 128
        
def standardize_column(df : pd.DataFrame, column : str) -> None:
    df[column] = df[column] / df[column].max()
    
def get_dataset2_df() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(Config.DATASET_PATH)
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    # Turn status into categorical
    # 1 for anomaly, 0 for normal -> our goal is to predict 1s
    # df['temperature_status'] = 1 - df['temperature_status'].astype('category').cat.codes
    # df['pressure_status'] = 1 - df['pressure_status'].astype('category').cat.codes
    # df['humidity_status'] = 1 - df['humidity_status'].astype('category').cat.codes
    # df['status'] = df['temperature_status'] | df['pressure_status'] | df['humidity_status']

    # df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
    
    # Scale the columns to the [0, 1] range
    # df['temperature'] = df['temperature'] / df['temperature'].max()
    # df['pressure'] = df['pressure'] / df['pressure'].max()
    # df['humidity'] = df['humidity'] / df['humidity'].max()

    standardize_column(df, 'Bearing 1')
    standardize_column(df, 'Bearing 2')
    standardize_column(df, 'Bearing 3')
    standardize_column(df, 'Bearing 4')

    return (df, ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'])
    
def get_data() -> tf.data.Dataset:
    df, data_columns = get_dataset2_df()
    
    print(df.head(10))
    df_length = len(df.index)
    sl = Config.SEQUENCE_LENGTH
    
    # Get data from dataframe, shape (3, SEQUENCE_LENGTH)
    xs = []
    for i in range(df_length-sl):
        data_point = [df[data_col].iloc[i:i+sl].to_list() for data_col in data_columns]
        data_point = [d for d in zip(*data_point)]
        xs.append(data_point)
    
    # Make the network reconstruct the data
    ys = xs
    
    # Turn into TF dataset and batch
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    
    count = tf.data.experimental.cardinality(ds)
    val_count = tf.floor(tf.cast(count, tf.float32) * Config.VALIDATION_SPLIT).numpy()
    val_ds = ds.take(val_count)
    ds = ds.skip(val_count)
    
    ds = (
        ds
        .shuffle(buffer_size=Config.SHUFFLE_BUFFER_SIZE)
        .batch(batch_size=Config.BATCH_SIZE)
    )
    
    val_ds = (
        val_ds
        .batch(1000)
    )
    
    return ds, val_ds