import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['__IMAGE_DESTINATION'] = 'img'
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import argparse
from data import get_dataset2_df
from data.data import DataConfig, create_sequence
from model import build_encoder_decoder_lstm, Analyzer

IMAGE_DESTINATION = os.environ['__IMAGE_DESTINATION']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='latest', dest='checkpoint_name', help='The name of the checkpoint. Input ´latest´ for the last created model.')
    parser.add_argument('-i', action='store_true', dest='images', help='Turn on to save graphs to images.')
    args = parser.parse_args()
    
    # Parameters
    if args.checkpoint_name == 'latest':
        files = os.listdir('./checkpoints')
        files = sorted(files, key=lambda f : os.stat(os.path.join('./checkpoints', f)).st_ctime)
        MODEL_CHECKPOINT_PATH = os.path.join('./checkpoints', files[-1])
    else:
        MODEL_CHECKPOINT_PATH = os.path.join('./checkpoints', args.checkpoint_name)
    MODEL_CHECKPOINT_PREFIX = os.path.join(MODEL_CHECKPOINT_PATH, 'ckpt')
    
    # Build the model
    model = build_encoder_decoder_lstm()
    model.summary()

    if args.images:
        os.makedirs(IMAGE_DESTINATION, exist_ok=True)
        tf.keras.utils.plot_model(model, 
                                to_file=os.path.join(IMAGE_DESTINATION, 'model.png'),
                                show_shapes=True,
                                expand_nested=True,
                                show_layer_activations=True)

    try:
        print(f'Loading weights from "{MODEL_CHECKPOINT_PREFIX}"')
        model.load_weights(MODEL_CHECKPOINT_PREFIX)
    except ValueError:
        raise Exception(f'Failed to load weights: Weights exists but model is incompatible')
    except tf.errors.NotFoundError:
        raise Exception(f'Failed to load weights: Weights do not exist')
    except Exception as e:
        raise e

    # Plot model metrics
    hist = os.path.join(MODEL_CHECKPOINT_PATH, 'history.json')
    with open(hist, 'r') as f:
        hist = json.load(f)
    
    fig = plt.figure(figsize=(20, 12))
    
    plt.plot(hist['loss'], label='Training Loss', linewidth=1)
    plt.plot(hist['val_loss'], label='Validation Loss', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if args.images:
        fig.savefig(os.path.join(IMAGE_DESTINATION, 'loss.png'))
    # Acquire data and set up analyzer, and plot anomalies
    df, val, data_cols, scalers = get_dataset2_df()

    analyzer = Analyzer(model)
    analyzer.calculate_reconstruction_errors(create_sequence(df, data_cols, DataConfig.SEQUENCE_LENGTH))
    print(f'Reconstruction errors: {analyzer.reconstruction_errors}')
    print(f'Scaled reconstruction errors: {np.array([scalers[i].inverse_transform([[analyzer.reconstruction_errors[i]]]) for i in range(4)]).flatten().tolist()}')

    df_analyze = pd.concat([df, val]).reset_index()
    sequences = create_sequence(df_analyze, data_cols, DataConfig.SEQUENCE_LENGTH)
    dates = mdates.datestr2num(df_analyze['timestamp'])
    
    plt.figure(figsize=(20, 12))
    
    for i, label in enumerate(data_cols):
        ax = plt.subplot(2, 2, i+1)
        plt.xticks(rotation=30)
        plt.title(label)
        plt.plot(dates, df_analyze[label], linewidth=1)
        
        ax.set_xticks(dates[::100])
        ax.set_xticklabels(dates[::100])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    if args.images:
        plt.savefig(os.path.join(IMAGE_DESTINATION, 'org_data.png'))
    
    analyzer.plot_anomalies(sequences, dates, data_cols, scalers=scalers, save_images=args.images)