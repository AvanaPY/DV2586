import os
from typing import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model

IMAGE_DESTINATION = os.environ['__IMAGE_DESTINATION']

class Analyzer:
    def __init__(self, model : Model):
        self._model = model
        self._reconstruction_errors = 0
        
    @property
    def model(self) -> Model:
        return self._model
    
    @property
    def reconstruction_errors(self) -> np.ndarray:
        return self._reconstruction_errors
    
    def calculate_reconstruction_errors(self, sequences : np.ndarray) -> np.ndarray:
        yhat = self._model.predict(sequences, verbose=0)
        e = np.abs(yhat - sequences)
        error = e[0,:,:]
        error = np.concatenate([error, e[1:,-1,:]])
        self._reconstruction_errors = np.max(error, axis=0)
        return self._reconstruction_errors
    
    def plot_anomalies(self, sequences : np.ndarray, 
                             dates : List[Any], 
                             data_cols : List[str], 
                             scalers : Optional[List[MinMaxScaler]]=None,
                             save_images : bool = False) -> None:
        org_data = sequences[:,0,:]                                # This line reconstructs the original data
        org_data = np.concatenate([org_data, sequences[-1,1:,:]])
        
        yhat = self._model.predict(sequences, verbose=0)
        err = np.abs(yhat - sequences)
        error = err[0,:,:]
        error = np.concatenate([error, err[1:,-1,:]])
        
        # We use the same method to build a list of the reconstructed data
        begin = yhat[0,:,:]
        reconstructed_data = yhat[1:,-1,:]
        reconstructed_data = np.concatenate([begin, reconstructed_data])
        
        # Create a dataframe
        df = pd.DataFrame(
            error,
            columns=[col + ' MAE' for col in data_cols]
        )
        
        for i, col in enumerate(data_cols):
            df[col] = org_data.T[i]
            df[col + ' Reconstruct'] = reconstructed_data.T[i]
            df[col + ' Threshold'] = self._reconstruction_errors[i]
            df[col + ' Anomaly'] = df[col + ' MAE'] > df[col + ' Threshold']

        df['timestamp'] = mdates.num2date(dates)
        
        # Scale the data to original range if we are provided with scalers
        if scalers:
            assert len(data_cols) == len(scalers), f'Number of data points ({len(data_cols)}) do not match number of scalers ({len(scalers)})'
            for i, col in enumerate(data_cols):
                df[[col]] = scalers[0].inverse_transform(df[[col]])
                df[[col + ' Reconstruct']] = scalers[0].inverse_transform(df[[col + ' Reconstruct']])
                df[[col + ' Threshold']] = scalers[0].inverse_transform(df[[col + ' Threshold']])

        fig = plt.figure(figsize=(20, 12))
        for i, column in enumerate(data_cols):
            ax = plt.subplot(2, 2, i+1)
            plt.xticks(rotation=30)
            plt.title(column)
            
            plt.plot(dates, df[column], label='Real data', linewidth=1)
            plt.plot(dates, df[column + ' Reconstruct'], linewidth=1, linestyle='dashed', label='Reconstructed')
            anomalies = df[df[column + ' Anomaly'] == True]
            plt.scatter(
                anomalies['timestamp'], 
                anomalies[column],
                marker='.',
                c='red',
                s=4,
                label='Anomaly',
                zorder=10)
            plt.legend()
            
            ax.xaxis_date()
            ax.set_xticks(dates[::100],)
            ax.set_xticklabels(dates[::100])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
        if save_images:
            fig.savefig(os.path.join(IMAGE_DESTINATION, 'reconstructed_anomalies.png'))
        
        fig = plt.figure(figsize=(20, 12))
        for i, column in enumerate(data_cols):
            ax = plt.subplot(2, 2, i+1)
            plt.title(f'{column}')
            plt.xticks(rotation=30)
            
            plt.plot(dates, df[column + ' Reconstruct'], label='Reconstructed')
            plt.plot(dates, df[column] + df[column + ' Threshold'], linewidth=1, linestyle='dashed', label='Thresholds', c='orchid')
            plt.plot(dates, df[column] - df[column + ' Threshold'], linewidth=1, linestyle='dashed', c='orchid')
            plt.legend()
            
            ax.xaxis_date()
            ax.set_xticks(dates[::100])
            ax.set_xticklabels(dates[::100])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
        if save_images:
            fig.savefig(os.path.join(IMAGE_DESTINATION, 'reconstructed_thresholds.png'))

        plt.show()