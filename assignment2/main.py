import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
np.set_printoptions(suppress=True, precision=4)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.close("all")

from data import get_data, get_dataset2_df
from model import build_encoder_decoder_lstm

# df, data_columns = get_dataset2_df().head(200)

# df_dates = df['timestamp']
# dates = mdates.datestr2num(df_dates)

# fig = plt.figure(figsize=(18, 8))
# ax = fig.subplots(1, 1)

# ax.plot(dates, df['Bearing 1'])
# ax.plot(dates, df['Bearing 2'])
# ax.plot(dates, df['Bearing 3'])
# ax.plot(dates, df['Bearing 4'])
# ax.xaxis_date()
# ax.set_xticks(df_dates[::39])
# ax.set_xticklabels(df_dates[::39])
# ax.set_ylim(0, 1)

# fig.autofmt_xdate(rotation=30)

# plt.show()

ds, val_ds = get_data()
model = build_encoder_decoder_lstm()
model.summary()

model.fit(ds, epochs=20)

re = model.calculate_reconstruction_error(ds)
print(f'Reconstruction error: {re}')

# model.plot_anomalies(val_ds)

for batch in val_ds.take(1):
    xs, ys = batch
    
    yhat = model(xs)
    error = ys - yhat
    
    print(f'True Ys:')
    print(np.c_[ys[0], ys[1]])
    print(f'Predicted Ys:')
    print(np.c_[yhat[0], yhat[1]])
    print(f'Errors:')
    print(np.c_[error[0], error[1]])
    
    # test_mae = np.mean(np.abs(error), axis=1)
    # plt.hist(test_mae, bins=50)
    # plt.xlabel('MAE')
    # plt.ylabel('Count')
    # plt.show()
    # print(f'{MAE=}')