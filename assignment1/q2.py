import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from data import create_data
from ai import MyModel
import numpy as np
import tensorflow as tf
import time

from custom_metrics import recall_m, precision_m, f1_m

PATH_TO_IMAGES = '/mnt/c/Users/emilk/Downloads/250000_Final/250000_Final'
PATH_TO_IMAGES = 'test_data'
CACHE_DIRECTORY = None
MODEL_NAME = time.strftime("%Y%m%d_%H%M%S")

# GPU Check
gpus = tf.config.list_physical_devices()
print(f'Found devices: {gpus}')

data, val_data = create_data(PATH_TO_IMAGES, cache=CACHE_DIRECTORY)
model = MyModel(model_filters=8, residual_layers=4, residual_filters=16, kernel_size=(5, 5), dropout=0.2, ff_dim=64, NUM_CLASSES=10)
model.compile(
    tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        'accuracy',
        'TruePositives',
        'TrueNegatives',
        'FalsePositives',
        'FalseNegatives'
    ]
)
model.build((None, 64, 64, 1))
model.summary()

# model.fit(data, validation_data=val_data, epochs=1)
# model.save(f'models/model_{MODEL_NAME}')

fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(
    left=0.1,
    bottom=0.1,
    right=0.9,
    top=0.9,
    wspace=0.5,
    hspace=0.5
)
grid = fig.add_gridspec(4, 4)
grids = (grid[0, 0], grid[0, 1], grid[0, 2], grid[0, 3],
         grid[1, 0], grid[1, 1], grid[1, 2], grid[1, 3],
         grid[2, 0], grid[2, 1], grid[2, 2], grid[2, 3],
         grid[3, 0], grid[3, 1], grid[3, 2], grid[3, 3],
         )
N = len(grids)

for images, labels in data.take(1):
    preds = model(images)
    for image, label, grid, img_preds in zip(images, labels, grids[:8], preds):
        ax = fig.add_subplot(grid)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)

        predicted = tf.argmax(img_preds, axis=-1)
        labl = tf.argmax(label).numpy()
        titl = f'{labl} | {predicted.numpy()} : {img_preds[predicted.numpy()]:.2f}'
        ax.set_title(titl)


for images, labels in val_data.take(1):
    preds = model(images)

    for image, label, grid, img_preds in zip(images, labels, grids[8:], preds):
        ax = fig.add_subplot(grid)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)

        predicted = tf.argmax(img_preds, axis=-1)
        labl = tf.argmax(label).numpy()
        titl = f'{labl} | {predicted.numpy()} : {img_preds[predicted.numpy()]:.2f}'
        ax.set_title(titl)

plt.show()