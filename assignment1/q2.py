import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from data import create_data
from ai import MyModel

import tensorflow as tf

PATH_TO_IMAGES = '/mnt/c/Users/emilk/Downloads/250000_Final/250000_Final'
PATH_TO_IMAGES = 'test_data'

data, val_data = create_data(PATH_TO_IMAGES, cache='cache_data')
model = MyModel(model_filters=4, residual_filters=4, kernel_size=(3, 3), dropout=0.2, ff_dim=128, NUM_CLASSES=10)
model.compile(
    tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.fit(data, validation_data=val_data, epochs=100)

fig = plt.figure()
grid = fig.add_gridspec(2, 2)
grids = (grid[0,0], grid[0, 1], grid[1, 0], grid[1, 1])

for images, labels in data.take(1):
    for image, label, grid in zip(images, labels, grids):
        ax = fig.add_subplot(grid)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(label.numpy())

for images, labels in val_data.take(1):
    for image, label, grid in zip(images, labels, grids[2:]):
        ax = fig.add_subplot(grid)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(label.numpy())

model.build((1, 64, 64, 1))
model.summary()

plt.show()