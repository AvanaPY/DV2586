import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from data import create_data
from ai import MyModel
import numpy as np
import tensorflow as tf
import time
import argparse

from custom_metrics import recall_m, precision_m, f1_m
from data import get_image_resize_dimensions

PATH_TO_IMAGES = '/mnt/c/Users/emilk/Downloads/250000_Final/250000_Final'
PATH_TO_IMAGES = '250000_Final'
CACHE_DIRECTORY = 'cache2'
MODEL_NAME = time.strftime("%Y%m%d_%H%M%S")

# GPU Check
gpus = tf.config.list_physical_devices('GPU')
print(f'Found devices: {gpus}')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        gpu, True
    )

def main(args):
    data, val_data = create_data(PATH_TO_IMAGES, cache=CACHE_DIRECTORY)
    
    if args.load_model is None:
        if not args.compile_model or not args.build_model:
            raise Exception(f'If you create a new model you must compile and build it with the --compile-model and --build-model flags')
        model = MyModel(model_filters=8, 
                        residual_layers=4, 
                        residual_filters=32,
                        residual_kernel_size=(3, 3), 
                        dropout=0.2, 
                        ff_dim=128, 
                        NUM_CLASSES=10)
    else:
        model = tf.keras.models.load_model(args.load_model)

    if args.compile_model:
        lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000,
            decay_rate=0.96
        )
        model.compile(
            tf.keras.optimizers.Adam(
                learning_rate=lr_sched
            ), 
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                'accuracy',
                'TruePositives',
                'TrueNegatives',
                'FalsePositives',
                'FalseNegatives'
            ]
        )
    
    if args.build_model:
        if not args.load_model is None:
            raise Exception(f'Running Build() on a SavedModel is not supported.')
        image_dims = get_image_resize_dimensions()
        model.build((None, image_dims[0], image_dims[1], 1))
        
    if args.verbose:
        model.summary()

    if args.fit:
        model.fit(data, validation_data=val_data, epochs=args.epochs)
        if not args.no_save:
            model.save(f'models/model_{MODEL_NAME}')
            print(f'Saved model to "models/model_{MODEL_NAME}"')

    if args.plot:
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

        for images, labels in val_data.take(1):
            preds = model(images)

            for image, label, grid, img_preds in zip(images, labels, grids[:], preds):
                ax = fig.add_subplot(grid)
                ax.imshow(image, cmap='gray', vmin=0, vmax=1)

                predicted = tf.argmax(img_preds, axis=-1)
                labl = tf.argmax(label).numpy()
                titl = f'{labl} | {predicted.numpy()} : {img_preds[predicted.numpy()]:.2f}'
                ax.set_title(titl)

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', '-l', type=str, default=None, help='Toggle to load a model followed by its name')
    parser.add_argument('--compile-model', '-c', action='store_true', help='Toggle to compile model')
    parser.add_argument('--build-model', '-b', action='store_true', help='Toggle to build model')
    parser.add_argument('--plot', '-p', action='store_true', help='Plot results in a figure')
    parser.add_argument('--fit', '-f', action='store_true', help='Toggle to fit the model to data')
    parser.add_argument('--epochs', '-e', type=int, help='How many epochs run when fitting the model')
    parser.add_argument('--no-save', action='store_false', help='Toggle off to not save the model after fitting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Turn on verbosity')
    args = parser.parse_args()

    main(args)