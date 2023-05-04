from typing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1024
PREFETCH_BUFFER_SIZE = 2

@tf.function
def scale_values(image, label):
    image = image / 255.0
    return image, label

@tf.function
def resize_image(image, label):
    image = tf.image.resize(image, get_image_dimensions())
    return image, label

@tf.function
def to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

@tf.function
def one_hot_y(image, label):
    return image, tf.one_hot(label, 10)

@tf.function
def process_image(image, label):
    image = image / 255.0
    image = tf.image.resize(image, get_image_dimensions())
    image = tf.image.rgb_to_grayscale(image)
    label = tf.one_hot(label, 10)
    return image, label

@tf.function
def transfer_process_image(image, label):
    image = tf.image.resize(image, get_image_dimensions())
    image = preprocess_input(image)
    label = tf.one_hot(label, 10)
    return image, label
    
def get_image_dimensions():
    return (48, 48)

def get_or_create_data(path : str, cache : str = None, batch_size : Optional[int] = None, for_model : Optional[str] = None):
    if not cache or not os.path.exists(cache):

        if not os.path.exists(path):
            raise Exception(f'Cannot find path to images: "{path}"')
        print(f'Found folder with images: "{path}"')

        dataset, validation_data = tf.keras.utils.image_dataset_from_directory(
            path,
            label_mode='int', 
            shuffle=True,
            seed=69420,
            batch_size=None, 
            image_size=get_image_dimensions(),
            validation_split=0.2,
            subset='both'
            )

        image_processor = {
            None : process_image,
            'resnet' : transfer_process_image,
            'vgg19'  : transfer_process_image,
            'densenet' : transfer_process_image
        }[for_model]

        validation_data = (
            validation_data
            .map(image_processor)
        )

        dataset = (
            dataset
            .map(image_processor)
        )

        if cache and not os.path.exists(cache):
            os.makedirs(cache)
            print(f'Writing to cache directory "{cache}"...')
            dataset.save(os.path.join(cache, 'train.tfds'))
            validation_data.save(os.path.join(cache, 'validation.tfds'))
            print('Finished writing to cache directory.')

        t_ds = dataset
        v_ds = validation_data

    else:
        print(f'Found cache directory "{cache}", loading...')
        t_ds = tf.data.Dataset.load(os.path.join(cache, 'train.tfds'))
        v_ds = tf.data.Dataset.load(os.path.join(cache, 'validation.tfds'))
        print(f'Loaded cached data.')

    batch_sz = BATCH_SIZE if not batch_size else batch_size
    t_ds = (
        t_ds
        .take(50_000)
        .batch(batch_sz)
        .cache()
        .prefetch(PREFETCH_BUFFER_SIZE)
    )

    v_ds = (
        v_ds
        .batch(batch_sz)
        .cache()
        .prefetch(PREFETCH_BUFFER_SIZE)
    )
    return t_ds, v_ds