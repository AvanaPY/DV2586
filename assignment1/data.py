import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import tensorflow as tf

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024
PREFETCH_BUFFER_SIZE = 4

@tf.function
def scale_values(image, label):
    image = image / 255.0
    return image, label

@tf.function
def resize_image(image, label):
    image = tf.image.resize(image, get_image_resize_dimensions())
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
    image = tf.image.resize(image, get_image_resize_dimensions())
    image = tf.image.rgb_to_grayscale(image)
    label = tf.one_hot(label, 10)
    return image, label

def get_image_resize_dimensions():
    return (48, 48)

def create_data(path : str, cache : str = None):
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
            image_size=get_image_resize_dimensions(),
            validation_split=0.2,
            subset='both'
            )

        validation_data = (
            validation_data
            .map(process_image)
        )

        dataset = (
            dataset
            .map(process_image)
        )

        if cache and not os.path.exists(cache):
            os.makedirs(cache)
            print(f'Writing to cache directory "{cache}"...')
            dataset.save(os.path.join(cache, 'train.tfds'))
            validation_data.save(os.path.join(cache, 'validation.tfds'))
            print('Finished.')

        t_ds = dataset
        v_ds = validation_data

    else:
        print(f'Found cache directory "{cache}", loading...')
        t_ds = tf.data.Dataset.load(os.path.join(cache, 'train.tfds'))
        v_ds = tf.data.Dataset.load(os.path.join(cache, 'validation.tfds'))
        print(f'Loaded cached data.')

    t_ds = (
        t_ds
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(PREFETCH_BUFFER_SIZE)
    )

    v_ds = (
        v_ds
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(PREFETCH_BUFFER_SIZE)
    )
    return t_ds, v_ds