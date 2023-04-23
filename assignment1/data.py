import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import tensorflow as tf

RESIZE_IMAGE_DIMESIONS = (64, 64)
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 1024

@tf.function
def scale_values(image, label):
    image = image / 255.0
    return image, label

@tf.function
def resize_image(image, label):
    image = tf.image.resize(image, RESIZE_IMAGE_DIMESIONS)
    return image, label

@tf.function
def to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

def create_data(path : str, cache : str = None):
    if cache and os.path.exists(cache):
        t_ds = tf.data.Dataset.load(os.path.join(cache, 'train.tfds'))
        v_ds = tf.data.Dataset.load(os.path.join(cache, 'validation.tfds'))
        return t_ds, v_ds

    if not os.path.exists(path):
        raise Exception(f'Cannot find path to images: "{path}"')
    print(f'Found folder with images: "{path}"')

    dataset, validation_data = tf.keras.utils.image_dataset_from_directory(
    # dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode='int', 
        shuffle=False,
        batch_size=None, 
        image_size=RESIZE_IMAGE_DIMESIONS,
        validation_split=0.2,
        subset='both'
        )

    
    validation_data = (
        # dataset.take(20)
        validation_data
        .map(scale_values)
        .map(to_grayscale)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

    dataset = (
        dataset
        .map(scale_values)
        .map(to_grayscale)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

    if cache and not os.path.exists(cache):
        os.makedirs(cache)
        dataset.save(os.path.join(cache, 'train.tfds'))
        validation_data.save(os.path.join(cache, 'validation.tfds'))
    return dataset, validation_data