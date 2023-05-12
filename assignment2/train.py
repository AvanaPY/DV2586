import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from data import get_data
from data.data import DataConfig
from model import build_encoder_decoder_lstm

CHECKPOINT_PATH = './checkpoints/run27/ckpt'
EPOCHS = 50

(sequences, y_train), (sequences, y_test) = get_data()

model = build_encoder_decoder_lstm()
model.summary()

try:
    print(f'Loading weights from "{CHECKPOINT_PATH}"')
    model.load_weights(CHECKPOINT_PATH)
except:
    print(f'Failed to load weights.')
    
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH, 
        monitor='loss',
        save_weights_only=True)
]

model.fit(
    sequences, sequences, 
    validation_data=(sequences, sequences),
    batch_size=DataConfig.BATCH_SIZE, 
    epochs=EPOCHS, 
    callbacks=callbacks,
    verbose=1
)