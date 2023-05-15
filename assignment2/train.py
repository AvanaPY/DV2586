import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json
from data import get_data
from data.data import DataConfig
from model import build_encoder_decoder_lstm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=10, help='How many total epochs to train the model.')
    parser.add_argument('--checkpoint-name', '-c', type=str, default='model1', help='The name of the checkpoint.')
    args = parser.parse_args()
    
    assert args.checkpoint_name != 'latest', f'`latest` is a reserved checkpoint name. Please choose another.'
    
    # Parameters
    CHECKPOINT_PATH = os.path.join('./checkpoints', args.checkpoint_name)
    CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_PATH, 'ckpt')
    EPOCHS = args.epochs

    # Build the model
    model = build_encoder_decoder_lstm()
    model.summary()

    try:
        print(f'Loading weights from "{CHECKPOINT_PREFIX}"')
        model.load_weights(CHECKPOINT_PREFIX)
    except ValueError:
        raise Exception(f'Failed to load weights: Weights exists but model is incompatible')
    except tf.errors.NotFoundError:
        print(f'Failed to load weights: Weights do not exist')
    except Exception as e:
        raise e
    
    # Acquire data and train the model
    (x_train, y_train), (x_test, y_test) = get_data()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PREFIX, 
            monitor='loss',
            save_weights_only=True)
    ]

    hist = model.fit(
        x_train, x_train, 
        validation_data=(x_test, x_test),
        batch_size=DataConfig.BATCH_SIZE, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        verbose=1
    )

    # Save history to json file
    hist_dict = hist.history
    dict_keys = ['loss', 'val_loss']
    j = {
        key:hist_dict[key] for key in dict_keys
    }
    json_path = os.path.join(CHECKPOINT_PATH, 'history.json')

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            org_data = json.load(f)
            j = {key:org_data[key] + j[key] for key in dict_keys}
            
    with open(json_path, 'w') as f:
        json.dump(j, f)