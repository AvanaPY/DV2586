import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50 as BaseModel
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
import numpy as np
import argparse

from data import get_or_create_data

class VGG19_10(Model):
    def __init__(self):
        super().__init__()
        self._rnet = BaseModel(
            weights='imagenet',
            include_top=False,
            input_shape=(48,48,3),
            classes=10
        )
        self._pooling = GlobalAveragePooling2D()
        self._dense = Dense(10, activation='softmax')
        
    def call(self, inputs):
        x = self._rnet(inputs)
        x = self._pooling(x)
        x = self._dense(x)
        return x

    # Work-around to get .summary() to properly work and display output shapes with subclassed Models
    def summary(self):
        image_dims = (48, 48)
        x = Input(shape=(image_dims[0], image_dims[1], 3))
        Model(inputs=[x], outputs=self.call(x)).summary()

BASE_MODEL_NAME = 'resnet'

def main(args):
    data, val_data = get_or_create_data('250000_Final', f'cache_{BASE_MODEL_NAME}', 256, BASE_MODEL_NAME)

    if args.load_model:
        model = tf.keras.models.load_model(f'models/{BASE_MODEL_NAME}_10')
    else:
        model = VGG19_10()
        model.compile(
            tf.keras.optimizers.Adam(
                learning_rate=1e-4
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
        model.build((None, 48, 48, 3))
    model.summary()

    if args.fit:
        model.fit(data, epochs=5)
        model.save(f'models/{BASE_MODEL_NAME}_10')

    if args.evaluate:
        print(f'Evaluating model metrics...')
        evaluations = model.evaluate(val_data)
        loss, accuracy, tp, tn, fp, fn = evaluations
        
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        f         = 2 * precision * recall / (precision + recall)
        
        true_acc = (tp + tn) / (tp + fp + fn + tn)
        
        for metric, metric_name in zip(evaluations + [true_acc, precision, recall, f], ['Loss', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'True Accuracy', 'Precision', 'Recall', 'F1']):
            print(f'{metric_name.rjust(18)} : {metric:10.3f}')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', action='store_true', default=None, help='Toggle to load the saved model')
    parser.add_argument('--fit', '-f', action='store_true', help='Toggle to fit the model')
    parser.add_argument('--evaluate', '-e', action='store_true', help='Toggle to evaluate the model')
    args = parser.parse_args()
    main(args)