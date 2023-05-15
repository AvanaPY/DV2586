# DV2586 - Assignment 2

This project is written in Python 3.9.15.

It should be plug-and-play, given you have all the required libraries installed, which includes 

* Numpy
* Tensorflow 2
* Keras
* Matplotlib
* Pandas
* sklearn

## Python Files

### `train.py`

The file `train.py` can be immediately started with `python train.py`, or alternatively check out the available flags with `python train.py -h`. The program trains a model, saves the weights in terms of a checkpoint.

### `analyze.py`

The file `analyze.py` can be immediately started with `python analyze.py`, or alternatively check out the available flags with `python analyze.py -h`. The program creates a tensorflow model, loads weights if available, and analyzes the data stored in `data/dataset2.csv` and plots any anomalies found. 

Do note that the default weights to load is set to `latest`, which means that it will load the latest weights that have been created. This program should be turned in with the weights for the latest model, which is the model used in the written PDF file. The weights are stored in the directory `checkpoints` with a `checkpoint_name`, i.e the weights included in this go by the `checkpoint_name` of "model_weights".

You should be able to immediately call `python analyze.py`, given the required libraries are installed.

### `data/data.py`

Contains functions and classes for data preparation and acquisition.

### `model/analyzer.py`

Contains the Analyzer class which analyzes data and creates plots.

### `model/model.py`

Contains functions and classes for building the Tensorflow 2 model.