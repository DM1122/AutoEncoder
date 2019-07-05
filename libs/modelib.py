import datetime as dt
import math
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


def callbacks(log, model):
    '''
    Returns configured keras callbacks.

    Args:
        log : log directory
        model : model directory 
    '''

    callback_NaN = keras.callbacks.TerminateOnNaN()     # NaN callback

    callback_checkpoint = keras.callbacks.ModelCheckpoint(      # checkpoint callback
        filepath=model,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)
    
    callback_early_stopping = keras.callbacks.EarlyStopping(        # early stopping callback
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto',
        baseline=None)

    callback_tensorboard = keras.callbacks.TensorBoard(     # tensorboard callback
        log_dir=log,
        histogram_freq=5,
        batch_size=32,
        write_graph=True,
        write_grads=True,
        write_images=False)

    callbacks = [callback_NaN, callback_checkpoint, callback_early_stopping, callback_tensorboard]

    return callbacks


def square_encoding(data):
    if int(math.sqrt(data.size) + 0.5) ** 2 == data.size:         # check if vector is perfect square
        data = np.reshape(data, (int(math.sqrt(data.size)) , int(math.sqrt(data.size) )))       # reshape vector to perfect square
    
    return data


def flatten(data):
    data = np.reshape(data, (1,data.size))
    return data
