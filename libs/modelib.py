import datetime as dt
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

def split_autoencoder(model):
    '''
    Splits an autoencoder model into its encoder and decoder components.

    Args:
        model: compiled keras autoencoder
    '''

    print(model.get_layer('Bottleneck').output.shape)
    model_encoder = keras.Model(inputs=model.input, outputs=model.get_layer('Bottleneck').output)
    model_decoder = keras.Model(inputs=keras.Input(shape=model.get_layer('Bottleneck').output.shape), outputs=model.output)

    return model_encoder, model_decoder