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

    model_encoder = keras.Model(inputs=model.input, outputs=model.get_layer('Bottleneck').output)
    # model_decoder = keras.Model(inputs=keras.Input(shape=model.get_layer('Bottleneck').output.shape), outputs=model.output)


    input_layer_encoded = keras.layers.Input(shape=(model.get_layer('Bottleneck').output.shape[1],))

    # model_decoder = keras.models.Model(inputs=input_layer_encoded, outputs=model.layers[-1](input_layer_encoded))
    model_decoder = keras.models.Model(inputs=input_layer_encoded, outputs=model.layers[-1].output)

    print(model_encoder.layers)
    print(model_decoder.layers)

    return model_encoder, model_decoder



    # LEGACY (replaced with functional api)
    # model = keras.Sequential()

    # model.add(keras.layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2])))

    # # Encoder layers
    # for l in range(encoder_layers-1):
    #     model.add(keras.layers.Dense(
    #         units= encoder_bottleneck_size * encoder_layers_ratio**(encoder_layers - l),
    #         activation='relu',
    #         use_bias=True,
    #         kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros',
    #         kernel_regularizer=None,
    #         bias_regularizer=None,
    #         activity_regularizer=None,
    #         kernel_constraint=None,
    #         bias_constraint=None,
    #         name='Encoder_{}'.format(l+1)))

    # # Bottleneck
    # model.add(keras.layers.Dense(
    #     units=encoder_bottleneck_size,
    #     activation='relu',
    #     use_bias=True,
    #     kernel_initializer='glorot_uniform',
    #     bias_initializer='zeros',
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     bias_constraint=None,
    #     name='Bottleneck'))

    # # Decoder layers
    # for l in range(encoder_layers-1):
    #     model.add(keras.layers.Dense(
    #         units= encoder_bottleneck_size * encoder_layers_ratio**(l+1),
    #         activation='relu',
    #         use_bias=True,
    #         kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros',
    #         kernel_regularizer=None,
    #         bias_regularizer=None,
    #         activity_regularizer=None,
    #         kernel_constraint=None,
    #         bias_constraint=None,
    #         name='Decoder_{}'.format(l+1)))
    
    # # Output layer
    # model.add(keras.layers.Dense(
    #     units=x_train[0].size,
    #     activation='relu',
    #     use_bias=True,
    #     kernel_initializer='glorot_uniform',
    #     bias_initializer='zeros',
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     bias_constraint=None,
    #     name='Output'))

    # model.add(keras.layers.Reshape((x_train.shape[1], x_train.shape[2])))
