import datetime as dt
import math
import matplotlib
from matplotlib import pyplot
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras

import libs
from libs import datalib, imagelib, modelib

data_dir = './data/train/'
log_dir = './logs/'
model_dir = './models/'

data_size = 64
data_split = 0.9

batch_size = 16
epochs = 10
learn_rate = 0.001

autoencoder_layers = 5
autoencoder_bottleneck = 16
autoencoder_ratio = 5


def build_autoencoder():
    '''
    Builds an encoder and decoder separately, then returns an autoencoder model using a combination of functional & sequential api
    '''

    #region Overview
    print('')
    print('Building model with the following parameters:')
    print('Input Layer:', x_train[0].size)
    nodes_sum = 0
    nodes_sum += x_train[0].size

    for l in range(autoencoder_layers):
        nodes = autoencoder_bottleneck * autoencoder_ratio**(autoencoder_layers - (l+1))
        nodes_sum += nodes
        print('Encoder Layer', l+1, ':', nodes)

    for l in range(autoencoder_layers-1):
        nodes = autoencoder_bottleneck * autoencoder_ratio**(l+1)
        nodes_sum += nodes
        print('Decoder Layer', l+1, ':', nodes)

    print('Output Layer:', x_train[0].size)
    nodes_sum += x_train[0].size
    print('Model Size: ', nodes_sum/1000, "kN")
    print('Compression Factor: ', round(x_train[0].size/autoencoder_bottleneck), '(', round(autoencoder_bottleneck*100/x_train[0].size, 2), '% )')
    print('')
    input('Press Enter to begin...')
    print('')
    #endregion


    #region Encoder assembly
    encoder = keras.Sequential()

    encoder_input = keras.layers.Input(shape=x_train[0].shape)      # [width, height, channels, batch]
    encoder.add(encoder_input)
    encoder.add(keras.layers.Flatten(input_shape=x_train[0].shape))        # because keras does not like to open models wihout an input_shape() in first layer


    for l in range(autoencoder_layers):
        encoder.add(keras.layers.Dense(
            units= autoencoder_bottleneck * autoencoder_ratio**(autoencoder_layers - (l+1)),
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name='Encoder_{}'.format(l+1)))
    #endregion


    #region Decoder assembly
    decoder = keras.Sequential()

    decoder_input = keras.layers.Input(shape=autoencoder_bottleneck)
    decoder.add(decoder_input)
    decoder.add(keras.layers.Flatten(input_shape=(autoencoder_bottleneck, )))

    for l in range(autoencoder_layers-1):
        decoder.add(keras.layers.Dense(
            units= autoencoder_bottleneck * autoencoder_ratio**(l+1),
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name='Decoder_{}'.format(l+1)))


    decoder.add(keras.layers.Dense(
        units=x_train[0].size,
        activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Output'))

    decoder.add(keras.layers.Reshape(x_train[0].shape))
    #endregion


    autoencoder = keras.models.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)))

    print('Model assembled successfully!')
    print('')
    print('Encoder Summary:')
    print(encoder.summary())
    print('')
    print('Decoder Summary:')
    print(decoder.summary())
    print('')
    print('Autoencoder Summary:')
    print(autoencoder.summary())

    return autoencoder, encoder, decoder


def preview_autoencoder():
    fig, axs = pyplot.subplots(3,8)

    for i in range(len(axs[1])):
        if x_test.size is not 0:
            sample = x_test[random.randint(0,x_test.shape[0]-1)]
        else:
            sample = x_train[random.randint(0,x_train.shape[0]-1)]

        # Plot original image
        if x_train[0].ndim == 3:
            axs[0,i].imshow(sample, cmap=None, interpolation=None)      # rgb
        else:
            axs[0,i].imshow(sample, cmap='gray', interpolation=None)        # grayscale

        # Plot encoding
        output_encoded = encoder.predict(np.expand_dims(sample, axis=0), batch_size=None, verbose=0, steps=None, callbacks=None)

        axs[1,i].imshow(modelib.square_encoding(output_encoded), cmap='gray', interpolation=None)
        
        # Plot decoding
        output_decoded = decoder.predict(output_encoded, batch_size=None, verbose=0, steps=None, callbacks=None)
        output_decoded = np.squeeze(output_decoded, axis=0)

        if x_train[0].ndim == 3:
            axs[2,i].imshow(output_decoded, cmap=None, interpolation=None)
        else:
            axs[2,i].imshow(output_decoded, cmap='gray', interpolation=None)

    fig.canvas.set_window_title('Autoencoder Output')
    
    pyplot.show()


def save_autoencoder():
    print('Saving model to disk...')

    model_dir_save = model_dir + os.path.splitext(os.path.basename(__file__))[0] + '_' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
    os.makedirs(model_dir_save)

    autoencoder.save(model_dir_save + 'autoencoder'  + '.h5')
    encoder.save(model_dir_save + 'encoder'  + '.h5')
    decoder.save(model_dir_save + 'decoder'  + '.h5')

    print('Success!')



if __name__ == '__main__':

    #region Data Import
    if os.listdir(data_dir) is not None:
        print('Loading data...')

        data = np.concatenate([imagelib.load_img(path=data_dir+i, size=data_size) for i in os.listdir(data_dir)])
        datalib.inspect(data)

        x_train, x_test = datalib.split(data, data_split)
    else:
        print('No data located. Importing toy dataset...')

        x_train, _, x_test, _ = datalib.load_toy('A')

        datalib.inspect(x_train, 'MNIST_training')
        datalib.inspect(x_test, 'MNIST_testing')
    #endregion

    autoencoder, encoder, decoder = build_autoencoder()

    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=learn_rate), loss='mse', metrics=['mae'])

    print('')
    print('Beginning training process...')
    if x_test.size is not 0:
        autoencoder.fit(
            x=x_train,
            y=x_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=None,         # modelib.callbacks(log_dir, model_dir)
            validation_data=(x_test,x_test),
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None)
    else:
        autoencoder.fit(
            x=x_train,
            y=x_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=None,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None)

    print('Training successful!')

    preview_autoencoder()

    if input('Save model? [Y/N]: ') == 'y':
        save_autoencoder()
    else: print('Model sent to android hell')


    print('\nDebug:\n$tensorboard --logdir=logs/')