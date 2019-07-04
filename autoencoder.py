
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

data_dir = './data/'
log_dir = './logs/'
model_dir = './models/'

autoencoder_layers = 3
autoencoder_bottleneck = 128
autoencoder_ratio = 4

batch_size = 1
epochs = 100
learn_rate = 0.001

data_split = 1
data_rgb = True



def build_autoencoder(shape_x, shape_y, shape_z=None):
    '''
    Builds an encoder and decoder separately, then returns an autoencoder model using a combination of functional & sequential api

    Args:
        shape_x: input width
        shape_y: input height
        shape_z: input channels (RGB)
    '''

    print('Building model with the following parameters:')
    if data_rgb:
        print('Input Layer:', shape_x*shape_y*shape_z)
    else:
        print('Input Layer:', shape_x*shape_y)
    for l in range(autoencoder_layers):
        print('Encoder Layer', l+1, ':', autoencoder_bottleneck * autoencoder_ratio**(autoencoder_layers - (l+1)))
    for l in range(autoencoder_layers-1):
        print('Decoder Layer', l+1, ':', autoencoder_bottleneck * autoencoder_ratio**(l+1))
    if data_rgb:
        print('Output Layer:', shape_x*shape_y*shape_z)
    else:
        print('Output Layer:', shape_x*shape_y)
    print('')
    input('Press Enter to continue...')


    #region Encoder assembly
    encoder = keras.Sequential()

    if shape_z is not None:
        encoder_input = keras.layers.Input(shape=(shape_x, shape_y, shape_z, ))      # [width, height, channels, batch]
    else:
        encoder_input = keras.layers.Input(shape=(shape_x, shape_y, ))      # [width, height, batch]
    
    encoder.add(encoder_input)
    
    encoder.add(keras.layers.Flatten())

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

    decoder_input = keras.layers.Input(shape=(autoencoder_bottleneck, ))
    decoder.add(decoder_input)

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

    if shape_z is not None:
        decoder.add(keras.layers.Dense(
            units=shape_x * shape_y * shape_z,
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

        decoder.add(keras.layers.Reshape((shape_x, shape_y, shape_z)))

    else:
        decoder.add(keras.layers.Dense(
            units=shape_x * shape_y,
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

        decoder.add(keras.layers.Reshape((shape_x, shape_y))) 
    #endregion


    autoencoder = keras.models.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)))

    return autoencoder, encoder, decoder

def preview_autoencoder():
    fig, axs = pyplot.subplots(3,8)

    for i in range(len(axs[1])):
        if x_test.size is not 0:
            sample = x_test[random.randint(0,x_test.shape[0]-1)]
        else:
            sample = x_train[random.randint(0,x_train.shape[0]-1)]

        # Plot original image
        if data_rgb:
            axs[0,i].imshow(sample, cmap=None, interpolation=None)
        else:
            axs[0,i].imshow(sample, cmap='gray', interpolation=None)

        # Plot encoding
        output_encoded = encoder.predict(np.expand_dims(sample, axis=0), batch_size=None, verbose=0, steps=None, callbacks=None)
        output_encoded = np.squeeze(output_encoded, axis=0)

        if int(math.sqrt(output_encoded.size) + 0.5) ** 2 == output_encoded.size:         # check if vector is perfect square and can be displayed in 2D
            output_encoded_reshaped = np.reshape(output_encoded, (int(math.sqrt(output_encoded.size)) , int(math.sqrt(output_encoded.size) )))       # reshape vector to perfect square
            axs[1,i].imshow(output_encoded_reshaped, cmap='gray', interpolation=None)
        else:
            axs[1,i].imshow(np.expand_dims(output_encoded, axis=0), cmap='gray', interpolation=None)
        
        # Plot decoding
        output_decoded = decoder.predict(np.expand_dims(output_encoded, axis=0), batch_size=None, verbose=0, steps=None, callbacks=None)
        output_decoded = np.squeeze(output_decoded, axis=0)

        if data_rgb:
            axs[2,i].imshow(output_decoded, cmap=None, interpolation=None)
        else:
            axs[2,i].imshow(output_decoded, cmap='gray', interpolation=None)        

    # fig.tight_layout()
    fig.canvas.set_window_title('Autoencoder Output')
    
    pyplot.show()



if __name__ == '__main__':

    #region Data Import
    if os.listdir(data_dir) is not None:
        print('Loading data...')

        data = np.concatenate([imagelib.load_img(data_dir + i) for i in os.listdir(data_dir)])
        datalib.inspect(data)

        x_train, x_test = datalib.split(data, data_split)
    else:
        print('No data located. Importing toy dataset...')

        x_train, _, x_test, _ = datalib.load_toy('A')

        datalib.inspect(x_train, 'MNIST_training')
        datalib.inspect(x_test, 'MNIST_testing')
    #endregion

    if data_rgb:
        autoencoder, encoder, decoder = build_autoencoder(x_train.shape[1], x_train.shape[2], x_train.shape[3])
    else:
        autoencoder, encoder, decoder = build_autoencoder(x_train.shape[1], x_train.shape[2])
    
    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=learn_rate), loss='mse', metrics=['mae'])

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
            callbacks=None,         # modelib.callbacks(log_dir, model_dir)
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
        print('Saving model to disk...')
        model_dir_save = model_dir + os.path.splitext(os.path.basename(__file__))[0] + '_' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
        os.makedirs(model_dir_save)
        autoencoder.save(model_dir_save + 'autoencoder'  + '.h5')
        encoder.save(model_dir_save + 'encoder'  + '.h5')
        decoder.save(model_dir_save + 'decoder'  + '.h5')
        print('Success!')
    else:
        print('Model sent to android hell')


    print('Debug:\n$tensorboard --logdir=logs/')