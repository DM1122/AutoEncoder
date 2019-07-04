
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
autoencoder_bottleneck = 64
autoencoder_ratio = 2

batch_size = 256
epochs = 5
learn_rate = 0.01


def build_autoencoder_funcapi(input_x, input_y):
    '''
    Builds an autoencoder using the Functional api

    Args:
        input_x: input width
        input_y: input height
    '''

    # Create model
    input_layer = keras.layers.Input(shape=(input_x,input_y,))      # [width, height, batch]

    encoder_layer = keras.layers.Flatten()(input_layer)

    encoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * encoder_layers_ratio**(autoencoder_layers),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Encoder_')(encoder_layer)
    
    encoder_layer = keras.layers.Dense(
        units=autoencoder_bottleneck,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Bottleneck')(encoder_layer)

    decoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * encoder_layers_ratio**(3),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Decoder_')(encoder_layer)
    
    decoder_layer = keras.layers.Dense(
        units=input_x * input_y,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Output')(decoder_layer)
    
    decoder_layer = keras.layers.Reshape((input_x,input_y))(decoder_layer)


    # Build autoencoder
    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder_layer)
    
    # Build encoder
    encoder = keras.models.Model(inputs=input_layer, outputs=encoder_layer)

    # Build decoder
    # create a placeholder for an encoded (32-dimensional) input
    decoder_input = keras.layers.Input(shape=(autoencoder_bottleneck,))
    # retrieve the last layer of the autoencoder model
    decoder_output = autoencoder.layers[-1]
    # create the decoder model
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output(decoder_input))


    # input_layer_encoded = keras.layers.Input(shape=(autoencoder_bottleneck,))
    # decoder = keras.models.Model(inputs=input_layer_encoded, outputs=autoencoder.layers[-1](input_layer_encoded))

    print(autoencoder.layers)
    print(encoder.layers)
    print(decoder.layers)

    return autoencoder, encoder, decoder

def build_autoencoder_funcapi_seg(input_x, input_y):
    '''
    Builds an encoder and decoder separately, then returns an autoencoder model using the functional api

    Args:
        input_x: input width
        input_y: input height
    '''


    # Encoder
    encoder_input = keras.layers.Input(shape=(input_x,input_y,))      # [width, height, batch]

    encoder_layer = keras.layers.Flatten()(encoder_input)

    encoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * autoencoder_ratio**(autoencoder_layers),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Encoder_')(encoder_layer)

    encoder_layer = keras.layers.Dense(
        units=autoencoder_bottleneck,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Bottleneck')(encoder_layer)

    encoder = keras.models.Model(inputs=encoder_input, outputs=encoder_layer)


    # Decoder
    decoder_input = keras.layers.Input(shape=(autoencoder_bottleneck,))

    decoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * autoencoder_ratio**(3),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Decoder_')(decoder_input)

    decoder_layer = keras.layers.Dense(
        units=input_x * input_y,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Output')(decoder_layer)
    
    decoder_layer = keras.layers.Reshape((input_x,input_y))(decoder_layer)

    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_layer)

    # Autoencoder
    autoencoder = keras.models.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)))

    return autoencoder, encoder, decoder
    
def build_autoencoder(shape_x, shape_y, shape_z=None):
    '''
    Builds an encoder and decoder separately, then returns an autoencoder model using a combination of functional & sequential api

    Args:
        shape_x: input width
        shape_y: input height
        shape_z: input channels (likely RGB)
    '''
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
            activation='relu',
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
            activation='relu',
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


    #region Autoencoder assembly
    autoencoder = keras.models.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)))
    #endregion

    return autoencoder, encoder, decoder






if __name__ == '__main__':

    #region Data Import
    if not os.listdir(data_dir):
        print('No data located. Importing toy dataset...')

        x_train, _, x_test, _ = datalib.load_toy('A')

        datalib.inspect(x_train, 'MNIST_training')
        datalib.inspect(x_test, 'MNIST_testing')

    else:
        print('Loading data...')
        # proceed with custom data induction
    #endregion

    autoencoder, encoder, decoder = build_autoencoder(x_train.shape[1], x_train.shape[2])
    autoencoder.compile(optimizer=keras.optimizers.Adam(lr=learn_rate), loss='mse', metrics=['mae'])

    autoencoder.fit(
        x=x_train,
        y=x_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=None,         # modelib.callbacks(log_dir, model_dir)
        validation_data=(x_test, x_test),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None)

    print('Training successful!')

    # Viewer
    fig, axs = pyplot.subplots(3,8)

    for i in range(len(axs[1])):
        j = random.randint(0,x_test.shape[0])       # used to sample random from dataset

        # Plot original image
        axs[0,i].imshow(x_test[j], cmap='gray', interpolation=None)

        # Plot encoding
        output_encoded = encoder.predict(np.expand_dims(x_test[j], axis=0), batch_size=None, verbose=0, steps=None, callbacks=None)
        output_encoded = np.squeeze(output_encoded, axis=0)

        if int(math.sqrt(output_encoded.size) + 0.5) ** 2 == output_encoded.size:         # check if vector is perfect square and can be displayed in 2D
            output_encoded_reshaped = np.reshape(output_encoded, (int(math.sqrt(output_encoded.size)) , int(math.sqrt(output_encoded.size) )))       # reshape vector to perfect square
            axs[1,i].imshow(output_encoded_reshaped, cmap='gray', interpolation=None)
        else:
            axs[1,i].imshow(np.expand_dims(output_encoded, axis=0), cmap='gray', interpolation=None)
        
        # Plot decoding
        output_decoded = decoder.predict(np.expand_dims(output_encoded, axis=0), batch_size=None, verbose=0, steps=None, callbacks=None)
        output_decoded = np.squeeze(output_decoded, axis=0)
        axs[2,i].imshow(output_decoded, cmap='gray', interpolation=None)


    # fig.tight_layout()
    fig.canvas.set_window_title('Autoencoder Output')
    
    pyplot.show()

    # query = input()
    # if query == 'y':
    #     print('Saving model to disk...')
    #     # Save
    #     print('Success!')
    # else:
    #     print('Sending model to android hell...')

    print('Saving model to disk...')
    autoencoder.save(model_dir + os.path.splitext(os.path.basename(__file__))[0] + '_' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '.h5')



    print('Debug:\n$tensorboard --logdir=logs/')