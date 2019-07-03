
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

import_path = './images/'
data_dir = './data/'

model_dir = './models/'

encoder_layers = 3
encoder_bottleneck_size = 32
encoder_layers_ratio = 2

batch_size = 256
epochs = 1
learn_rate = 0.01



if __name__ == '__main__':

    # Import data
    if not os.listdir(data_dir):
        print('No data located. Importing toy dataset...')

        x_train, _, x_test, _ = datalib.load_toy('A')

        datalib.inspect(x_train, 'MNIST_training')
        datalib.inspect(x_test, 'MNIST_testing')

    else:
        print('Loading data...')
        # proceed with custom data induction


    model = keras.Sequential()

    model.add(keras.layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2])))

    # Encoder layers
    for l in range(encoder_layers-1):
        model.add(keras.layers.Dense(
            units= encoder_bottleneck_size * encoder_layers_ratio**(encoder_layers - l),
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

    # Bottleneck
    model.add(keras.layers.Dense(
        units=encoder_bottleneck_size,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Bottleneck'))

    # Decoder layers
    for l in range(encoder_layers-1):
        model.add(keras.layers.Dense(
            units= encoder_bottleneck_size * encoder_layers_ratio**(l+1),
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
    
    # Output layer
    model.add(keras.layers.Dense(
        units=x_train[0].size,
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

    model.add(keras.layers.Reshape((x_train.shape[1], x_train.shape[2])))

    model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate), loss='mse', metrics=['mae'])


    # log_dir = './logs/' + os.path.basename(__file__) + '/{0}/'.format(dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
    # model_dir = './models/' + os.path.basename(__file__) + '/model.keras'

    model.fit(
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

    fig, axs = pyplot.subplots(3,8)


    # Viewer
    for i in range(len(axs[1])):
        
        model_encoder, model_decoder = modelib.split_autoencoder(model)

        j = random.randint(0,x_test.shape[0])       # used to sample random from dataset
        sample = x_test[j]

        # Plot original image
        axs[0,i].imshow(sample, cmap='gray', interpolation=None)

        # Plot encoding
        output_enc = model_encoder.predict(np.expand_dims(sample, axis=0), batch_size=None, verbose=0, steps=None, callbacks=None)

        if int(math.sqrt(output_enc.shape[1]) + 0.5) ** 2 == output_enc.shape[1]:         # check if vector is perfect square and can be displayed in 2D
            output_enc_reshaped = np.reshape(output_enc, (int(math.sqrt(output_enc.shape[1])) , int(math.sqrt(output_enc.shape[1]) )))       # reshape vector to perfect square
            axs[1,i].imshow(output_enc_reshaped, cmap='gray', interpolation=None)
        else:
            axs[1,i].imshow(output_enc, cmap='gray', interpolation=None)
        
        # Plot decoding
        output_dec = model_decoder.predict(np.expand_dims(output_enc, axis=0), batch_size=None, verbose=0, steps=None, callbacks=None)
        output_dec = np.squeeze(output, axis=0)
        axs[2,i].imshow(output_dec, cmap='gray', interpolation=None)


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
    model.save(model_dir + os.path.splitext(os.path.basename(__file__))[0] + '_' + dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '.h5')



    print('Debug:\n$tensorboard --logdir=logs/')