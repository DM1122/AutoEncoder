
import datetime as dt
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

encoder_layers = 6
encoder_bottleneck_size = 10
encoder_layers_ratio = 2

batch_size = 256
epochs = 25
learn_rate = 0.01



if __name__ == '__main__':
    print('Importing toy dataset')
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    datalib.inspect(x_train, 'MNIST_training')
    datalib.inspect(x_test, 'MNIST_testing')

    # Scaling
    x_train = x_train / 255
    x_test = x_test / 255


    
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
        activation='relu',        # might need sigmoid!
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


    log_dir = './logs/' + os.path.basename(__file__) + '/{0}/'.format(dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
    model_dir = './models/' + os.path.basename(__file__) + '/model.keras'

    model.fit(
        x=x_train,
        y=x_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=None,         # modelib.callbacks(log_dir, model_dir)
        validation_split=0.0,
        validation_data=(x_test, x_test),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None)

    fig, axs = pyplot.subplots(3,8)

    for i in range(len(axs[1])):
        
        j = random.randint(0,x_test.shape[0])       # used to sample random from dataset
        input = np.expand_dims(x_test[j], axis=0)

        # Plot original image        
        axs[0,i].imshow(x_test[j], cmap='gray', interpolation=None)

        # Plot latent space vector
        model_latent = keras.Model(inputs=model.input, outputs=model.get_layer('Bottleneck').output)
        output_latent = model_latent.predict(input, batch_size=None, verbose=0, steps=None, callbacks=None)
        axs[1,i].imshow(output_latent, cmap='gray', interpolation=None)
        
        # Plot output
        input = np.expand_dims(x_test[j], axis=0)
        output = model.predict(input, batch_size=None, verbose=0, steps=None, callbacks=None)
        output = np.squeeze(output, axis=0)
        axs[2,i].imshow(output, cmap='gray', interpolation=None)


    # fig.tight_layout()
    fig.canvas.set_window_title('Autoencoder Output') 
    
    pyplot.show()

    print('Debug:\n$tensorboard --logdir=logs/')