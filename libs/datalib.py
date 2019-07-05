import matplotlib
from matplotlib import pyplot
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

import libs
from libs import imagelib


def load_toy(sel='A'):
    '''
    Imports toy datasets

    Args:
        sel (str): dataset selection. Either A (MNIST) or B (MNIST_fashion)
    '''

    if sel == 'A':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    elif sel == 'B':
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError('Toy dataset selection nonexistent')

    # Preprocessing
    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test


def inspect(data, name='data'):
    '''
    Returns data statistics from a random selection.
    Data must be fed with batch dimension [batch, height, width, channel].

    Args:
        data (arr): numpy array containing data
        name (str): optional differentiator
    '''
    print('Inspecting ' + name + '...')

    i = random.randint(0,data.shape[0]-1)       # iterator used to sample random from dataset along batch axis
    
    if data.ndim == 3:      # grayscale images
        fig, ax = pyplot.subplots(1,1)
        fig.canvas.set_window_title('Data Inspection') 

        ax.imshow(data[i], cmap='gray', interpolation=None)

        fig.tight_layout()
        
    
    elif data.ndim == 4 and data.shape[3] == 3:      # RGB images
        fig, axs = pyplot.subplots(2,2)
        fig.canvas.set_window_title('Data Inspection') 

        data_r, data_g, data_b = imagelib.RGBsplitter(data[i])

        axs[0,0].imshow(data[i], cmap=None, interpolation=None)
        axs[0,1].imshow(data_r, cmap='Reds', interpolation=None)
        axs[1,0].imshow(data_g, cmap='Greens', interpolation=None)
        axs[1,1].imshow(data_b, cmap='Blues', interpolation=None)

        fig.tight_layout()

    else:
        print('No preview available')
    
    print('Shape: ', data.shape)
    print('Dims: ', data.ndim)
    print('Size: ', data.size)
    print('Type: ', data.dtype)
    print('Min/Max: ', np.amin(data), '/', np.amax(data))

    pyplot.show()


def split(data, split):
    a = data[0:int(data.shape[0] * split)]
    b = data[int(data.shape[0] * split):]

    return a, b


if __name__ == '__main__':
    print('Nothing to see here...')