import matplotlib
from matplotlib import pyplot
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras

import libs
from libs import imagelib

import_path = './images/'

def inspect_data(data):
    print('Inspecting data...')
    
    if data.ndim == 2:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(data, cmap='gray')

        fig.tight_layout()
    
    elif data.ndim == 3:
        
        data_r, data_g, data_b = imagelib.RGBsplitter(data)


        fig, axs = pyplot.subplots(2,2)

        axs[0,0].imshow(data)
        axs[0,1].imshow(data_r, cmap='Reds')
        axs[1,0].imshow(data_g, cmap='Greens')
        axs[1,1].imshow(data_b, cmap='Blues')


        # gridspec_kw={'width_ratios': [3, 1]}
        fig.tight_layout()

    else:
        print('No preview available')
    
    print('Shape: ', data.shape)
    print('Dims: ', data.ndim)
    print('Size: ', data.size)
    print('Head: ', data[:1])
    print('Tail: ', data[-1:])

    fig.canvas.set_window_title('Data Inspection') 
    pyplot.show()


if __name__ == '__main__':
    print('Importing MNIST dataset')
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    inspect_data(x_train[0])

    # data = imagelib.load_img(import_path)
    # inspect_data(data[0])


