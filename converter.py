import datetime as dt
import math
import matplotlib
from matplotlib import pyplot
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import libs
from libs import datalib, imagelib, modelib


input_dir = './data/load/'
model_dir =  './models/load/'
output_dir = './plots/'

input_size = 256


if __name__ == '__main__':
    mode = input('Converter mode encode (1) decode (2): ')

    if mode == '1':
        model = keras.models.load_model(model_dir + 'encoder.h5')
    elif mode == '2':
        model = keras.models.load_model(model_dir + 'decoder.h5')
    else:
        raise ValueError('Huh?')


    print('Loading data...')

    data = np.concatenate([imagelib.load_img(path=input_dir+i, size=inout_size) for i in os.listdir(input_dir)])
    datalib.inspect(data)



