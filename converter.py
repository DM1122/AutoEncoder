import datetime as dt
import math
import matplotlib
from matplotlib import pyplot
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow import keras

import libs
from libs import datalib, imagelib, modelib


input_dir = './data/load/'
model_dir =  './models/load/'
output_dir = './models/load/output/'

input_size = 128


if __name__ == '__main__':
    if os.listdir(output_dir) is not None:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)


    mode = input('Converter mode encode(1) decode(2) autoencoder(3): ')

    if mode == '1':
        model = keras.models.load_model(model_dir + 'encoder.h5')
        model.compile(optimizer=keras.optimizers.Adam(lr=0), loss='mse')
        print('Encoder model compiled successfully!')

        print('Loading input...')
        data = np.concatenate([imagelib.load_img(path=input_dir+i, size=input_size) for i in os.listdir(input_dir)])
        datalib.inspect(data)
        
        print('Running prediction...')
        output = model.predict(data, batch_size=None, verbose=1, steps=None, callbacks=None)

        for i in range(output.shape[0]):
            output_converted = np.expand_dims(output[i],axis=0)
            output_converted = modelib.square_encoding(output_converted)

            key = np.amax(output_converted)        # get max pixel; will be used to uncompress features from image
            output_converted = output_converted / key * 255
            
            imagelib.save_img(array=output_converted, path=output_dir+'{}.png'.format(i))
            k = open(output_dir+'{}.txt'.format(i), 'w')
            k.write(str(key))
            k.close()
        
        print('Output(s) saved to disk!')



    elif mode == '2':
        model = keras.models.load_model(model_dir + 'decoder.h5')
    
    elif mode == '3':
        model = keras.models.load_model(model_dir + 'autoencoder.h5')

    else:
        raise ValueError('Huh?')
    



