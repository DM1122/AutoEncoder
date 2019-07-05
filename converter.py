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


input_size = 64


def generate_repo():
    name = os.path.splitext(os.path.basename(__file__))[0]

    repo = {
        'encode':'./'+name+'/encode/',
        'decode':'./'+name+'/decode/',
        'model':'./'+name+'/model/',
        'output':'./'+name+'/output/'
    }

    if not os.path.isdir('./'+name+'/'):
        print('No repo located. Instantiating one now...')

        for dir in repo:
            os.makedirs(repo.get(dir))
    
    return repo


if __name__ == '__main__':
    repo = generate_repo()

    mode = input('Converter Modes:\n(1) Encode\n(2) Decode\n(3) Autoencoder\n')

    # Encoder mode
    if mode == '1':
        repo['model'] = repo['model']+os.listdir(repo['model'])[0]+'/encoder.h5'
        model = keras.models.load_model(repo['model'])
        model.compile(optimizer=keras.optimizers.Adam(lr=0), loss='mse')
        print('Encoder model compiled successfully!')

        print('Loading input...')
        data = np.concatenate([imagelib.load_img(path=repo['encode']+i, size=input_size) for i in os.listdir(repo['encode'])])
        datalib.inspect(data)
        
        print('Running encoder...')
        output = model.predict(data, batch_size=None, verbose=1, steps=None, callbacks=None)

        repo['output'] = repo['output']+'encoded_'+dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'/'
        os.makedirs(repo['output'])        
        for i in range(output.shape[0]):
            out = np.expand_dims(output[i],axis=0)
            out = modelib.square_encoding(out)
            datalib.inspect(np.expand_dims(out,axis=0))

            key = np.amax(out)        # get max pixel; will be used to uncompress features from image
            out = out / key * 255
            
            imagelib.save_img(array=out, path=repo['output']+'{}.png'.format(i))

            k = open(repo['output']+'{}.txt'.format(i), 'w')       # write key to disk
            k.write(str(key))
            k.close()
        
        print('Output(s) saved to disk!')


    # Decoder mode
    elif mode == '2':
        repo['model'] = repo['model']+os.listdir(repo['model'])[0]+'/decoder.h5'
        model = keras.models.load_model(repo['model'])
        model.compile(optimizer=keras.optimizers.Adam(lr=0), loss='mse')
        print('Decoder model compiled successfully!')
    
        print('Loading input...')
        repo['decode'] = repo['decode']+os.listdir(repo['decode'])[0]
        data = imagelib.load_img(path=repo['decode'])
        data, _, _ = imagelib.RGBsplitter(data[0])
        data = np.expand_dims(data,axis=0)
        datalib.inspect(data)

        key = input('Please enter decoder key: ')
        if key == '': key = 1
        else:
            key = float(key)
        
        data = data * key

        print('Unpacking vector...')
        
        if data.ndim == 3:
            data = modelib.flatten(data)

        print('Running decoder...')
        output = model.predict(data, batch_size=None, verbose=1, steps=None, callbacks=None)
        datalib.inspect(output)
        output = (np.squeeze(output,axis=0) * 255).astype('uint8')
        
        repo['output'] = repo['output']+'decoded_'+dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'/'
        os.makedirs(repo['output'])
        imagelib.save_img(array=output, path=repo['output']+'decoded.png')
        print('Output saved to disk!')

    # Autoencoder mode
    elif mode == '3':
        model = keras.models.load_model(model_dir + 'autoencoder.h5')

    else:
        raise ValueError('Huh?')
    



