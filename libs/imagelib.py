import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import PIL


def load_img(path, scale=1, size=None):
    img = PIL.Image.open(path)

    # Resize
    if size is not None:
        scale = size / max(img.size)

    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), PIL.Image.ANTIALIAS)

    # Convert image to array
    img_array = np.array(img)       # [height, width, channel]

    img_array = np.expand_dims(img_array, axis=0)       # brodcast image array with batch dimension [batch, height, width, channel]

    img_array = img_array / 255

    return img_array


def save_img(array, path):
    img = PIL.Image.fromarray(array)
    img.convert('RGB').save(path)


def unload_img_vgg19(img):
    print('Unloading image')
    img = np.squeeze(img, axis=0)       # squeeze batch dimension

    assert img.ndim == 3

    # Reverse channels normalized by mean
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]

    img = np.clip(img, 0, 255).astype('uint8')

    return img


def RGBsplitter(img):
    '''
    Returns individual RGB channel arrays from image 

    Args:
        img: image array of shape [width, height, channel]
    '''
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    return r, g, b


if __name__ == '__main__':
    print('Nothing here mate.')

