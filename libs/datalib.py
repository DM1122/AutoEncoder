import matplotlib
from matplotlib import pyplot
import numpy as np


def inspect(data, name='data'):
    print('Inspecting ' + name + '...')
    
    if data.ndim == 2:
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(data, cmap='gray', interpolation=None)

        fig.tight_layout()
        fig.canvas.set_window_title('Data Inspection') 
    
    elif data.ndim == 3 and len(data[2]) == 3:
        fig, axs = pyplot.subplots(2,2)

        data_r, data_g, data_b = imagelib.RGBsplitter(data)

        axs[0,0].imshow(data, cmap=None, interpolation=None)
        axs[0,1].imshow(data_r, cmap='Reds', interpolation=None)
        axs[1,0].imshow(data_g, cmap='Greens', interpolation=None)
        axs[1,1].imshow(data_b, cmap='Blues', interpolation=None)

        fig.tight_layout()
        fig.canvas.set_window_title('Data Inspection') 

    else:
        print('No preview available')
    
    print('Shape: ', data.shape)
    print('Dims: ', data.ndim)
    print('Size: ', data.size)
    print('Type: ', data.dtype)
    print('Min/Max: ', np.amin(data), '/', np.amax(data))
    # print('Head: ', data[:1])
    # print('Tail: ', data[-1:])

    pyplot.show()