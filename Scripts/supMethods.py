import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import matplotlib.animation as animation
import scipy
import json
import time
from functools import wraps

def create_frame(foldername, i, margins = (-0.45, -0.45), name = "test_img"):
    n_i = scipy.sparse.load_npz(foldername+f"sp_frame_n{i}.npz")
    nh_i = scipy.sparse.load_npz(foldername+f"sp_frame_nh{i}.npz")

    fig = plt.figure()
    plt.contour(n_i.toarray(), cmap = "Reds")
    plt.contour(nh_i.toarray(), cmap = "Blues")
    plt.margins(margins[0], margins[1])
    
    plt.title(f"N and Nh distribution at timestep {i}")
    plt.savefig(f'./{name}/img_{i}.png', transparent = False,  
            facecolor = 'white')
    plt.close()

def write2json(name, params, sim_params):
    with open(name + 'params.json', 'w') as fp:
        json.dump(params, fp)

    with open(name + 'sim_params.json', 'w') as fp:
        json.dump(sim_params, fp)

def time_conv(st):
    return time.strftime("%H:%M:%S", time.gmtime(st))

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'{func.__name__} took {time_conv(total_time)}')
        return result
    return timeit_wrapper

def minmax_norm(array):
    max_val = array.max()
    min_val = array.min()
    res = (array - min_val)/np.ptp(array)
    return res

def makeGif(animation_frame_stack, name): #no longer used
    raise(NotImplementedError)
#     fig = plt.figure()

#     ani = animation.ArtistAnimation(
#         fig, animation_frame_stack, interval=50, blit=True, repeat_delay=1000)
#     ani.save(name + ".gif")
#     plt.close()
#     return 1