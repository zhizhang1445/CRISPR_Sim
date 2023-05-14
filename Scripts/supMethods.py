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

def makeGif(frame_stack, name):
    fig = plt.figure()

    animation_frames = []
    for frame in frame_stack:
        frame = np.squeeze(frame)
        animation_frames.append([plt.imshow(frame, animated=True)])

    ani = animation.ArtistAnimation(
        fig, animation_frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(name + ".gif")
    plt.close()
    return 1

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
