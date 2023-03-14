import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import matplotlib.animation as animation
import scipy
import json

def makeGif(frame_stack, name):
    fig = plt.figure()

    animation_frames = []
    for frame in frame_stack:
        frame = np.squeeze(frame)
        animation_frames.append([plt.plot(frame, animated=True)])

    ani = animation.ArtistAnimation(
        fig, animation_frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(name + "1D.gif")
    plt.close()
    return 1

def write2json(name, params, sim_params):
    with open(name + 'params1D.json', 'w') as fp:
        json.dump(params, fp)

    with open(name + 'sim_params1D.json', 'w') as fp:
        json.dump(sim_params, fp)