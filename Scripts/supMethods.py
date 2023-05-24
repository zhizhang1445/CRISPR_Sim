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
import matplotlib.colors as mcolors
from mutation import calc_diff_const

def write2json(name, params, sim_params):
    with open(name + '/params.json', 'w') as fp:
        json.dump(params, fp)

    with open(name + '/sim_params.json', 'w') as fp:
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

def extract_xy(list):
    try:
        A = np.array(list).squeeze()
        x_val = A[:,0]
        y_val = A[:,1]
    except TypeError:
        x_val = np.zeros(len(list))
        y_val = np.zeros(len(list))
        for x, y, item in zip(x_val, y_val, list):
            x = item[0]
            y = item[1]

    return x_val, y_val

def calculate_velocity(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = 24*np.log(N*np.power(D*np.power(s,2), 1/3))
    v = np.power(s, 1/3)*np.power(D, 2/3)*np.power(common_log, 1/3)
    return v
