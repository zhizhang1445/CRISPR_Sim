import numpy as np
import scipy
import json
import os
import time
from functools import wraps
import matplotlib.colors as mcolors

def write2json(name, params, sim_params):
    with open(name + '/params.json', 'w') as fp:
        json.dump(params, fp)

    with open(name + '/sim_params.json', 'w') as fp:
        json.dump(sim_params, fp)

def read_json(foldername):
    params = {}
    sim_params = {}
    
    try:
        # Read params.json
        with open(foldername+'/params.json', 'r') as params_file:
            params = json.load(params_file)
    except FileNotFoundError:
        print("params.json not found")
    
    try:
        # Read sim_params.json
        with open(foldername+'/sim_params.json', 'r') as sim_params_file:
            sim_params = json.load(sim_params_file)
    except FileNotFoundError:
        print("sim_params.json not found")
    
    return params, sim_params

def load_last_output(foldername):
    files_in_directory = [f for f in os.listdir(foldername) if f.startswith('sp_frame_nh') and f.endswith('.npz')]

    if not files_in_directory:
        raise FileNotFoundError("No sp_frame_n*.npz files found in the current directory.")

    highest_numeric_value = float('-inf')

    for filename in files_in_directory:
        numeric_value = int(filename.split('sp_frame_nh')[1].split('.npz')[0]) #USE Nh instead of N cuz N will give you h0
        if numeric_value > highest_numeric_value:
            highest_numeric_value = numeric_value
    
    n = scipy.sparse.load_npz(foldername + f"/sp_frame_n{highest_numeric_value}.npz").todok()
    nh = scipy.sparse.load_npz(foldername + f"/sp_frame_nh{highest_numeric_value}.npz").todok()
    return highest_numeric_value, n, nh

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

def extract_xy(list) : #if you have a list [[x0,y0], [x1, y1], ...] and you want [x0, x1, ...] and [y0, y1, ...]
    try:
        if len(list) == 0:
            x_val = y_val = []

        elif len(list) == 1:
            A = np.array(list).squeeze()
            x_val = A[0]
            y_val = A[1]
            
        else:
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

