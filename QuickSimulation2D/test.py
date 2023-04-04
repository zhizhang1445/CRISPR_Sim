import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed
import scipy

from methods import *
from initMethods import *

params = { #parameters relevant for the equations
    "Nh":             20,
    "N0":             20,
    "R0":               5,
    "M":                1, #Also L, total number of spacers
    "D":                3, #Unused
    "mu":               2, #mutation rate
    "gamma_shape":     200, 
    "Np":               0, #Number of Cas Protein
    "dc":               3, #Required number of complexes to activate defence
    "h":                4, #coordination coeff
    "r":             1000, #cross-reactivity kernel
    "rho":           5e-5, #spacer sharing coeff
}
sim_params = { #parameters relevant for the simulation (including Inital Valuess)
    "xdomain":                    5,
    "dx":                         1,
    "t0":                         0, 
    "tf":                       100,
    "dt":                         1,
    "initial_mean":           [0,0],
    "initial_var":                5,
    "n_step_prior":               5,
    "conv_size":               1000,
    "num_threads":               2,
}



n = init_guassian(params["N0"], sim_params)
nh = init_guassian(params["Nh"], sim_params)
kernel = init_kernel(params, sim_params)



# plt.figure()
# plt.title("before")
# plt.imshow(nh)

Nh = params["Nh"]
N = np.sum(n)
num_threads = sim_params["num_threads"]
nh = nh + n
num_to_remove = np.sum(nh) - Nh

nonzero_indices = np.nonzero(nh)
nonzero_values = [nh[index] for index in zip(*nonzero_indices)]
index_nonzero_w_repeats = []
for value, index in zip(nonzero_values, zip(*nonzero_indices)):
    for i in range(int(value)):
        index_nonzero_w_repeats.append(index)

sample_flat_ind = np.random.choice(len(index_nonzero_w_repeats), num_to_remove,replace = False)

ind_per_thread_list = np.split(sample_flat_ind, num_threads)

def remove_points(array, flat_index):
    sample_ind = [index_nonzero_w_repeats[i] for i in flat_index]
    for x,y in sample_ind:
        array[x, y] -= 1

    return array

array = np.zeros(nh.shape)
results = Parallel(n_jobs=num_threads)(
    delayed(remove_points)(array, flat_index) for flat_index in ind_per_thread_list)
nh = nh + np.sum(results, axis=0)

if np.sum(nh) != Nh:
    raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh))
if np.min(nh) < 0:
    raise ValueError("bacteria population is negative")

