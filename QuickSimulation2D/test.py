import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed
import scipy

from methods_parallel import *
from initMethods import *

params = { #parameters relevant for the equations
    "Nh":            1000,
    "N0":             100,
    "R0":               5,
    "M":                1, #Also L, total number of spacers
    "D":                3, #Unused
    "mu":               2, #mutation rate
    "v0":               2,
    "gamma_shape":     200, 
    "Np":               0, #Number of Cas Protein
    "dc":               3, #Required number of complexes to activate defence
    "h":                4, #coordination coeff
    "r":             1000, #cross-reactivity kernel
    "rho":           5e-5, #spacer sharing coeff
}
sim_params = { #parameters relevant for the simulation (including Inital Valuess)
    "xdomain":                  100,
    "dx":                         1,
    "t0":                         0, 
    "tf":                       100,
    "dt":                         1,
    "initial_mean":           [0,0],
    "initial_var":                5,
    "n_step_prior":               5,
    "conv_size":               1000,
    "num_threads":                4,
    "tail_axis":            [1, -1],
}

n = init_guassian_parallel(params["N0"], sim_params)
nh = init_exptail_parrallel(params["Nh"], params, sim_params)
kernel = init_kernel(params, sim_params)

# p = coverage_parrallel_convolution(nh, kernel, params, sim_params)
x_ind, y_ind = np.nonzero(nh)
# print(non_zero_ind)
x_ind_subsets = np.array_split(x_ind, 4)
y_ind_subsets = np.array_split(y_ind, 4)
ksize = sim_params["conv_size"]

def conv_sparse_index(indexes):
    array = np.zeros(nh, dtype=float)
    for x_ind, y_ind in indexes:
        value = n[x_ind, y_ind]
        array[x_ind-ksize:x_ind+ksize, y_ind-ksize:y_ind+ksize] += kernel*value
    return array



