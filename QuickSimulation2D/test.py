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
    "Nh":            1000,
    "N0":             100,
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
    "xdomain":                  100,
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

p = coverage_parrallel_convolution(nh, kernel, params, sim_params)
# num_cores = sim_params["num_threads"]
# input_data = nh/params["Nh"]

# def convolve_subset(input_data_subset):
#     if np.sum(input_data_subset) == 0:
#         return input_data_subset
#     else:
#         return scipy.signal.convolve2d(input_data_subset, kernel, mode='same')

# input_data_subsets = square_split(input_data, num_cores)
# print(input_data_subsets)
# # plt.figure()
# plt.title("before")
# plt.imshow(nh)

