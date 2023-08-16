import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import scipy
from joblib import Parallel, delayed
from scipy import sparse
import json
import os
import sys

sys.path.insert(0, "../Scripts")
from initMethods import *
from coverage import *
from altImmunity import *
from immunity import *
from fitness import *
from mutation import *
from supMethods import *

params = { #parameters relevant for the equations
    "Nh":             1E8,
    "N0":             1E7, #This Will be updated by self-consitent solution
    "R0":              20, 
    "M":                1, #Also L, total number of spacers
    "mu":            0.01, #mutation rate
    "gamma_shape":     20, 
    "Np":               0, #Number of Cas Protein
    "dc":               3, #Required number of complexes to activate defence
    "h":                4, #coordination coeff
    "r":             2000, #cross-reactivity kernel
    "beta":        -0.001,
}
sim_params = { #parameters relevant for the simulation (including Inital Valuess)
    "xdomain":                   1000,
    "dx":                           1,
    "tf":                        2000,
    "dt":                           1,
    "initial_mean_n":           [0,0],
    "initial_mean_nh":          [0,0],
    "conv_size":                 4000,
    "num_threads":                 32,
    "tail_axis":               [1, 1],
    "t_snapshot":                  10,
}

params, sim_params = init_cond(params, sim_params)

i = 7
foldername = f"../Data/test{i}"
while os.path.exists(foldername):
    i += 1
    foldername = f"../Data/test{i}"

try:
    write2json(foldername, params, sim_params)
except FileNotFoundError:
    os.mkdir(foldername)
    write2json(foldername, params, sim_params)



n = init_guassian(params["N"], sim_params, "n")
nh = init_exptail(params["Nh"], params, sim_params, "nh")
kernel_quarter = init_quarter_kernel(params, sim_params)
kernel_exp = init_quarter_kernel(params, sim_params, type="Boltzmann")

for t in range(sim_params["tf"]):

    if t%sim_params["t_snapshot"] == 0:
        sparse.save_npz(foldername+f"/sp_frame_n{i}",n.tocoo())
        sparse.save_npz(foldername+f"/sp_frame_nh{i}",nh.tocoo())

    p = elementwise_coverage(nh, n, kernel_quarter, params, sim_params)
    f = fitness_spacers(n, nh, p, params, sim_params) #f is now a masked array (where mask is where eff_R0 = 0)
    f = norm_fitness(f, n, params, sim_params) #renormalize f
    n = virus_growth(n, f, params, sim_params) #update

    n = mutation(n, params, sim_params)
    nh_gain = immunity_gain_from_kernel(nh, n, kernel_exp, params, sim_params) #update nh
    nh = immunity_loss_uniform(nh_gain, n, params, sim_params)