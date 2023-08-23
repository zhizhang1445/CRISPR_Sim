import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
import scipy
from joblib import Parallel, delayed
import multiprocessing
from scipy import sparse
from copy import deepcopy
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


def main(params, sim_params):
    np.random.seed(sim_params['seed'])
    foldername = sim_params["foldername"]

    if sim_params["continue"]:
        params, sim_params = read_json(foldername)
        kernel_quarter = init_quarter_kernel(params, sim_params)
        kernel_exp = init_quarter_kernel(params, sim_params, type="Boltzmann")
        t, n, nh = load_last_output(foldername)

    else:
        params, sim_params = init_cond(params, sim_params)
        try:
            write2json(foldername, params, sim_params)
        except FileNotFoundError:
            os.mkdir(foldername)
            write2json(foldername, params, sim_params)

        n = init_guassian(params["N"], sim_params, "n")
        nh = init_exptail(params["Nh"], params, sim_params, "nh")
        kernel_quarter = init_quarter_kernel(params, sim_params)
        kernel_exp = init_quarter_kernel(params, sim_params, type="Boltzmann")
        t = 0

    while( t < sim_params["tf"]):

        if t%sim_params["t_snapshot"] == 0:
            sparse.save_npz(foldername+f"/sp_frame_n{t}",n.tocoo())
            sparse.save_npz(foldername+f"/sp_frame_nh{t}",nh.tocoo())

        p = elementwise_coverage(nh, n, kernel_quarter, params, sim_params)
        f = fitness_spacers(n, nh, p, params, sim_params) #f is now a masked array (where mask is where eff_R0 = 0)
        f = norm_fitness(f, n, params, sim_params) #renormalize f
        n = virus_growth(n, f, params, sim_params) #update

        n = mutation(n, params, sim_params)
        nh_gain = immunity_gain_from_kernel(nh, n, kernel_exp, params, sim_params) #update nh
        nh = immunity_loss_uniform(nh_gain, n, params, sim_params)
        t += sim_params["dt"]
    return 1

if __name__ == '__main__':

    params = { #parameters relevant for the equations
        "Nh":             1E7,
        "N0":             1E7, #This Will be updated by self-consitent solution
        "R0":              20, 
        "M":                1, #Also L, total number of spacers
        "mu":            0.01, #mutation rate
        "gamma_shape":     20, 
        "Np":               0, #Number of Cas Protein
        "dc":               3, #Required number of complexes to activate defence
        "h":                4, #coordination coeff
        "r":             2000, #cross-reactivity kernel
        "beta":         0.000,
    }
    sim_params = { #parameters relevant for the simulation (including Inital Valuess)
        "continue":                 False, #DO NOT CREATE ARBITRARY FOLDERS ONLY FOR TESTS
        "xdomain":                   1000,
        "dx":                           1,
        "tf":                        2000,
        "dt":                           1,
        "initial_mean_n":           [0,0],
        "initial_mean_nh":          [0,0],
        "conv_size":                 4000,
        "num_threads":                 32,
        "t_snapshot":                  10,
        "foldername":            "../Data",
        "seed":                         0,
    }

    if len(sys.argv) > 1:
        sim_params["seed"] = int(sys.argv[1])
        n_seeds = int(sys.argv[1])

    if len(sys.argv) > 2:
        sim_params["num_threads"] = int(sys.argv[2])

    
    # for beta in [-0.01, -0.001, 0, 0.001, 0.01]:
    #     main(params, sim_params)
    # i = 0
    # foldername = sim_params["foldername"] + f"/test{i}"
    # while os.path.exists(foldername):
    #     i += 1
    #     foldername = sim_params["foldername"] + f"/test{i}"


    params_list = []
    sim_params_list = []

    for beta in [-0.1, -0.01, -0.001, 0, 0.001, 0.01]: 
        num_cores = multiprocessing.cpu_count()

        for seed in range(n_seeds):

            params["beta"] = beta

            sim_params["num_threads"] = 2
            sim_params["seed"] = seed
            sim_params["foldername"] = "../Data_Parallel" + f"/beta{beta}_seed{seed}"

            if not os.path.exists(sim_params["foldername"]):
                os.mkdir(sim_params["foldername"])
            params_list.append(deepcopy(params))
            sim_params_list.append(deepcopy(sim_params))

    results = Parallel(n_jobs=len(params_list))(delayed(main)
            (params, sim_params) for params, sim_params in zip(params_list, sim_params_list))
