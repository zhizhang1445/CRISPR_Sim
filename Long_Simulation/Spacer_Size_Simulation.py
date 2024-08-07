import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from copy import deepcopy
import random
import os
import sys

from antigenicWaveSimulationMethods import main as coEvoSimulation
from antigenicWaveSimulationMethods import make_paramslists

if __name__ == '__main__':

    params = { #parameters relevant for the equations
        "Nh":                     1E2,
        "N0":                     1E2, #This Will be updated by self-consitent solution
        "R0":                      20, 
        "M":                      500, #Also L, total number of spacers
        "mu":                       1, #mutation rate
        "gamma_shape":             20, 
        "Np":                     100, #Number of Cas Protein
        "dc":                       3, #Required number of complexes to activate defence
        "h":                        4, #coordination coeff
        "r":                     2000, #cross-reactivity kernel
        "beta":                     0,
        "rate_HGT":                 0,
        "HGT_bonus_acq_ratio":      1,
        "rate_recovery":          0.1,
        "HGT_type":                 0,
    }
    sim_params = { #parameters relevant for the simulation (including Inital Valuess)
        "continue":                   False, #DO NOT CREATE ARBITRARY FOLDERS ONLY FOR TESTS
        "xdomain":                     5000,
        "dx":                             1,
        "tf":                          1000,
        "dt":                             1,
        "dt_exact_fitness":               1,
        "dt_snapshot":                    1,
        "initial_mean_n":             [0,0],
        "initial_mean_nh":            [0,0],
        "conv_size":                   4000,
        "num_threads":                    32,
        "foldername":  "../Data_Memory_Size_Large2",
        "seed":                            0,
        "hard_N0":                      True,
    }

    list_to_sweep = [int(i) for i in np.arange(2, 200)]
    params_list, sim_params_list = make_paramslists(params, sim_params, "M", list_to_sweep)

    try:
        for params, sim_params in zip(params_list, sim_params_list):
            sim_params["num_threads"] = 32
            results = coEvoSimulation(params, sim_params, normalize_f=True)
        # results = Parallel(n_jobs=len(params_list))(delayed(coEvoSimulation)
        #     (params, sim_params) for params, sim_params in zip(params_list, sim_params_list))
    except KeyboardInterrupt:
        print("Hopefully Every Closed")
