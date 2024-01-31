import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from copy import deepcopy
import random
import os
import sys

from antigenicWaveSimulationMethods import main as coEvoSimulation

if __name__ == '__main__':

    params = { #parameters relevant for the equations
        "Nh":                     1E6,
        "N0":                     1E7, #This Will be updated by self-consitent solution
        "R0":                      20, 
        "M":                        1, #Also L, total number of spacers
        "mu":                     0.1, #mutation rate
        "gamma_shape":             20, 
        "Np":                       0, #Number of Cas Protein
        "dc":                       3, #Required number of complexes to activate defence
        "h":                        4, #coordination coeff
        "r":                     2000, #cross-reactivity kernel
        "beta":                     0,
        "rate_HGT":                 0,
        "HGT_bonus_acq_ratio":      1,
        "rate_recovery":          0.1,
        "HGT_type":                 1,
    }
    sim_params = { #parameters relevant for the simulation (including Inital Valuess)
        "continue":                 False, #DO NOT CREATE ARBITRARY FOLDERS ONLY FOR TESTS
        "xdomain":                   1000,
        "dx":                           1,
        "tf":                       10000,
        "dt":                           1,
        "dt_exact_fitness":             1,
        "dt_snapshot":                 25,
        "initial_mean_n":           [0,0],
        "initial_mean_nh":          [0,0],
        "conv_size":                 4000,
        "num_threads":                  1,
        "foldername":   "../Data_HGT_rate_type_2",
        "seed":                         0,
    }
    continue_flag = False
    num_threads_set = False
    n_seeds = 1
    foldername = sim_params["foldername"]
    #the call is python3 antigenicWaveSimulation.py <num_cores> <num_seeds> <0 for restart or 1 for continue>

    if len(sys.argv) > 1:
        sim_params["num_threads"] = int(sys.argv[2])
        num_threads_set = True

    if len(sys.argv) > 2:
        n_seeds = int(sys.argv[2])

    if len(sys.argv) > 3:
        if int(sys.argv[3]) == 1:
            if os.path.isdir(foldername):
                continue_flag = True
                print("Continuing where we left off")
                sim_params["continue"] = True

            else:
                os.mkdir(foldername)
                print("Created new folder: ", foldername)
                continue_flag = False
                sim_params["continue"] = False
    
        elif int(sys.argv[3]) == 0:
            continue_flag = False
            sim_params["continue"] = False
        else:
            ValueError("Error in arguments")

    if sim_params["num_threads"] == 0 or n_seeds == 0:
        raise ValueError("something wrong with the num threads or num seeds")

    seed_list = random.sample(range(0, 255), n_seeds)
    params_list = []
    sim_params_list = []
    # list_to_sweep1 = [-1, -1.25, -1.5, -1.75, -2, -2.25, -2.5, -2.75, -3, -3.25, -3.5, -3.75]
    # list_to_sweep2 = [0, -0.01, -0.02, -0.05, 0.01, 0.05, 0.1, 0.5]
    list_to_sweep = [1/10, 1/20, 1/40, 1/80, 1/160]

    num_cores = multiprocessing.cpu_count()
    if not num_threads_set:
        best_ratio = int(num_cores // (len(list_to_sweep)*n_seeds))
        num_cores_per_run = best_ratio if best_ratio >= 1 else 1
        sim_params["num_threads"] = num_cores_per_run
        print(f"Each Run is done with {num_cores_per_run} cores")

    for i, rate_HGT in enumerate(list_to_sweep): 

        for seed_num, seed in enumerate(seed_list):
            params["rate_HGT"] = rate_HGT
            sim_params["seed"] = seed
            sim_params["foldername"] = foldername + f"/HGT_rate_{rate_HGT}_seed{seed_num}"

            if not os.path.exists(sim_params["foldername"]):
                os.mkdir(sim_params["foldername"])

            params_list.append(deepcopy(params))
            sim_params_list.append(deepcopy(sim_params))

    try:
        results = Parallel(n_jobs=len(params_list))(delayed(coEvoSimulation)
            (params, sim_params) for params, sim_params in zip(params_list, sim_params_list))
    except KeyboardInterrupt:
        print("Hopefully Every Closed")
