import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from scipy import sparse
from copy import deepcopy
import time
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


        nh_total = params["Nh"]
        n_total = params["N"]
        uc = params["uc"]
        sigma = params["sigma"]

        with open(foldername+'/runtime_stats.txt','a') as file:
            file.write(f't: {t}| Restarted  | Phage Population: {n_total:.4f}| Spacer Population: {nh_total:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}\n')

    else:
        params, sim_params = init_cond(params, sim_params)
        try:
            write2json(foldername, params, sim_params)
        except FileNotFoundError:
            os.mkdir(foldername)
            write2json(foldername, params, sim_params)

        st = time.time()
        n = init_guassian(params["N"], sim_params, "n")
        nh = init_exptail(params["Nh"], params, sim_params, "nh")
        kernel_quarter = init_quarter_kernel(params, sim_params)
        kernel_exp = init_quarter_kernel(params, sim_params, type="Boltzmann")
        ed = time.time()
            
        t = 0
        nh_total = params["Nh"]
        n_total = params["N"]
        uc = params["uc"]
        sigma = params["sigma"]
        with open(foldername+'/runtime_stats.txt','w') as file:
            file.write(f't: {t}| init_functions: {time_conv(ed-st)}| Phage Population: {n_total:.4f}| Spacer Population: {nh_total:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}\n')

    while(t < sim_params["tf"]):

        if t%sim_params["t_snapshot"] == 0:
            sparse.save_npz(foldername+f"/sp_frame_n{t}",n.tocoo())
            sparse.save_npz(foldername+f"/sp_frame_nh{t}",nh.tocoo())

        st1 = time.time()
        p = elementwise_coverage(nh, n, kernel_quarter, params, sim_params)

        st2 = time.time()
        f = fitness_spacers(n, nh, p, params, sim_params) #f is now a masked array (where mask is where eff_R0 = 0)
        f = norm_fitness(f, n, params, sim_params) #renormalize f
        n = virus_growth(n, f, params, sim_params) #update

        st3 = time.time()

        n = mutation(n, params, sim_params)

        st4 = time.time()
        nh = immunity_update(nh, n, params, sim_params)
        # nh_gain = immunity_gain_from_kernel(nh, n, kernel_exp, params, sim_params) #update nh

        # nh = immunity_loss_uniform(nh_gain, n, params, sim_params)
        ed = time.time()

        with open(foldername+'/runtime_stats.txt','a') as file:
            outstring = f"t: {t}| Coverage: {time_conv(st2-st1)}| Growth: {time_conv(st3-st2)}| Mutation: {time_conv(st4-st3)}| Immunity: {time_conv(ed-st4)} \n"
            file.write(outstring)

        t += sim_params["dt"]
    return 1

if __name__ == '__main__':

    params = { #parameters relevant for the equations
        "Nh":             1E6,
        "N0":             1E7, #This Will be updated by self-consitent solution
        "R0":              20, 
        "M":                1, #Also L, total number of spacers
        "mu":             0.1, #mutation rate
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
        "tf":                       10000,
        "dt":                           1,
        "dt_exact_fitness":             0,
        "initial_mean_n":           [0,0],
        "initial_mean_nh":          [0,0],
        "conv_size":                 4000,
        "num_threads":                  1,
        "t_snapshot":                  10,
        "foldername":"../Data_temp_name5",
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
                sim_params['continue'] = False
    
        elif int(sys.argv[3]) == 0:
            continue_flag = False
            sim_params["continue"] = False
        else:
            ValueError("Error in arguments")

    params_list = []
    sim_params_list = []
    list_to_sweep1 = [-1, -1.25, -1.5, -1.75, -2, -2.25, -2.5, -2.75, -3, -3.25, -3.5, -3.75]
    list_to_sweep2 = [0, -0.01, -0.02, -0.05]

    num_cores = multiprocessing.cpu_count()
    if not num_threads_set:
        best_ratio = int(num_cores // (len(list_to_sweep2)*n_seeds))
        num_cores_per_run = best_ratio if best_ratio >= 1 else 1
        sim_params["num_threads"] = num_cores_per_run
        print(f"Each Run is done with {num_cores_per_run} cores")

    for i, beta in enumerate(list_to_sweep2): 

        for seed in range(n_seeds):
            params["beta"] = beta
            sim_params["seed"] = seed
            sim_params["foldername"] = foldername + f"/beta{beta}_seed{seed}"

            if not os.path.exists(sim_params["foldername"]):
                os.mkdir(sim_params["foldername"])
                # print("check for problem with file creation")
            params_list.append(deepcopy(params))
            sim_params_list.append(deepcopy(sim_params))

    results = Parallel(n_jobs=len(params_list))(delayed(main)
            (params, sim_params) for params, sim_params in zip(params_list, sim_params_list))
