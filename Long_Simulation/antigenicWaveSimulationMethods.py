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
from formulas import compute_shift
from supMethods import *
from randomHGT import *


def main(params, sim_params) -> int :
    np.random.seed(sim_params['seed'])
    foldername = sim_params["foldername"]
    shift_vector = [0,0] #This is unused if dt_exact_fitness is zero

    if sim_params["continue"]:
        try:
            st1 = time.time()
            params, sim_params = read_json(foldername)
            kernel_conv = init_quarter_kernel(params, sim_params)
            kernel_immunity = init_quarter_kernel(params, sim_params, type="Boltzmann")
            t, n, nh = load_last_output(foldername)
            nh_total = params["Nh"]
            n_total = params["N"]
            uc = params["uc"]
            sigma = params["sigma"]
            M = params["M"]

            with open(foldername+'/runtime_stats.txt','a') as file:
                file.write(f't: {t}| Restarted  | Phage Population: {n_total:.4f}| Spacer Population: {nh_total:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}| M: {M:.4f} \n')

        except KeyError or (nh is None): #the folders were empty so better restart everything
            sim_params["continue"] = False


    if not sim_params["continue"]: #not an else statement because this also serves as error catch for the if statement
        params, sim_params = init_cond(params, sim_params)
        try:
            write2json(foldername, params, sim_params)
        except FileNotFoundError:
            os.mkdir(foldername)
            write2json(foldername, params, sim_params)

        st1 = time.time()
        n = init_guassian(params["N"], sim_params, "n")
        nh = init_exptail(params["Nh"]*params["M0"], params, sim_params, "nh")
        kernel_conv = init_quarter_kernel(params, sim_params)
        kernel_immunity = init_quarter_kernel(params, sim_params, type="Boltzmann")
        ed = time.time()
            
        t = 0
        nh_total = params["Nh"]
        n_total = params["N"]
        uc = params["uc"]
        sigma = params["sigma"]
        M0 = params["M0"]

        with open(foldername+'/runtime_stats.txt','w') as file:
            file.write(f't: {t}| init_functions: {time_conv(ed-st1)}| Phage Population: {n_total:.4f}| Spacer Population: {nh_total:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}| M0: {M:.4f} \n')

    try:
        while(t < sim_params["tf"]):

            if t%sim_params["dt_snapshot"] == 0:
                sparse.save_npz(foldername+f"/sp_frame_n{t}",n.tocoo())
                sparse.save_npz(foldername+f"/sp_frame_nh{t}",nh.tocoo())

            if t%sim_params["dt_exact_fitness"] == 0:
                str:float = time.time()
                p = elementwise_coverage(nh, n, kernel_conv, params, sim_params)
                st2 = time.time()
                f = fitness_spacers(n, nh, p, params, sim_params)
                sparse.save_npz(foldername+f"/sp_frame_f{t}", f.tocoo())
                f = norm_fitness(f, n, params, sim_params) #renormalize f

            else:
                st1 = time.time()
                raise NotImplementedError("dt_fast_fitness better be 1")
                f = fitness_spacers_fast(f, shift_vector, params)

                st2 = time.time()
                f = norm_fitness(f, n, params, sim_params)

            n = virus_growth(n, f, params, sim_params) #update
            
            if (np.sum(n) <= 0) or (np.sum(n) >= (1/2)*np.sum(nh)):
                with open(foldername+'/runtime_stats.txt','a') as file:
                    outstring = f"DEAD at: {t}| N: {np.sum(n)}| Coverage: {time_conv(st2-st1)}| Growth: {time_conv(st3-st2)}| Mutation: {time_conv(st4-st3)}| Immunity: {time_conv(ed-st4)}| Shift Amount: {np.linalg.norm(shift_vector)} \n"
                    file.write(outstring)
                return 1

            st3 = time.time()
            n = mutation(n, params, sim_params)

            st4 = time.time()
            nh_prev = nh

            params, sim_params, num_to_add, num_to_remove = HGT_logistic_event(t, n, params, sim_params)
            nh_gain = immunity_gain_from_kernel(nh, n, kernel_immunity, params, sim_params, num_to_add) #update nh
            nh = immunity_loss_uniform(nh_gain, n, params, sim_params, num_to_remove)
            
            diff_of_acquisition = num_to_add-num_to_remove
            shift_vector = compute_shift(nh, nh_prev, "max")
            ed = time.time()

            with open(foldername+'/runtime_stats.txt','a') as file:
                M = params["M"]
                outstring = f"t: {t}| N: {np.sum(n)}| Coverage: {time_conv(st2-st1)}| Growth: {time_conv(st3-st2)}| Mutation: {time_conv(st4-st3)}| Immunity: {time_conv(ed-st4)}| M: {M:.4f}| Net_Acq_Diff: {diff_of_acquisition:.4f}| Shift Amount: {np.linalg.norm(shift_vector):.4f} \n"
                file.write(outstring)

            t += sim_params["dt"]

    except KeyboardInterrupt or ValueError:
        write2json(foldername, params, sim_params)
        print(f"Stopped at time: {t}")
        return 0

    return 1

