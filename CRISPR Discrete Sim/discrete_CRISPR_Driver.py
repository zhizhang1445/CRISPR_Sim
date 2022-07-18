import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import convolve
import matplotlib.animation as animation
import warnings
import json
import scipy
import os

import discrete_CRISPR_methods as ds
import discrete_CRISPR_sim_methods as ds2

def main(params, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    x_linspace = np.arange(-x_range, x_range, dx)
    x_size = np.size(x_linspace)
    sim_params["x_size"] = x_size

    t_size = (sim_params["tf"]-sim_params["t0"])/sim_params["dt"]

    s = np.zeros([x_size, x_size], dtype=int)
    n = np.zeros([x_size, x_size], dtype=int) 
    nh = np.zeros([x_size, x_size], dtype=int) #nh(x,t) = Nh*h(x,t) there should be a total of Nh*M elements 
    coordmap = np.meshgrid(x_linspace, x_linspace) #coordmap is kinda stupid since we are not in any real space
    c = nh.copy() # c is unused (supposed to be coverage)

    n = ds2.init_n(n, params, sim_params)
    nh = ds2.init_nh(nh, params, sim_params)

    t_start = sim_params["t0"] #Time parameters
    t_stop = sim_params["tf"]
    t_step = sim_params["dt"]

    frames_n = [] #Store the frames as gifs
    frames_nh = []
    frames_f = []
    times = []

    N = []

    for t in np.arange(t_start, t_stop, t_step):

        f = ds.fitness_spacers(n, nh, params, sim_params) #f is now a masked array (where mask is where eff_R0 = 0)
        n = ds.virus_growth(n, f, params, sim_params) #update
        n = ds.mutation(n, params, sim_params)

        nh = ds.immunity_gain(nh, n) #update nh
        nh = ds.immunity_loss(nh, n)

        current_N = np.sum(n)
        current_Nh = np.sum(nh)
        
        frames_nh.append([nh])
        frames_f.append([f])
        frames_n.append([n])
        times.append([t])
        N.append([current_N])

        # n_step_prior = sim_params["n_step_prior"]

        if (current_N > current_Nh/2) and (t > (t_stop - t_start)/2):
            print("Population Reset")
            break

        if (current_N == 0):
            print("Population Death")
            break

    os.mkdir(sim_params["folder_name"])
    os.chdir(sim_params["folder_name"])

    ds.write2json("", params, sim_params)
    ds.makeGif(frames_n, "n_simulation")
    ds.makeGif(frames_nh, "nh_simulation")
    ds.makeGif(frames_f, "f_simulation")
    return True


if __name__ == "__main__":
    params = { #parameters relevant for the equations
        "Nh":             100,
        "N0":            1000,
        "R0":             10,
        "M":              100, #Also L, total number of spacers
        "D":                3, #Unused
        "mu":             0.1, #mutation rate
        "gamma_shape":     20, 
        "Np":              10, #Number of Cas Protein
        "dc":               8, #Required number of complexes to activate defence
        "h":               10, #coordination coeff
        "r":             0.5, #cross-reactivity kernel
    }
    sim_params = { #parameters relevant for the simulation (including Inital Valuess)
        "xdomain":                   10,
        "dx":                         1,
        "t0":                         0, 
        "tf":                       100,
        "dt":                       0.1,
        "noise_mean":                 0,
        "noise_std":                0.1,
        "initial_mean":           [0,0],
        "initial_var":                5,
        "n_step_prior":               5,
        "folder_name":  "simulation#11/",
        "conv_size":                  1,
    }
    main(params, sim_params)