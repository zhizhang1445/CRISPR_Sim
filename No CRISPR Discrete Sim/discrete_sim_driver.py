import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.ndimage import convolve
import matplotlib.animation as animation
import warnings
import json
import scipy
import os

import discrete_simulation_methods as ds

def main():
  
    params = { #parameters relevant for the equations
        "Nh":          100000,
        "N0":             100,
        "R0":               3,
        "M":                1, #Also L
        "D":                3, #Unused
        "dc":               5, #Unused
        "mu":             0.1, 
        "gamma_shape":     20, 
    }
    sim_params = { #parameters relevant for the simulation (including Inital Valuess)
        "xdomain":                 1000,
        "dx":                         1,
        "t0":                         0, 
        "tf":                       100,
        "dt":                       0.1,
        "noise_mean":                 0,
        "noise_std":                0.1,
        "initial_mean":           [0,0],
        "initial_var":                5,
        "n_step_prior":              10,
        "folder_name":   "simulation#12",
    }
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


    n0 = np.zeros(n.size, dtype=int) #Initial value of n0, initialized with a gaussian distributed
    N0 = int(params["N0"])
    initial_position = sim_params["initial_mean"]
    initial_var = sim_params["initial_var"]

    x_map = coordmap[0]
    y_map = coordmap[1]

    rsqrd = (x_map-initial_position[0])**2 + (y_map-initial_position[0])**2

    p0 = np.exp(-rsqrd/(2*(initial_var**2)))
    p0 = p0/np.sum(p0) # initial prob distribution for n: Gaussian dist

    p0 = p0.ravel()

    for i in range(N0):
        index = np.random.choice(n.size, p=p0) #Should really have a better way of doing this, this is extremely slow: maybe MPI?
        n0[index] += 1

    n = copy.deepcopy(n0.reshape(n.shape))

    nh0 = np.zeros(nh.size, dtype=int) #Similarly, this is the initial value for nh
    Nh = int(params["Nh"])
    M = int(params["M"])
    initial_position = sim_params["initial_mean"]
    initial_var = 10*sim_params["initial_var"] #just 10 times the variance of n0, maybe change this?

    x_map = coordmap[0]
    y_map = coordmap[1]

    rsqrd = (x_map-initial_position[0])**2 + (y_map-initial_position[0])**2

    p0 = np.exp(-rsqrd/(2*(initial_var**2)))
    p0 = p0/np.sum(p0) # initial prob distribution for n: Gaussian dist

    p0 = p0.ravel()

    for i in range(Nh*M):
        index = np.random.choice(nh.size, p=p0) #similarly, this is really slow
        nh0[index] += 1

    nh = copy.deepcopy(nh0.reshape(nh.shape))

    t_start = sim_params["t0"] #Time parameters
    t_stop = sim_params["tf"]
    t_step = sim_params["dt"]

    frames_n = [] #Store the frames as gifs
    frames_nh = []
    frames_f = []
    times = []

    N = []

    for t in np.arange(t_start, t_stop, t_step):

        f = ds.fitness(nh, params, sim_params)
        n = ds.virus_growth(n, f, params, sim_params) #update n
        n = ds.mutation(n, params, sim_params)

        nh_old = copy.deepcopy(nh)
        nh = ds.immunity_gain(nh, n) #update nh
        nh = ds.immunity_loss(nh, n)

        current_N = np.sum(n)
        current_Nh = np.sum(nh)
        
        frames_nh.append([nh])
        frames_f.append([f])
        frames_n.append([n])
        times.append([t])
        N.append([current_N])
        if (current_N > current_Nh/2) and (t > (t_stop - t_start)/2):
            print("Population Reset")
            break


        if (current_N == 0):
            print("Population Death")
            break
    
    os.mkdir(sim_params["folder_name"])
    os.chdir(sim_params["folder_name"])

    ds.write2json("test", params, sim_params)
    ds.makeGif(frames_n, "n_simulation")
    ds.makeGif(frames_nh, "nh_simulation")
    ds.makeGif(frames_f, "f_simulation")



if __name__ == "__main__":
    main()