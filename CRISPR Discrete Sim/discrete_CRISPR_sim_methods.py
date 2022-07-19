import numpy as np
import copy
import matplotlib.pyplot as plt

def init_n(n, params, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    N0 = int(params["N0"])
    initial_position = sim_params["initial_mean"]
    initial_var = sim_params["initial_var"]

    x_linspace = np.arange(-x_range, x_range, dx)
    coordmap = np.meshgrid(x_linspace, x_linspace)
    n0 = np.zeros(n.size, dtype=int) #Initial value of n0, initialized with a gaussian distributed

    x_map = coordmap[0]
    y_map = coordmap[1]

    rsqrd = (x_map-initial_position[0])**2 + (y_map-initial_position[0])**2

    p0 = np.exp(-rsqrd/(2*(initial_var**2)))
    p0 = p0/np.sum(p0) # initial prob distribution for n: Gaussian dist

    p0 = p0.ravel()

    for i in range(N0):
        index = np.random.choice(n.size, p=p0) #Should really have a better way of doing this, this is extremely slow: maybe MPI?
        n0[index] += 1

    return n0.reshape(n.shape)

def init_nh(nh, params, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    Nh = int(params["Nh"])
    M = int(params["M"])
    initial_position = sim_params["initial_mean"]
    initial_var = 10*sim_params["initial_var"] #just 10 times the variance of n0, maybe change this?

    x_linspace = np.arange(-x_range, x_range, dx)
    coordmap = np.meshgrid(x_linspace, x_linspace)
    nh0 = np.zeros(nh.size, dtype=int) #Similarly, this is the initial value for nh    

    x_map = coordmap[0]
    y_map = coordmap[1]

    rsqrd = (x_map-initial_position[0])**2 + (y_map-initial_position[0])**2

    p0 = np.exp(-rsqrd/(2*(initial_var**2)))
    p0 = p0/np.sum(p0) # initial prob distribution for n: Gaussian dist

    p0 = p0.ravel()

    for i in range(Nh*M):
        index = np.random.choice(nh.size, p=p0) #similarly, this is really slow
        nh0[index] += 1

    return nh0.reshape(nh.shape)


