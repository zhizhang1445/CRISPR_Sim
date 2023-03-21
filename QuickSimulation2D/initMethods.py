import numpy as np
import copy
import matplotlib.pyplot as plt

def init_guassian(init_num, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    tt_len = (2*x_range*dx)**2
    N0 = init_num
    initial_position = sim_params["initial_mean"]
    initial_var = sim_params["initial_var"]

    x_linspace = np.arange(-x_range, x_range, dx)
    coordmap = np.meshgrid(x_linspace, x_linspace)
    n0 = np.zeros(tt_len, dtype=int) #Initial value of n0, initialized with a gaussian distributed

    x_map = coordmap[0]
    y_map = coordmap[1]

    rsqrd = (x_map-initial_position[0])**2 + (y_map-initial_position[0])**2

    p0 = np.exp(-rsqrd/(2*(initial_var**2)))
    p0 = p0/np.sum(p0) # initial prob distribution for n: Gaussian dist

    p0 = p0.ravel()

    for i in range(N0):
        index = np.random.choice(tt_len, p=p0) #Should really have a better way of doing this, this is extremely slow: maybe MPI?
        n0[index] += 1

    return n0.reshape([2*x_range*dx, 2*x_range*dx])

def init_uniform(number, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    tt_len = (2*x_range*dx)**2
    nh0 = np.zeros([2*x_range*dx, 2*x_range*dx], dtype=int) #Similarly, this is the initial value for nh    

    for i in range(number):
        index = np.random.choice(nh0.size) #similarly, this is really slow
        nh0[index] += 1

    return nh0.reshape([2*x_range*dx, 2*x_range*dx])