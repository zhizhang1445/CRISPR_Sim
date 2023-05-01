import numpy as np
import copy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def init_guassian(init_num, sim_params, type = "n"):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    tt_len = (2*x_range*dx)**2
    N0 = init_num
    initial_position = sim_params["initial_mean_"+type]
    initial_var = sim_params["initial_var_"+type]

    x_linspace = np.arange(-x_range, x_range, dx)
    coordmap = np.array(np.meshgrid(x_linspace, x_linspace)).squeeze()
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

def init_guassian_parallel(init_num, sim_params, type = "n"):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    N0 = init_num
    initial_position = sim_params["initial_mean_"+type]
    initial_var = sim_params["initial_var_"+type]
    num_threads = sim_params["num_threads"]

    x_linspace = np.arange(-x_range+initial_position[0], x_range+initial_position[0], dx)
    y_linspace = np.arange(-x_range+initial_position[1], x_range+initial_position[1], dx)
    tt_len = len(x_linspace)
    

    p_marg_x = np.exp(-x_linspace**2/(2*(initial_var**2)))
    p_marg_x = p_marg_x/np.sum(p_marg_x) # initial prob distribution for n: Gaussian dist

    p_marg_y = np.exp(-x_linspace**2/(2*(initial_var**2)))
    p_marg_y = p_marg_y/np.sum(p_marg_y) 

    iter_per_thread = np.array_split(np.arange(0, N0), num_threads)

    def add_Gaussian_noise(subset):
        array = np.zeros([tt_len, tt_len], dtype=np.int16)

        for i in subset:
            x_index = np.random.choice(tt_len, p=p_marg_x) 
            y_index = np.random.choice(tt_len, p=p_marg_y) 
            array[x_index, y_index]+= 1
        return array

    results = Parallel(n_jobs=num_threads)(delayed(add_Gaussian_noise)(subset) for subset in iter_per_thread)
    out = np.sum(results, axis=0)

    return out

def init_uniform(number, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    tt_len = (2*x_range*dx)**2
    nh0 = np.zeros([2*x_range*dx, 2*x_range*dx], dtype=int) #Similarly, this is the initial value for nh    

    for i in range(number):
        x_index = np.random.choice(nh0.shape[0])
        y_index = np.random.choice(nh0.shape[1])  #similarly, this is really slow
        nh0[x_index, y_index] += 1

    return nh0

def init_uniform_parallel(init_num, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    N = init_num
    num_threads = sim_params["num_threads"]

    x_linspace = np.arange(-x_range, x_range, dx)
    y_linspace = np.arange(-x_range, x_range, dx)
    tt_len_x = len(x_linspace)
    tt_len_y = len(y_linspace)

    iter_per_thread = np.array_split(np.arange(0, N), num_threads)

    def add_Gaussian_noise(subset):
        array = np.zeros([tt_len_x, tt_len_y], dtype=int)

        for i in subset:
            x_index = np.random.choice(tt_len_x) 
            y_index = np.random.choice(tt_len_x) 
            array[x_index, y_index] += 1
        return array

    results = Parallel(n_jobs=num_threads)(delayed(add_Gaussian_noise)(subset) for subset in iter_per_thread)
    out = np.sum(results, axis=0)

    return out

def init_exptail_parallel(init_num, params, sim_params, type = "nh"):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    N0 = init_num
    initial_position = sim_params["initial_mean_"+type]
    initial_var = sim_params["initial_var_"+type]
    num_threads = sim_params["num_threads"]
    tau = params["M"]*params["Nh"]/params["N0"]
    v0 = params["v0"]
    axis = sim_params["tail_axis"]

    x_linspace = np.arange(-x_range+initial_position[0], x_range+initial_position[0], dx)
    y_linspace = np.arange(-x_range+initial_position[1], x_range+initial_position[1], dx)
    tt_len = len(x_linspace)
    
    p_marg_x = np.exp(-x_linspace**2/(2*(initial_var**2)))
    p_marg_y = np.exp(-y_linspace**2/(2*(initial_var**2)))

    if axis[0] == 0:
        p_marg_x = np.exp(-x_linspace/(v0*tau))*np.heaviside(axis[1]*(x_linspace-initial_position[0]), 0)

 # initial prob distribution for n: Gaussian dist

    if axis[0] == 1:
        p_marg_y = np.exp(-y_linspace/(v0*tau))*np.heaviside(axis[1]*(y_linspace-initial_position[1]), 0)


    p_marg_x = p_marg_x/np.sum(p_marg_x)
    p_marg_y = p_marg_y/np.sum(p_marg_y) 

    iter_per_thread = np.array_split(np.arange(0, N0), num_threads)

    def add_Gaussian_noise(subset):
        array = np.zeros([tt_len, tt_len], dtype=np.int16)

        for i in subset:
            x_index = np.random.choice(tt_len, p=p_marg_x) 
            y_index = np.random.choice(tt_len, p=p_marg_y) 
            array[x_index, y_index]+= 1
        return array

    results = Parallel(n_jobs=num_threads)(delayed(add_Gaussian_noise)(subset) for subset in iter_per_thread)
    out = np.sum(results, axis=0)

    return out

def init_quarter_kernel(params, sim_params):
    kernel = params["r"]
    conv_ker_size = sim_params["conv_size"]

    x_linspace = np.arange(0, conv_ker_size, 1)
    coordmap = np.array(np.meshgrid(x_linspace, x_linspace)).squeeze()

    radius = np.sqrt(np.sum((coordmap)**2, axis=0))
    matrix_ker = np.exp(-radius/kernel)
    return matrix_ker