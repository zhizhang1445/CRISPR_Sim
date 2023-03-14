import numpy as np
import scipy.stats 

def init_n(array, params, sim_params):
    if (np.sum(array) != 0):
        array.fill(0)

    x_range = sim_params["xdomain"] #Initialize the spaces
    x_space = np.arange(-x_range, x_range)
    N0 = int(params["N0"])
    initial_position = sim_params["initial_mean"]
    initial_var = sim_params["initial_var"]

    prob = np.exp(-(x_space - initial_position)**2/(2*initial_var))
    prob = prob/np.sum(prob)

    for i in range(N0):
        index = np.random.choice(np.arange(0, len(array)), p = prob)
        array[index] += 1

    return array

def init_nh(array, params, sim_params):
    if (np.sum(array) != 0):
        array.fill(0)

    x_range = sim_params["xdomain"] #Initialize the spaces
    x_space = np.arange(-x_range, x_range)
    Nh = int(params["Nh"])

    for i in range(Nh):
        index = np.random.choice(np.arange(0, 2*x_range))
        array[index] += 1

    return array