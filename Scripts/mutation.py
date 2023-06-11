import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
from supMethods import timeit

def num_mutants_parallel(n, params, sim_params): #TODO PARALLELIZE THIS
    mu = params["mu"]
    dt = sim_params["dt"]

    x_ind, y_ind = np.nonzero(n)
    map = np.zeros(n.shape, dtype=np.int16)

    p = 1-np.exp(-1*mu*dt)
    if scipy.sparse.issparse(n):
        map[x_ind, y_ind] = np.random.binomial(n[x_ind, y_ind].todense(), p) #   so this is really prob of finding k mutation in n possible virus with prob p in 1-e^-mudt
    else:
        map[x_ind, y_ind] = np.random.binomial(n[x_ind, y_ind], p)
    
    return map
                                    #   so p to not mutated is really e^-mu*dt

def num_mutation(params, sim_params): #Sparse is not needed
    mu = params["mu"]
    dt = sim_params["dt"]

    out = np.random.poisson(mu*dt)
    if out >= 1 :
        return out #This is what mu, is the average rate of mutation
    else:
        try:
            return num_mutation(params, sim_params) #conditioned as to have at least one mutation
        except RecursionError:
            return 0

def mutation_jump(m, params, sim_params):
    shape_param = params["gamma_shape"]
    dx = sim_params["dx"]

    jump = np.zeros(2)
    mean = 2*dx
    theta = mean/shape_param
    
    for i in range(m):
        angle = np.random.uniform(0, 2*np.pi) #Goodammit couldn't this have been clearer??? It's supposed to be isotropic
        jump = jump + np.random.gamma(shape_param, theta)*np.array([np.cos(angle), np.sin(angle)])
        #The distribution of jump is a sum of gamma distribution. 

    jump = np.round(jump)
    return jump

@timeit
def mutation(n, params, sim_params):
    num_threads = sim_params["num_threads"]
    checksum = np.sum(n)

    mutation_map = num_mutants_parallel(n, params, sim_params) # The mutation maps tells you how many virus have mutated at each location
    x_ind, y_ind = np.nonzero(mutation_map) #finding out where the mutations happends

    x_ind_subsets = np.array_split(x_ind, num_threads)
    y_ind_subsets = np.array_split(y_ind, num_threads)

    def mutation_single(x_ind, y_ind):
        n_to_add = scipy.sparse.dok_matrix(n.shape, dtype=np.int64)
        for x_i, y_i in zip(x_ind, y_ind):
            num_mutants_at_site = mutation_map[x_i, y_i] # unpacking the number of mutated virus
            n_to_add[x_i, y_i] -= num_mutants_at_site

            for j in range(num_mutants_at_site): #Find out where those virus have moved to
                num_mutation_at_site = num_mutation(params, sim_params) #Sampling how many single mutation for a single virus (num of jumps) 
                jump = mutation_jump(num_mutation_at_site, params, sim_params) #Sampling the jump

                try:
                    new_x_loc = (x_i + jump[0]).astype(int)
                    new_y_loc = (y_i + jump[1]).astype(int)
                    n_to_add[new_x_loc, new_y_loc] += 1
                except IndexError: #Array Out of Bounds
                    if new_x_loc < 0:
                        new_x_loc = 0
                    if new_y_loc < 0:
                        new_y_loc = 0

                    if new_x_loc >= n.shape[0]:
                        new_x_loc = -1 #lmao this is gonna be a pain in cpp
                    if new_y_loc >= n.shape[1]:
                        new_y_loc = -1

                    n_to_add[new_x_loc, new_y_loc] += 1
        return n_to_add

    results = Parallel(n_jobs=num_threads)(delayed(mutation_single)(x_ind, y_ind) for x_ind, y_ind in zip(x_ind_subsets, y_ind_subsets))
    # results = mutation_single(x_ind, y_ind)
    # n = n+results

    n = n+np.sum(results, axis = 0)
    if checksum != np.sum(n):
        raise ValueError("Cries cuz Bacteria died during mutation")
    return n
