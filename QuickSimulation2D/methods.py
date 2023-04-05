import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed
from numpy.random import default_rng
import scipy

def coverage_convolution(nh, kernel, params, sim_params):
    h = nh/params["Nh"]

    if sim_params["conv_size"] == 1:
        return h/params["M"]
    else:
        out = scipy.signal.convolve2d(h, kernel, mode='same')
        return out/params["M"]
    
def square_split(array, num_split):
    if np.ndim(array) == 1:
        return np.split(array, num_split)
    elif np.ndim(array) > 2:
        raise IndexError("2D or 1D plz")
    
    res = []
    column_split = np.array_split(array, num_split, axis=0)
    for col in column_split:
        row_col_split = np.array_split(col, num_split, axis=0)
        res.extend(row_col_split)
    return res
    
def coverage_parrallel_convolution(nh, kernel, params, sim_params):
    num_cores = sim_params["num_threads"]
    input_data = nh/params["Nh"]

    def convolve_subset(input_data_subset):
        if np.sum(input_data_subset) == 0:
            return input_data_subset
        else:
            return scipy.signal.convolve2d(input_data_subset, kernel, mode='same')

    input_data_subsets = square_split(input_data, num_cores)
    # input_data_subsets = np.split(input_data, num_cores, axis=0)

    results = Parallel(n_jobs=num_cores)(delayed(convolve_subset)(subset) for subset in input_data_subsets)

    output_data = np.concatenate(results, axis = 0)

    return output_data/params["M"]
    
def alpha(d, params):
    dc = params["dc"]
    h = params["h"]

    return d**h/(d**h + dc**h)

def binomial_pdf(n, x, p):
    if x == 0 or x == n:
        multiplicity = 1
    elif x  == 1 or x == n-1:
        multiplicity = n
    else:
        multiplicity = scipy.special.binom(n, x)
    
    if multiplicity == 0:
        return ValueError("Sorry Bernouilli is rolling in his grace")
    
    bernouilli = (p**x)*((1-p)**(n-x))
    return multiplicity*bernouilli

def p_zero_spacer(h, p_coverage, params, sim_params):
    M = params["M"]
    p = p_coverage
    return binomial_pdf(M, 0, p)

def p_single_spacer(h, p_coverage, params, sim_params):
    M = params["M"]
    Nh = params["Nh"]
    Np = params["Np"]

    p = p_coverage
    p_1_spacer = binomial_pdf(M, 1, p)
    p_shared = 0
    for d in range(1, Np):
        p_shared += binomial_pdf(Np, d, 1/M)*p_1_spacer*(1-alpha(d, params))
    return p_shared

def fitness_spacers(n, nh, p, params, sim_params):
    R0 = params["R0"]
    Nh = params["Nh"]
    M = params["M"]
    h = nh/Nh

    f_new = np.zeros(n.shape)
    x_ind, y_ind = np.nonzero(n)

    p_0_spacer = p_zero_spacer(h, p, params, sim_params)
    p_1_spacer = p_single_spacer(h, p, params, sim_params)
    # p_tt = p_0_spacer + p_1_spacer
    p_tt = (1-p)**M #Remove this is just for testing

    if (np.min(p_tt)) < 0:
        raise ValueError("negative probability")
        
    f_new[x_ind, y_ind] = np.log(R0*n[x_ind, y_ind])
    return f_new

def control_fitness(f, n, params, sim_params):
    f_avg = np.sum(f*n)/np.sum(n)
    f_norm = f-f_avg

    f_norm = np.clip(f_norm, 0, None)
    
    if np.min(f_norm) < 0 :
        return ValueError("Dafuq is list comprehension")
    
    return f_norm

def virus_growth(n, f, params, sim_params):
    dt = sim_params["dt"]
    cond1 = (1+f*dt) > 0
    cond2 = n > 0

    x_ind, y_ind = np.where(np.bitwise_and(cond1, cond2))
    n[x_ind, y_ind] = np.random.poisson((1+f[x_ind, y_ind]*dt)*n[x_ind, y_ind]).astype(np.int16)
    x_ind, y_ind = np.where(np.invert(cond1))
    n[x_ind, y_ind] = 0
    return  n

def num_mutants(n, params, sim_params):
    mu = params["mu"]
    dt = sim_params["dt"]

    x_ind, y_ind = np.nonzero(n)
    map = np.zeros(n.shape, dtype=np.int16)

    p = 1-np.exp(-1*mu*dt)
    map[x_ind, y_ind] = np.random.binomial(n[x_ind, y_ind], p) #   so this is really prob of finding k mutation in n possible virus with prob p in 1-e^-mudt
    return map
                                    #   so p to not mutated is really e^-mu*dt

def num_mutation(params, sim_params):
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

def immunity_update(nh, n, params, sim_params):
    nh = nh + n # to gain immunity you need some amount infected

    N = np.sum(n)
    checksum = np.sum(nh)

    for i in range(N):
        indexes = np.argwhere(nh > 0)
        index = np.random.choice(indexes.shape[0]) # Choose random spots uniformly to loose immunity

        nh[indexes[index, 0], indexes[index, 1]] -= 1 #There is a race condition, don't fuck with this
    
    if np.any(nh<0):
        raise ValueError("Immunity is negative")
    elif np.sum(nh) != checksum - N :
        raise ValueError("In and out total value don't match")

    return nh

def immunity_update_parallel(nh, n, params, sim_params):
    Nh = params["Nh"]
    N = np.sum(n)
    num_threads = sim_params["num_threads"]
    nh = nh + n
    num_to_remove = int(np.sum(nh) - Nh)

    nonzero_indices = np.nonzero(nh)
    nonzero_values = [nh[index] for index in zip(*nonzero_indices)]
    index_nonzero_w_repeats = []
    for value, index in zip(nonzero_values, zip(*nonzero_indices)):
        for i in range(int(value)):
            index_nonzero_w_repeats.append(index)

    sample_flat_ind = np.random.choice(len(index_nonzero_w_repeats), num_to_remove,replace = False)

    ind_per_thread_list = np.array_split(sample_flat_ind, num_threads)

    def remove_points(flat_index):
        array = np.zeros(nh.shape, dtype = np.int64)
        sample_ind = [index_nonzero_w_repeats[i] for i in flat_index]
        for x,y in sample_ind:
            array[x, y] -= 1

        return array

    results = Parallel(n_jobs=num_threads)(
        delayed(remove_points)(flat_index) for flat_index in ind_per_thread_list)
    nh = nh + np.sum(results, axis=0)

    if np.sum(nh) != Nh:
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh))
    if np.min(nh) < 0:
        raise ValueError("bacteria population is negative")

    return nh


def mutation(n, params, sim_params): #this is the joined function for the mutation step
    checksum = np.sum(n)

    mutation_map = num_mutants(n, params, sim_params) # The mutation maps tells you how many virus have mutated at each location
    x_ind, y_ind = np.nonzero(mutation_map) #finding out where the mutations happends
    num_mutation_sites = x_ind.size 
    
    for i in range(num_mutation_sites): 
        num_mutants_at_site = mutation_map[x_ind[i], y_ind[i]] # unpacking the number of mutated virus
        n[x_ind[i], y_ind[i]] -= num_mutants_at_site #first remove all the virus that moved

        for j in range(num_mutants_at_site): #Find out where those virus have moved to
            num_mutation_at_site = num_mutation(params, sim_params) #Sampling how many single mutation for a single virus (num of jumps) 
            jump = mutation_jump(num_mutation_at_site, params, sim_params) #Sampling the jump

            try:
                new_x_loc = (x_ind[i] + jump[0]).astype(int)
                new_y_loc = (y_ind[i] + jump[1]).astype(int)
                n[new_x_loc, new_y_loc] += 1
            except IndexError: #Array Out of Bounds
                if new_x_loc >= n.shape[0]:
                    new_x_loc = -1 #lmao this is gonna be a pain in cpp
                if new_y_loc >= n.shape[1]:
                    new_y_loc = -1
                n[new_x_loc, new_y_loc] += 1

    if np.sum(n) != checksum : #Should conserve number of virus/infection
        raise ValueError('mutation changed total number of n')
    elif np.any(n<0): #Should definitely not be negative
        raise ValueError('mutation made n negative')

    return n

def mutation_parallel(n, params, sim_params):
    num_threads = sim_params["num_threads"]
    checksum = np.sum(n)

    mutation_map = num_mutants(n, params, sim_params) # The mutation maps tells you how many virus have mutated at each location
    x_ind, y_ind = np.nonzero(mutation_map) #finding out where the mutations happends

    x_ind_subsets = np.array_split(x_ind, num_threads)
    y_ind_subsets = np.array_split(y_ind, num_threads)

    def mutation_single(x_ind, y_ind):
        n_to_add = np.zeros(n.shape, dtype=np.int64)
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
                    if new_x_loc >= n.shape[0]:
                        new_x_loc = -1 #lmao this is gonna be a pain in cpp
                    if new_y_loc >= n.shape[1]:
                        new_y_loc = -1
                    n_to_add[new_x_loc, new_y_loc] += 1
        return n_to_add



    results = Parallel(n_jobs=num_threads)(delayed(mutation_single)(x_ind, y_ind) for x_ind, y_ind in zip(x_ind_subsets, y_ind_subsets))
    # results = mutation_single(n_to_add, x_ind, y_ind)

    n = n+np.sum(results, axis = 0)
    if checksum != np.sum(n):
        raise ValueError("Cries cuz Bacteria died during mutation")
    return n
    
