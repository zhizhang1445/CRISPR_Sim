import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed
import scipy

def coverage_convolution(nh, kernel, params, sim_params):
    h = nh/params["Nh"]

    if sim_params["conv_size"] == 1:
        return h/params["M"]
    else:
        out = scipy.signal.convolve2d(h, kernel, mode='same')
        return out/params["M"]
    
def coverage_parrallel_convolution(nh, kernel, params, sim_params):
    num_cores = 32
    input_data = nh/params["Nh"]

    def convolve_subset(input_data_subset):
        return scipy.signal.convolve2d(input_data_subset, kernel, mode='same')

    input_data_subsets = np.array_split(input_data, num_cores, axis = 0)

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
    h = nh/Nh

    p_0_spacer = p_zero_spacer(h, p, params, sim_params)
    p_1_spacer = p_single_spacer(h, p, params, sim_params)
    p_tt = p_1_spacer + p_0_spacer

    if (np.min(p_tt)) < 0:
        raise ValueError("negative probability")
    
    eff_R0 = p_tt*R0
    return eff_R0

def control_fitness(f, n, params, sim_params):
    f_avg = np.sum(f*n)/np.sum(n)
    f_norm = f-f_avg

    f_norm = np.clip(f_norm, 0, None)
    
    if np.min(f_norm) < 0 :
        return ValueError("Dafuq is list comprehension")
    
    return f_norm

def virus_growth(n, f, params, sim_params):
    dt = sim_params["dt"]
    n = np.random.poisson((1+f*dt)*n)
    return  n

def num_mutants(n, params, sim_params):
    mu = params["mu"]
    dt = sim_params["dt"]

    p = 1-np.exp(-1*mu*dt)
    return np.random.binomial(n, p) #   so this is really prob of finding k mutation in n possible virus with prob p in 1-e^-mudt
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

def immunity_loss_by_index(nh, n, params, simparams):
    nh = nh + n

    non_zero_ind = np.where(nh != 0)
    prob_ind = binomial_pdf(n, 0, n[non_zero_ind])
    pass


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


