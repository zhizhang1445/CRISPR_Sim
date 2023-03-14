import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import matplotlib.animation as animation
import scipy
import json
from scipy import signal

def coverage(h, params, sim_params):
    if (params["r"] == 0) or (sim_params["conv_size"] == 0 ): 
        return h/params["M"]

    conv_size = sim_params["conv_size"]
    rad = params["r"]
    M = params["M"]

    kernel = np.zeros(2*conv_size + 1)
    for i, ind in enumerate(kernel):
        kernel[i] = np.exp(-1*np.abs(i - conv_size)/rad)

    res = convolve(h, kernel, "same")/M
    return res #I know this looks stupid but the coverage is not necessarily just a scale

def alpha(d, params):
    dc = params["dc"]
    h = params["h"]

    return d**h/(d**h + dc**h)

def binomial_pdf(n, x, p):
    multiplicity = scipy.special.binom(n, x)
    bernouilli = (p**x)*((1-p)**(n-x))
    return multiplicity*bernouilli

def p_zero_spacer(h, params, sim_params):
    M = params["M"]
    p = h/M
    return binomial_pdf(M, 0, p)

def p_single_spacer(h, params, sim_params):
    M = params["M"]
    Nh = params["Nh"]
    Np = params["Np"]

    p = coverage(h, params, sim_params)
    p_1_spacer = binomial_pdf(M, 1, p)
    p_shared = 0
    for d in range(1, Np):
        p_shared += binomial_pdf(Np, d, 1/M)*p_1_spacer*(1-alpha(d, params))
    return p_shared

def fitness(n, nh, params, sim_params):
    R0 = params["R0"]
    Nh = params["Nh"]
    h = nh/Nh

    p_0_spacer = p_zero_spacer(h, params, sim_params)
    p_1_spacer = p_single_spacer(h, params, sim_params)
    p_tt = p_1_spacer + p_0_spacer

    if (np.min(p_tt)) < 0:
        print("negative probability")
        raise ValueError
    
    eff_R0 = p_tt*R0
    mask = (eff_R0 <= 0)
    ma_eff_R0 = ma.masked_array(eff_R0, mask = mask)
    res = ma.log(ma_eff_R0)

    mask2 = ((1+res)<=0).filled()
    res.mask = mask2
    return res

def virus_growth(n, f, params, sim_params):
    dt = 1
    samples = np.random.poisson((1+f*dt)*n)
    res = ma.array(samples, mask = f.mask, fill_value=0)
    return  res.filled()

def num_mutants(n, params, sim_params):
    mu = params["mu"]
    dt = 1

    p = 1-np.exp(-1*mu*dt)
    return np.random.binomial(n, p) #   so this is really prob of finding k mutation in n possible virus with prob p in 1-e^-mudt
                                    #   so p to not mutated is really e^-mu*dt

def num_mutation(params, sim_params):
    mu = params["mu"]
    dt = 1

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
    dx = 1

    jump = np.zeros(1)
    mean = 2*dx
    theta = mean/shape_param
    direction = np.random.choice([-1, 1])
    
    for i in range(m):
        jump = jump + direction*np.random.gamma(shape_param, theta)

    jump = np.round(jump)
    return jump

def immunity_gain(nh, n):
    return nh + n # to gain immunity you need some amount infected

def immunity_loss(nh, n):
    N = np.sum(n)
    checksum = np.sum(nh)

    for i in range(N):
        indexes = np.nonzero(nh)[0]
        index = np.random.choice(indexes) # Choose random spots uniformly to loose immunity

        nh[index] -= 1 #There is a race condition, don't fuck with this
    
    if np.any(nh<0):
        raise ValueError("Immunity is negative")
    elif np.sum(nh) != checksum - N :
        raise ValueError("In and out total value don't match")

    return nh

def mutation(n, params, sim_params): #this is the joined function for the mutation step
    checksum = np.sum(n)

    mutation_map = num_mutants(n, params, sim_params) # The mutation maps tells you how many virus have mutated at each location
    x_ind = np.nonzero(mutation_map)[0] #finding out where the mutations happends
    num_mutation_sites = (x_ind).size

    for i in range(num_mutation_sites): 
        num_mutants_at_site = mutation_map[x_ind[i]] # unpacking the number of mutated virus
        n[x_ind[i]] -= num_mutants_at_site #first remove all the virus that moved
        for j in range(num_mutants_at_site): #Find out where those virus have moved to
            num_mutation_at_site = num_mutation(params, sim_params) #Sampling how many single mutation for a single virus (num of jumps) 
            jump = mutation_jump(num_mutation_at_site, params, sim_params) #Sampling the jump

            try:
                new_x_loc = (x_ind[i] + jump).astype(int)
                n[new_x_loc] += 1
            except IndexError: #Array Out of Bounds
                if new_x_loc >= n.size:
                    new_x_loc = -1 #lmao this is gonna be a pain in cpp
                n[new_x_loc] += 1

    if np.sum(n) != checksum : #Should conserve number of virus/infection
        raise ValueError('mutation changed total number of n')
    elif np.any(n<0): #Should definitely not be negative
        raise ValueError('mutation made n negative')

    return n