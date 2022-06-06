import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import matplotlib.animation as animation
import scipy
import json

def makeGif(frame_stack, name):
    fig = plt.figure()

    animation_frames = []
    for frame in frame_stack:
        frame = np.squeeze(frame)
        animation_frames.append([plt.imshow(frame, animated=True)])

    ani = animation.ArtistAnimation(
        fig, animation_frames, interval=50, blit=True, repeat_delay=1000)
    ani.save(name + ".gif")

    return 1

def write2json(name, params, sim_params):
    with open(name + '_params.json', 'w') as fp:
        json.dump(params, fp)

    with open(name + '_sim_params.json', 'w') as fp:
        json.dump(sim_params, fp)

def coverage(h, params, sim_params):
    return h/params["M"] #I know this looks stupid but the coverage is not necessarily just a scale

def fitness(nh, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    Nh = params["Nh"]


    h = nh/Nh
    eff_R0 = R0*(1-coverage(h, params, sim_params))**M
    res = np.log(eff_R0, out=np.zeros_like(eff_R0), where=(eff_R0!=0))  # I'm not convinced this is right, the log(0) = 0 as 
                                                                        # is might not be correct but R0 should not be smaller than 1 
    # check = R0*(1-coverage(h))**M
    # out = np.log(R0*((1-coverage(h))**M)) # This just fails as log(0) happens regularly
    return res


def virus_growth(n, f, params, sim_params):
    dt = sim_params["dt"]
    return np.random.poisson((1+f*dt)*n) #growth rate is (1+fdt)n as normal really

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
        return num_mutation(params, sim_params) #conditioned as to have at least one mutation


def mutation_jump(m, params, sim_params):
    shape_param = params["gamma_shape"]
    dx = sim_params["dx"]

    jump = np.zeros(2)
    mean = 2*dx
    theta = mean/shape_param
    
    for i in range(m):
        jump = jump + np.random.gamma(shape_param, theta, size=2) #The distribution of jump is a sum of gamma distribution. 

    jump = np.round(jump)
    return jump

def immunity_gain(nh, n): 
    return nh + n # to gain immunity you need some amount infected

def immunity_loss(nh, n):
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