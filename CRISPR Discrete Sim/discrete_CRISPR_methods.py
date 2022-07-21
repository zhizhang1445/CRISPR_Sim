import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import matplotlib.animation as animation
import scipy
from scipy import signal
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
    kernel = params["r"]
    conv_ker_size = sim_params["conv_size"]

    x_linspace = np.arange(-conv_ker_size, conv_ker_size, 1)
    coordmap = np.meshgrid(x_linspace, x_linspace)

    radius = np.sqrt((coordmap[0])**2 + (coordmap[1])**2)
    matrix_ker = np.exp(-radius/kernel)

    res = signal.convolve(h, matrix_ker, mode= "same")
    return res/params["M"] #I know this looks stupid but the coverage is not necessarily just a scale

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

    p = h/M
    p_1_spacer = binomial_pdf(M, 1, p)
    p_shared = 0
    for d in range(1, Np):
        p_shared += binomial_pdf(Np, d, 1/M)*p_1_spacer*(-1*alpha(d, params))
    return p_shared

def fitness(n, nh, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    Nh = params["Nh"]

    h = nh/Nh
    eff_R0 = R0*(1-coverage(h, params, sim_params))**M
    mask = (eff_R0 <= 0)
    ma_eff_R0 = ma.masked_array(eff_R0, mask = mask)
    res = ma.log(ma_eff_R0)
    return res

def fitness_spacers(n, nh, params, sim_params):
    R0 = params["R0"]
    Nh = params["Nh"]
    h = nh/Nh

    p_0_spacer = p_zero_spacer(h, params, sim_params)
    p_1_spacer = p_single_spacer(h, params, sim_params)
    p_tt = p_1_spacer + p_0_spacer

    if (np.min(p_0_spacer-p_1_spacer)) < 0:
        print("negative probability")
        raise ValueError
    
    eff_R0 = p_tt*R0
    mask = (eff_R0 <= 0)
    ma_eff_R0 = ma.masked_array(eff_R0, mask = mask)
    res = ma.log(ma_eff_R0)

    mask2 = ((1+res*sim_params["dt"])<=0).filled()
    res.mask = mask2
    return res

def fitness_shared_spacers(n, nh, params, sim_params):
    R0 = params["R0"]
    Nh = params["Nh"]
    rho = params["rho"]
    h = nh/Nh

    p_0_spacer = p_zero_spacer(h, params, sim_params)
    p_1_spacer = p_single_spacer(h, params, sim_params)
    p_tt = p_1_spacer + p_0_spacer

    if (np.min(p_0_spacer-p_1_spacer)) < 0:
        print("negative probability")
        raise ValueError
    
    eff_R0 = R0*(p_tt)*np.exp(-1*n*rho)
    mask = (eff_R0 <= 0)
    ma_eff_R0 = ma.masked_array(eff_R0, mask = mask)
    res = ma.log(ma_eff_R0)

    mask2 = ((1+res*sim_params["dt"])<=0).filled()
    res.mask = mask2
    return res


def fitness_controlled(n, nh, params, sim_params):
    f = fitness(n, nh, params, sim_params)
    f_avg = np.sum(f*n)/np.sum(n)
    f_norm = f-f_avg

    mask2 = ((1+f_norm*sim_params["dt"])<=0).filled()
    f_norm.mask = mask2
    return f_norm

def fitness_spacers_controlled(n, nh, params, sim_params):
    f = fitness_spacers(n, nh, params, sim_params)
    f_avg = np.sum(f*n)/np.sum(n)
    f_norm = f-f_avg

    mask2 = ((1+f_norm*sim_params["dt"])<=0).filled()
    f_norm.mask = mask2
    return f_norm

def virus_growth(n, f, params, sim_params):
    dt = sim_params["dt"]
    samples = np.random.poisson((1+f*dt)*n)
    res = ma.array(samples, mask = f.mask, fill_value=0)
    return  res.filled()

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