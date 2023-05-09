import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
import scipy

def alpha(d, params): #This doesn't need to be sparsed
    dc = params["dc"]
    h = params["h"]

    return d**h/(d**h + dc**h)

def binomial_pdf(n, x, p_dense): #TODO Not Tested but sparsed in Theory
    if scipy.sparse.issparse(n):
        x_ind, y_ind = np.nonzero(n)
        multiplicity = scipy.sparse.dok_matrix(n.shape)
        multiplicity[x_ind, y_ind] = scipy.special.binom(n[x_ind, y_ind].todense(), x)
    else:
        multiplicity = scipy.special.binom(n, x)

    bernouilli = np.power(p_dense, x)*np.power((1-p_dense), (n-x))
    return multiplicity*bernouilli

def p_zero_spacer(p_dense, params, sim_params): #TODO SPARSE
    M = params["M"]
    return binomial_pdf(M, 0, p_dense)

def p_single_spacer(p_dense, params, sim_params): #TODO Sparsed in Theory
    M = params["M"]
    Np = params["Np"]

    p_1_spacer = binomial_pdf(M, 1, p_dense)
    p_shared = 0
    for d in range(1, Np):
        p_shared += binomial_pdf(Np, d, 1/M)*p_1_spacer*(1-alpha(d, params))
    return p_shared

def fitness_spacers(n, nh, p_sparse, params, sim_params): #TODO PARALLELIZE THIS
    R0 = params["R0"]
    Nh = params["Nh"]
    M = params["M"]

    x_ind, y_ind = np.nonzero(p_sparse) #also == np.nonzero(n)
    p_dense = np.array(p_sparse[x_ind, y_ind].todense()).squeeze()

    p_0_spacer = p_zero_spacer(p_dense, params, sim_params)
    p_1_spacer = p_single_spacer(p_dense, params, sim_params)
    p_tt = p_0_spacer + p_1_spacer
    # p_tt = np.power((1-p_dense), M)

    if np.min(p_tt) < 0:
        raise ValueError("Negative Probability")
        
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)
    res[x_ind, y_ind] = np.log(R0*p_tt)
    return res

def virus_growth(n, f_sparse, params, sim_params): #TODO PARALLELIZE THIS
    dt = sim_params["dt"]
    x_ind, y_ind = n.nonzero()
    if scipy.sparse.issparse(f_sparse):
        f_dense = np.array(f_sparse[x_ind, y_ind].todense())
    else:
        f_dense = f_sparse[x_ind, y_ind]

    if scipy.sparse.issparse(n):
        n_dense = np.array(n[x_ind, y_ind].todense())
    else:
        n_dense = n[x_ind, y_ind]

    mean = np.clip((1+f_dense*dt), a_min = 0, a_max=None)*n_dense

    n_new = scipy.sparse.dok_matrix(n.shape)
    n_new[x_ind, y_ind] = np.random.poisson(mean)
    return  n_new