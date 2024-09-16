import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from scipy.sparse import issparse
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from supMethods import timeit
from concurrent.futures import as_completed
from formulas import p_infection, binomial_pdf

def fitness(n, p, params, sim_params):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(p, np.ndarray):
        return fitness_1D(p, params, sim_params)
    elif ndim == 2 and issparse(n):
        return fitness_2D(n, p, params, sim_params)
    else:
        raise TypeError(f"Something went wrong with fitness n_dim: {ndim} but type is {type(n)}")

def fitness_2D(n, p_sparse, params, sim_params): #TODO PARALLELIZE THIS
    R0 = params["R0"]
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)

    x_ind, y_ind = np.nonzero(p_sparse) #also == np.nonzero(n)
    p_dense = np.array(p_sparse[x_ind, y_ind].todense()).squeeze()

    if np.sum(p_dense) == 0:
        raise ValueError("Zero spacer probability")
    
    p_tt = p_infection(p_dense, params, sim_params)

    if np.min(p_tt) < 0:
        raise ValueError("Negative Probability")
        
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)
    res[x_ind, y_ind] = np.log(R0*p_tt)
    return res

def fitness_1D(p_coverage, params, sim_params):
    R0 = params["R0"]

    p_inf = p_infection(p_coverage, params, sim_params)
    fit = np.log(R0*p_inf)
    return fit

def norm_fitness(f, n, params, sim_params):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(f, np.ndarray):
        f_avg = np.sum(f*n)/np.sum(n)
        new_f = f-f_avg

        return new_f
    elif ndim == 2 and issparse(f):
        return norm_fitness_2D(f, n, params, sim_params)
    else:
        raise TypeError(f"Something went wrong with Norm_F| n_dim: {ndim} but type is {type(n)}")

def norm_fitness_2D(f_sparse, n, params, sim_params):
    f_avg = np.sum(f_sparse.multiply(n))/np.sum(n)

    x_ind, y_ind = f_sparse.nonzero()
    new_f = scipy.sparse.dok_matrix(f_sparse.shape, dtype=float)
    new_f[x_ind, y_ind] = f_sparse[x_ind, y_ind].toarray() - f_avg
    return new_f

def phage_growth(n, f, params, sim_params, det_growth = False):
    ndim = sim_params["ndim"]
    if ndim == 1 and isinstance(f, np.ndarray):
        return phage_growth_1D(n, f, params, sim_params, det_growth = False)
    elif ndim == 2 and issparse(f):
        return phage_growth_2D(n, f, params, sim_params, det_growth = False)
    else:
        raise TypeError(f"Something went wrong with Growth| n_dim: {ndim} but type is {type(n)}")

def phage_growth_1D(n, f, params, sim_params, det_growth = False):
    dt = sim_params["dt"]

    n_new = np.zeros_like(n)
    if not det_growth:
        mean = np.clip((1+f*dt), a_min = 0, a_max=None)*n
        n_new = np.random.poisson(mean)
    else:
        n_new = np.clip(n + f*n, a_min=0, a_max=None)
    return n_new

def phage_growth_2D(n, f_sparse, params, sim_params, deterministric_growth = False): #TODO PARALLELIZE THIS
    dt = sim_params["dt"]
    x_ind, y_ind = n.nonzero()
    if scipy.sparse.issparse(f_sparse):
        f_dense = np.array(f_sparse[x_ind, y_ind].todense())
    else:
        f_dense = f_sparse[x_ind, y_ind]

    if scipy.sparse.issparse(n):
        n_dense = np.array(n[x_ind, y_ind].todense())
        n_new = scipy.sparse.dok_matrix(n.shape, dtype = int)
    else:
        n_dense = n[x_ind, y_ind]
        n_new = np.zeros_like(n)

    if not deterministric_growth:
        mean = np.clip((1+f_dense*dt), a_min = 0, a_max=None)*n_dense
        n_new[x_ind, y_ind] = np.random.poisson(mean)
    else:
        n_new[x_ind, y_ind] = np.clip(n_dense + f_dense*n_dense, a_min=0, a_max=None)
    return  n_new

def fitness_n_spacers(p_coverage, params, sim_params):
    M = params["M"]
    R0 = params["R0"]
    Np = params["Np"]
    dc = params["dc"]

    def p_n_infection(p_coverage, M, Np, dc):
        p_infection = (1-p_coverage)**M

        for n in range(1, M+1):
            p_n_spacer = binomial_pdf(M, n, p_coverage)
            for d in range(0, dc+1):
                p_infection += binomial_pdf(Np, d, n/M)*p_n_spacer
        return p_infection
    
    p_inf = p_n_infection(p_coverage, M, Np, dc)
    return np.log(R0*p_inf)

def derivative_p_infection(p_coverage, params, sim_params):
    M = params["M"]
    R0 = params["R0"]
    Np = params["Np"]
    dc = params["dc"]
    n_order_spacer = params["n_spacer"]
    if n_order_spacer > M:
        n_order_spacer = M

    derivative_p_infection = 0
    for n in range(0, n_order_spacer+1):
        derivative_p_n_spacer = n*(binomial_pdf(M, n, p_coverage)/p_coverage)
        derivative_p_n_spacer -= (M-n)*(binomial_pdf(M, n, p_coverage)/(1-p_coverage))

        for d in range(0, dc+1):
            derivative_p_infection += binomial_pdf(Np, d, n/M)*derivative_p_n_spacer
    return derivative_p_infection

def derivative_fitness(p_coverage, params, sim_params):
    p_inf = p_infection(p_coverage, params, sim_params)
    derivative_p_inf = derivative_p_infection(p_coverage, params, sim_params)
    derivative_fit = (1/p_inf)*derivative_p_inf
    return derivative_fit

def find_root_fitness(params, sim_params, n_itr = 1000, err = 1e-7, to_print = False):
    x_old = 0.5

    for i in range(n_itr):
        f0 = fitness_1D(x_old, params, sim_params)
        if to_print:
            print("New Root: ", x_old,"|  New Fitness:  ", f0)
        f_prime = derivative_fitness(x_old, params, sim_params)
        
        x_new = x_old - (f0/f_prime)
        if np.abs(x_new - x_old) < err:
            break
        x_old = x_new

    return x_new

