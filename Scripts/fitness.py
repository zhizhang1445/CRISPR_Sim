import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from supMethods import timeit
from concurrent.futures import as_completed
from formulas import find_max_value_location

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
    for d in range(0, Np+1):
        p_shared += binomial_pdf(Np, d, 1/M)*p_1_spacer*(1-alpha(d, params))
    return p_shared

def fitness_spacers(n, nh, p_sparse, params, sim_params): #TODO PARALLELIZE THIS
    R0 = params["R0"]
    Nh = params["Nh"]
    M = params["M"]
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)

    x_ind, y_ind = np.nonzero(p_sparse) #also == np.nonzero(n)
    p_dense = np.array(p_sparse[x_ind, y_ind].todense()).squeeze()

    if np.sum(p_dense) == 0:
        raise ValueError("Zero spacer probability")
    
    p_0_spacer = p_zero_spacer(p_dense, params, sim_params)
    p_1_spacer = p_single_spacer(p_dense, params, sim_params)
    p_tt = p_0_spacer + p_1_spacer
    # p_tt = np.power((1-p_dense), M)

    if np.min(p_tt) < 0:
        raise ValueError("Negative Probability")
        
    res = scipy.sparse.dok_matrix(n.shape, dtype=float)
    res[x_ind, y_ind] = np.log(R0*p_tt)
    return res


def fitness_spacers_fast(f_prev, shift, params):
    x_ind, y_ind = np.nonzero(f_prev)
    shift_x, shift_y = shift

    res = scipy.sparse.dok_matrix(f_prev.shape, dtype = float)
    try:
        res[x_ind+shift_x, y_ind+shift_y] = f_prev[x_ind, y_ind]
    except IndexError:
        return f_prev
    return res

def norm_fitness(f_sparse, n, params, sim_params):
    f_avg = np.sum(f_sparse.multiply(n))/np.sum(n)

    x_ind, y_ind = f_sparse.nonzero()
    new_f = scipy.sparse.dok_matrix(f_sparse.shape, dtype=float)
    new_f[x_ind, y_ind] = f_sparse[x_ind, y_ind].toarray() - f_avg
    return new_f

def virus_growth(n, f_sparse, params, sim_params, deterministric_growth = False): #TODO PARALLELIZE THIS
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


    n_new = scipy.sparse.dok_matrix(n.shape, dtype = int)

    if not deterministric_growth:
        mean = np.clip((1+f_dense*dt), a_min = 0, a_max=None)*n_dense
        n_new[x_ind, y_ind] = np.random.poisson(mean)
    else:
        n_new[x_ind, y_ind] = np.clip(n_dense + f_dense*n_dense, a_min=0, a_max=None)
    return  n_new

def pred_value(params, sim_params):
    erf = scipy.special.erf
    var_nh = sim_params["initial_var_nh"]
    r = params["r"]
    const = np.exp(-var_nh**2/(2*np.power(r,2)))
    neg_exp = lambda x: (1/2)*np.exp(-x/r)
    pos_exp = lambda x: (1/2)*np.exp(x/r)

    div_const = 1/(var_nh*np.sqrt(2))
    erf_mean = var_nh**2/r
    neg_erf = lambda x: erf(div_const*(x-erf_mean))
    pos_erf = lambda x: erf(div_const*(x+erf_mean))

    def anal_result(x):
        return const*(neg_exp(x)*(neg_erf(x)+1)-pos_exp(x)*(pos_erf(x)-1))
    
    def anal_delay(x):
        return anal_result(x-erf_mean)
    return anal_delay