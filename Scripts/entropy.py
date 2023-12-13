import numpy as np
import numpy.ma as ma
import scipy
from joblib import Parallel, delayed, parallel_backend
from supMethods import timeit
from formulas import find_max_value_location
from trajectory import fit_GMM

def compute_entropy(array, k=1):
    x_ind, y_ind = np.nonzero(array)    
    
    if scipy.sparse.issparse(array):
        non_zero_values = array[x_ind, y_ind].toarray().squeeze()
    else:
        non_zero_values = array[x_ind, y_ind]
    entropy = k*np.sum(non_zero_values*np.log(non_zero_values))
    return (-1)*entropy

def compute_fitness_variance(fitness, n, params, sim_params):
    if scipy.sparse.issparse(fitness) and scipy.sparse.issparse(n):
        var_f_n = np.sum(np.power(fitness.toarray(), 2)*n.toarray())
    else:
        raise NotImplementedError("Not implemented for non sparse fitness variance")
    return var_f_n*sim_params["dt"]

def compute_entropy_production_Gaussian(var, Diff_const, t, N = 1, dim = 2):
    term = 2*dim*Diff_const/(2*var + 4*Diff_const*t)
    return N*term

def compute_entropy_production_2D_mutation(n_old, params, sim_params):
    _, cov, count = fit_GMM(n_old, params, sim_params)

    eigval, _ = np.linalg.eigh(cov.squeeze())
    Diff_const = params["D"]
    entropy_mutation = compute_entropy_production_Gaussian(eigval[0], Diff_const, 1, N = count, dim = 1) \
                        + compute_entropy_production_Gaussian(eigval[1], Diff_const, 1, N = count, dim = 1)
    return entropy_mutation

def compute_entropy_Gaussian(var, Diff_const, t, N = 1, dim =2):
    size_cor = N*np.log(N) if N !=1 else 0 
    return -size_cor + N*(dim/2)*(1+np.log((2*var+4*Diff_const*t)*np.pi))

def compute_entropy_change(n_new, n_old, dt=1):
    return (compute_entropy(n_new) - compute_entropy(n_old))*dt


