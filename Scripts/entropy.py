import numpy as np
import numpy.ma as ma
import scipy
from joblib import Parallel, delayed, parallel_backend
from supMethods import timeit
from formulas import find_max_value_location

def compute_entropy(array, k=1):
    x_ind, y_ind = np.nonzero(array)
    non_zero_values = array[x_ind, y_ind].toarray().squeeze()
    entropy = k*np.sum(non_zero_values*np.log(non_zero_values))
    return (-1)*entropy

def compute_fitness_variance(fitness, n, params):
    var_f_n = np.sum(np.power(fitness, 2).multiply(n))/np.sum(n)
    return var_f_n

def entropy_production_mutation(stddev, Diff_const, t):
    term = 4*Diff_const/(2*stddev + 4*Diff_const*t)
    return term

def compute_entropy_change(n_new, n_old, dt=1):
    return (compute_entropy(n_new) - compute_entropy(n_old))*dt


