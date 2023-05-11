import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
from supMethods import timeit

@timeit
def immunity_update(nh, n, params, sim_params):
    Nh = params["Nh"]
    N = np.sum(n)
    num_threads = sim_params["num_threads"]
    nh = nh + n
    num_to_remove = np.sum(nh) - Nh

    nonzero_indices = np.nonzero(nh)
    nonzero_values = [nh[index] for index in zip(*nonzero_indices)]
    index_nonzero_w_repeats = []
    for value, index in zip(nonzero_values, zip(*nonzero_indices)):
        for i in range(int(value)):
            index_nonzero_w_repeats.append(index)

    sample_flat_ind = np.random.choice(len(index_nonzero_w_repeats), num_to_remove,replace = False)

    ind_per_thread_list = np.array_split(sample_flat_ind, num_threads)

    def remove_points(flat_index):
        array = scipy.sparse.dok_matrix(nh.shape, dtype=int)
        sample_ind = [index_nonzero_w_repeats[i] for i in flat_index]
        for x,y in sample_ind:
            array[x, y] -= 1

        return array

    results = Parallel(n_jobs=num_threads)(
        delayed(remove_points)(flat_index) for flat_index in ind_per_thread_list)
    nh = nh + np.sum(results, axis=0)

    if np.sum(nh) != Nh:
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh))
    
    min_val = np.min(nh.tocoo()) if (scipy.sparse.issparse(nh)) else np.min(nh)

    if min_val < 0:
        raise ValueError("bacteria population is negative")

    return nh