import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
from supMethods import timeit

def square_split(array, split_size): #Find smallest matrix possible and make it I'm thinking 100x100
    if np.ndim(array) == 1:
        return np.split(array, len(array)//split_size)
    elif np.ndim(array) > 2:
        raise IndexError("2D or 1D plz")

    def getIndexes(subarrays, axis):
        sizes = [subarray.shape[axis] for subarray 
                        in subarrays]
        index = 0
        indexes = [0]
        for size in sizes[:-1]:
            index = index+size
            indexes.append(index)
        return indexes
    
    subarrays = []
    start_indices = []

    col_subarrays = np.array_split(array, array.shape[0]//split_size, axis=0)
    col_indexes = getIndexes(col_subarrays, 0)

    for subarray, col_index in zip(col_subarrays, col_indexes):
        row_subarrays = np.array_split(subarray, array.shape[1]//split_size, axis=1)
        subarrays.extend(row_subarrays)
        row_indexes = getIndexes(row_subarrays, 1)
        for row_index in row_indexes:
            start_indices.append((col_index, row_index))

    # Create a list of starting indices for each subarray
    return subarrays, start_indices

def split_coverage(nh, n, kernel, params, sim_params): #TODO This is already parallel, SPARSE THIS
    num_cores = sim_params["num_threads"]
    input_data = nh/params["Nh"]
    if scipy.sparse.issparse(nh):
        nh = nh.toarray()

    def masked_array(subarray, loc_i): 
        subarray_shape = subarray.shape
        array_masked = np.zeros_like(input_data)

        x_loc, y_loc = loc_i
        array_masked[x_loc:x_loc+subarray_shape[0], 
                     y_loc:y_loc+subarray_shape[1]] = subarray

        return array_masked
    
    def convolve_subset(input_data_subset, start_index):
        if np.sum(input_data_subset) == 0:
            return np.array([])
        
        else:
            masked_input = masked_array(input_data_subset, start_index)
            return scipy.signal.convolve2d(masked_input, kernel, mode='same')

    input_data_subsets, start_indexes = square_split(input_data, 32)

    # results = Parallel(n_jobs=num_cores, backend="multiprocessing")
    results = Parallel(n_jobs=num_cores)(delayed(convolve_subset)(subset, index) for subset, index 
                in zip(input_data_subsets, start_indexes))

    # output = np.sum(results, axis=0)
    x_ind, y_ind = np.nonzero(n)
    output = scipy.sparse.dok_matrix(input_data.shape, dtype=float)
    for result in results:
        if result.size == 0:
            continue
        else:
            output[x_ind, y_ind] += result[x_ind, y_ind]

    return output/params["M"]

def elementwise_coverage(nh, n, kernel_quarter, params, sim_params):
    conv_size = sim_params["conv_size"]
    Nh = params["Nh"]
    M = params["M"]
    num_threads = sim_params["num_threads"]

    x_ind_nh, y_ind_nh = nh.nonzero()
    x_ind_n, y_ind_n = n.nonzero()

    x_nh_sets = np.array_split(x_ind_nh, num_threads)
    y_nh_sets = np.array_split(y_ind_nh, num_threads)

    input_h = np.divide(nh, Nh)

    def convolve_subset(x_ind_nh, y_ind_nh):
        res = scipy.sparse.dok_matrix(nh.shape, dtype=float)
        for x_nh, y_nh in zip(x_ind_nh, y_ind_nh):
            value = input_h[x_nh, y_nh]

            for x_n, y_n in zip(x_ind_n, y_ind_n):

                x_kernel = np.abs(x_nh-x_n)
                y_kernel = np.abs(y_nh-y_n)

                if np.any((x_kernel >= conv_size, y_kernel >= conv_size)):
                    continue
                
                try:
                    interaction = kernel_quarter[x_kernel, y_kernel]
                except(IndexError):
                    print("wtf? Convolution out of Bounds??", x_kernel, y_kernel)
                    break

                res[x_n, y_n] += value*interaction
        return res

    results = Parallel(n_jobs=num_threads)(delayed(convolve_subset)
        (x_ind_nh, y_ind_nh) 
            for x_ind_nh, y_ind_nh
                in zip(x_nh_sets, y_nh_sets))
    
    out = np.sum(results, axis=0)
    return out/M