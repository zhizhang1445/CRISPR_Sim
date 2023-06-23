import numpy as np
import numpy.ma as ma
import scipy
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
from supMethods import timeit
from formulas import find_max_value_location


@timeit
def immunity_update(nh, n, params, sim_params):
    Nh = params["Nh"]
    num_threads = sim_params["num_threads"]
    nh = nh + n
    total_number = np.sum(nh)
    num_to_remove = int(total_number - Nh)

    nonzero_indices = np.transpose(nh.nonzero())
    nonzero_indices_subset = np.array_split(nonzero_indices, num_threads, axis=0)
    nonzero_values = nh[nonzero_indices[:, 0], nonzero_indices[:, 1]].toarray().squeeze()
    nonzero_values_subset = np.array_split(nonzero_values, num_threads, axis=0)

    def process_value(values, indexes):
        index_nonzero_w_repeats = []
        for value, index in zip(values, indexes):
            index_nonzero_w_repeats.extend([index for _ in range(int(value))])
        return index_nonzero_w_repeats

    set_index_w_repeats = Parallel(n_jobs=num_threads)(delayed(process_value)(values, indexes)
                                for values, indexes in zip(nonzero_values_subset, nonzero_indices_subset))

    set_num_to_remove = [int(num_to_remove*(len(set)/total_number)) 
                         for set in set_index_w_repeats]
    set_num_to_remove_ex = int(num_to_remove - np.sum(set_num_to_remove))

    if set_num_to_remove_ex < 0:
        raise ValueError("Fuck why is set_num_to_remove_ex negative")
    else:
        thread_num = np.random.choice(num_threads, set_num_to_remove_ex)
        for i in thread_num:
            set_num_to_remove[i] += 1

    def remove_points(sub_index_w_repeats, sub_num_to_remove):
        array = scipy.sparse.dok_matrix(nh.shape, dtype=int)
        if sub_num_to_remove == 0:
            return array
        
        sampled_flat_ind = np.random.choice(len(sub_index_w_repeats), 
                                            sub_num_to_remove,replace = False)
        for i in sampled_flat_ind:
            x, y = sub_index_w_repeats[i]
            array[x, y] -= 1
        return array

    results = Parallel(n_jobs=num_threads)(
        delayed(remove_points)(sub_index_w_repeats, sub_num_to_remove) 
            for sub_index_w_repeats, sub_num_to_remove 
            in zip(set_index_w_repeats, set_num_to_remove))
    nh = nh + np.sum(results, axis=0)

    if np.sum(nh) != Nh:
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh))
    
    min_val = np.min(nh.tocoo()) if (scipy.sparse.issparse(nh)) else np.min(nh)

    if min_val < 0:
        raise ValueError("bacteria population is negative")

    return nh

@timeit
def immunity_mean_field(nh, n, params, sim_params):
    Nh = params["Nh"]
    nh = nh + n
    total_number = np.sum(nh)
    num_to_remove = int(total_number - Nh)
    ratio = 1-(num_to_remove/total_number)

    nh_new = scipy.sparse.dok_matrix(nh.shape)
    for (row, col), value in nh.items():
        nh_new[row, col] = int(np.rint(value*ratio))

    new_tt_number = np.sum(nh_new)
    error = int(Nh - new_tt_number)
    # print(error)
    
    x_max, y_max = find_max_value_location(nh)
    nh_new[x_max, y_max] += error


    if np.sum(nh_new) != Nh:
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh))
    
    min_val = np.min(nh_new.tocoo()) if (scipy.sparse.issparse(nh_new)) else np.min(nh_new)

    if min_val < 0:
        raise ValueError("bacteria population is negative")

    return nh

@timeit
def immunity_update_SerialChoice(nh, n, params, sim_params):
    Nh = params["Nh"]
    num_threads = sim_params["num_threads"]
    nh = nh + n
    num_to_remove = int(np.sum(nh) - Nh)

    nonzero_indices = np.transpose(nh.nonzero())
    nonzero_indices_subset = np.array_split(nonzero_indices, num_threads, axis=0)
    nonzero_values = nh[nonzero_indices[:, 0], nonzero_indices[:, 1]].toarray().squeeze()
    nonzero_values_subset = np.array_split(nonzero_values, num_threads, axis=0)

    def process_value(values, indexes):
        index_nonzero_w_repeats = []
        for value, index in zip(values, indexes):
            index_nonzero_w_repeats.extend([index for _ in range(int(value))])
        return index_nonzero_w_repeats

    results = Parallel(n_jobs=num_threads)(delayed(process_value)(values, indexes)
                                for values, indexes in zip(nonzero_values_subset, nonzero_indices_subset))

    index_nonzero_w_repeats = []
    for sublist in results:
        index_nonzero_w_repeats.extend(sublist)

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