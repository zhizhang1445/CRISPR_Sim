import numpy as np
import numpy.ma as ma
import scipy
from joblib import Parallel, delayed, parallel_backend
from supMethods import timeit
from formulas import find_max_value_location

def immunity_gain_from_kernel(nh, n, kernel, params, sim_params):
    Nh = params["Nh"]
    num_threads = sim_params["num_threads"]
    conv_size = sim_params["conv_size"]
    num_to_add = np.sum(n)
    x_ind, y_ind = n.nonzero()
    support_size = len(x_ind)
    x_max, y_max  = find_max_value_location(nh)

    x_nh_sets = np.array_split(x_ind, num_threads)
    y_nh_sets = np.array_split(y_ind, num_threads)

    def generate_prob(x_ind, y_ind):
        probability = np.zeros(len(x_ind))
        for i, (x_n, y_n) in enumerate(zip(x_ind, y_ind)):
            multiplicity = n[x_n, y_n]
            # print(multiplicity)
            x_kernel = np.abs(x_n-x_max)
            y_kernel = np.abs(y_n-y_max)

            if np.any((x_kernel >= conv_size, y_kernel >= conv_size)):
                continue
            
            try:
                probability[i] = multiplicity*kernel[x_kernel, y_kernel]
                # print(multiplicity, kernel[x_kernel, y_kernel])
            except(IndexError):
                print("wtf? Immunity Convolution out of Bounds??", x_kernel, y_kernel)
                break
        return probability

    results = Parallel(n_jobs=num_threads)(delayed(generate_prob)
        (x_ind, y_ind) for x_ind, y_ind in zip(x_nh_sets, y_nh_sets))
    
    current_prob = np.concatenate(results, axis=0)
    Z_partition = np.sum(current_prob)
    
    if Z_partition == 0:
        print("No Acquisition")
        return nh
    else:
        current_prob = current_prob/Z_partition

    def add_points(itr_list):
        array = scipy.sparse.dok_matrix(nh.shape, dtype=int)
        sample_ind = np.random.choice(support_size, len(itr_list), p = current_prob)
        for i in sample_ind:
            x = x_ind[i]
            y = y_ind[i]
            array[x, y] += 1
        return array
    
    main_itr = np.arange(num_to_add)
    sub_itr_sets = np.array_split(main_itr, num_threads)
    
    results = Parallel(n_jobs=num_threads)(delayed(add_points)
        (itr) for itr in sub_itr_sets)
    
    nh_integrated = nh+np.sum(results, axis = 0)
    return nh_integrated

def immunity_loss_uniform(nh_intergrated, n, params, sim_params):
    Nh = params["Nh"]
    num_threads = sim_params["num_threads"]
    num_to_remove = int(np.sum(nh_intergrated) - Nh)

    nonzero_indices = np.transpose(nh_intergrated.nonzero())
    nonzero_indices_subset = np.array_split(nonzero_indices, 
                                            num_threads, axis=0)
    nonzero_values = nh_intergrated[nonzero_indices[:, 0], 
                                    nonzero_indices[:, 1]].toarray().squeeze()
    nonzero_values_subset = np.array_split(nonzero_values, 
                                           num_threads, axis=0)

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
        array = scipy.sparse.dok_matrix(nh_intergrated.shape, dtype=int)
        sample_ind = [index_nonzero_w_repeats[i] for i in flat_index]
        for x,y in sample_ind:
            array[x, y] -= 1

        return array

    results = Parallel(n_jobs=num_threads)(
        delayed(remove_points)(flat_index) for flat_index in ind_per_thread_list)
    nh_new = nh_intergrated + np.sum(results, axis=0)

    if np.sum(nh_new) != Nh:
        raise ValueError("bacteria died/reproduced at immunity gain, Nh = ", np.sum(nh))
    
    if(scipy.sparse.issparse(nh_new)):
            min_val = np.min(nh_new.tocoo()) 
    else: min_val = np.min(nh_new)

    if min_val < 0:
        raise ValueError("bacteria population is negative")

    return nh_new