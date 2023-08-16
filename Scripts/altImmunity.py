import numpy as np
import numpy.ma as ma
import scipy
from joblib import Parallel, delayed, parallel_backend
from supMethods import timeit
from formulas import find_max_value_location

@timeit
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

@timeit
def immunity_loss_uniform(nh, n, params, sim_params):
    Nh = params["Nh"]
    num_threads = sim_params["num_threads"]

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