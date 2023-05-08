import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import convolve
from scipy import signal
from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from concurrent.futures import as_completed
import scipy

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
  
def coverage_parrallel_convolution(nh, n, kernel, params, sim_params): #TODO This is already parallel, SPARSE THIS
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
            return np.nan
        
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
        if np.isnan(result):
            continue
        else:
            output[x_ind, y_ind] += result[x_ind, y_ind]

    return output/params["M"]

def coverage_sparse_parrallel(nh, n, kernel_quarter, params, sim_params):
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
    # out = convolve_subset()
    return out/M

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

def fitness_spacers_parallel(n, nh, p_sparse, params, sim_params): #TODO PARALLELIZE THIS
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

def virus_growth_parallel(n, f_sparse, params, sim_params): #TODO PARALLELIZE THIS
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

def num_mutants_parallel(n, params, sim_params): #TODO PARALLELIZE THIS
    mu = params["mu"]
    dt = sim_params["dt"]

    x_ind, y_ind = np.nonzero(n)
    map = np.zeros(n.shape, dtype=np.int16)

    p = 1-np.exp(-1*mu*dt)
    if scipy.sparse.issparse(n):
        map[x_ind, y_ind] = np.random.binomial(n[x_ind, y_ind].todense(), p) #   so this is really prob of finding k mutation in n possible virus with prob p in 1-e^-mudt
    else:
        map[x_ind, y_ind] = np.random.binomial(n[x_ind, y_ind], p)
    
    return map
                                    #   so p to not mutated is really e^-mu*dt

def num_mutation(params, sim_params): #Sparse is not needed
    mu = params["mu"]
    dt = sim_params["dt"]

    out = np.random.poisson(mu*dt)
    if out >= 1 :
        return out #This is what mu, is the average rate of mutation
    else:
        try:
            return num_mutation(params, sim_params) #conditioned as to have at least one mutation
        except RecursionError:
            return 0

def mutation_jump(m, params, sim_params):
    shape_param = params["gamma_shape"]
    dx = sim_params["dx"]

    jump = np.zeros(2)
    mean = 2*dx
    theta = mean/shape_param
    
    for i in range(m):
        angle = np.random.uniform(0, 2*np.pi) #Goodammit couldn't this have been clearer??? It's supposed to be isotropic
        jump = jump + np.random.gamma(shape_param, theta)*np.array([np.cos(angle), np.sin(angle)])
        #The distribution of jump is a sum of gamma distribution. 

    jump = np.round(jump)
    return jump

def immunity_update_parallel(nh, n, params, sim_params):
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

    ind_per_thread_list = np.split(sample_flat_ind, num_threads)

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

def mutation_parallel(n, params, sim_params):
    num_threads = sim_params["num_threads"]
    checksum = np.sum(n)

    mutation_map = num_mutants_parallel(n, params, sim_params) # The mutation maps tells you how many virus have mutated at each location
    x_ind, y_ind = np.nonzero(mutation_map) #finding out where the mutations happends

    x_ind_subsets = np.array_split(x_ind, num_threads)
    y_ind_subsets = np.array_split(y_ind, num_threads)

    def mutation_single(x_ind, y_ind):
        n_to_add = scipy.sparse.dok_matrix(n.shape, dtype=np.int64)
        for x_i, y_i in zip(x_ind, y_ind):
            num_mutants_at_site = mutation_map[x_i, y_i] # unpacking the number of mutated virus
            n_to_add[x_i, y_i] -= num_mutants_at_site

            for j in range(num_mutants_at_site): #Find out where those virus have moved to
                num_mutation_at_site = num_mutation(params, sim_params) #Sampling how many single mutation for a single virus (num of jumps) 
                jump = mutation_jump(num_mutation_at_site, params, sim_params) #Sampling the jump

                try:
                    new_x_loc = (x_i + jump[0]).astype(int)
                    new_y_loc = (y_i + jump[1]).astype(int)
                    n_to_add[new_x_loc, new_y_loc] += 1
                except IndexError: #Array Out of Bounds
                    if new_x_loc >= n.shape[0]:
                        new_x_loc = -1 #lmao this is gonna be a pain in cpp
                    if new_y_loc >= n.shape[1]:
                        new_y_loc = -1
                    n_to_add[new_x_loc, new_y_loc] += 1
        return n_to_add

    results = Parallel(n_jobs=num_threads)(delayed(mutation_single)(x_ind, y_ind) for x_ind, y_ind in zip(x_ind_subsets, y_ind_subsets))
    # results = mutation_single(n_to_add, x_ind, y_ind)

    n = n+np.sum(results, axis = 0)
    if checksum != np.sum(n):
        raise ValueError("Cries cuz Bacteria died during mutation")
    return n