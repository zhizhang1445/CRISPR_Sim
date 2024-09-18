import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import norm
from joblib import Parallel, delayed
from formulas import calc_diff_const, trail_exp
from randomHGT import get_time_next_HGT
from supMethods import sum_parallel
from fitness import derivative_fitness, find_root_fitness
from formulas import gaussian1D, semi_exact_nh

def return_v_tau(params, sim_params):
    r = params["r"]
    root_c = find_root_fitness(params, sim_params)
    A = root_c/(1-root_c)
    return r/A

def return_s(params, sim_params):
    r = params["r"]

    c_0 = find_root_fitness(params, sim_params)
    derivative_c_0 = c_0*(-1/r)
    der_fit = derivative_fitness(c_0, params, sim_params)
    return derivative_c_0*der_fit

def fill_parameters(params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    N = params["N"]
    Nh = params["Nh"]
    r = params["r"]
    A = params["A"]
    v0 = 0
    sigma = 0

    params["D"] = D = calc_diff_const(params, sim_params)
    params["s"] = s = return_s(params, sim_params)
    params["tau"] = (M*Nh/N)*(1/A)

    common_log = 24*np.log(N*np.power(D*np.power(s,2), 1/3))
    # print(f"D: {D}| s:{s}| common_log: {common_log}")
    if common_log < 0:
        raise(ValueError("INCREASE Nh YOUR EXPECTED POPULATION IS 0"))
    sigma = np.power(D/s, 1/3)*np.power(common_log, 1/6)
    v0 = np.power(s, 1/3)*np.power(D, 2/3)*np.power(common_log, 1/3)
    uc = s*np.power(sigma, 4)/(4*D)

    params["v_tau"] = return_v_tau(params, sim_params)
    params["v0"] = v0
    params["sigma"] = sigma
    params["uc"] = uc
    params["M0"] = M
    return params, sim_params

def init_cond(params, sim_params, out_print = False):
    Nh = params["Nh"]
    N0 = params["N0"]
    params["N"] = N0
    i = 0

    if sim_params["hard_N0"]:
        params, sim_params = fill_parameters(params, sim_params)
        i = 101

    while(i < 100):
        params, sim_params = fill_parameters(params, sim_params)
        N0 = params["N"]
        uc = params["uc"]
        sigma = params["sigma"]
        A = params["A"]

        if out_print:
            print(f"Phage Population: {N0:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}")
        
        if np.isnan(uc) or np.isnan(sigma):
            raise(ValueError("You need >10E6 Nh or >10E3 N0"))
        
        N = Nh*params["v0"]*params["M"]/(params["v_tau"]*A)
        params["N"] = N
        i   += 1
        if np.abs(N0-N) <= 0.5:
            params["N"] = int(N)
            params, sim_params = fill_parameters(params, sim_params)
            uc = params["uc"]
            sigma = params["sigma"]
            if out_print:
                print(f"Phage Population: {N:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}")
            break

    uc = params["uc"]
    sigma = params["sigma"]
    params["N0"] = params["N"] #update actual N
    sim_params["initial_var_n"] = sigma
    sim_params["initial_var_nh"] = np.sqrt(1.66*np.power(sigma, 2))
    sim_params["time_next_event"] = get_time_next_HGT(0, params, sim_params)

    print("Assumptions Checks: ")

    mu = params["mu"]
    print(f"mu >> 1 : mu = {mu} >> 1")

    gamma_shape = params["gamma_shape"]
    r = params["r"]
    print(f"del_x << r : gamma_shape = {gamma_shape} << r = {r}")

    v_tau = return_v_tau(params, sim_params)
    print(f"v*tau >> sigma : v*tau = {v_tau} >> sigma = {sigma}")

    print(f"uc << r : uc = {uc} << r = {r}")
    return params, sim_params

def init_guassian_n(params, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    dim = sim_params["ndim"]
    N0 = params["N"]
    num_threads = sim_params["num_threads"]

    x_linspace = np.arange(-x_range, x_range, dx)
    tt_len_x = len(x_linspace)
    
    p_marg_x = gaussian1D(x_linspace, 0, params, sim_params, prob = True)
    # p_marg_x = p_marg_x/np.sum(p_marg_x) # initial prob distribution for n: Gaussian dist

    if dim == 1:
        array = np.zeros_like(x_linspace, dtype = int)
        for i in range(N0):
            x_ind = np.random.choice(tt_len_x, p = p_marg_x)
            array[x_ind] += 1
        return array

    # 2D initialization
    tt_len_y = len(y_linspace)
    y_linspace = np.arange(-x_range, x_range, dx)
    p_marg_y = gaussian1D(x_linspace, 0, params, sim_params, direction="transverse", prob=True)
    # p_marg_y = p_marg_y/np.sum(p_marg_y) 

    iter_per_thread = np.array_split(np.arange(0, N0), num_threads)

    def add_Gaussian_noise(subset):
        array = scipy.sparse.dok_matrix((tt_len_x, tt_len_y), dtype=int)

        for i in subset:
            x_index = np.random.choice(tt_len_x, p=p_marg_x) 
            y_index = np.random.choice(tt_len_y, p=p_marg_y) 
            array[x_index, y_index]+= 1
        return array

    # out = add_Gaussian_noise(np.arange(0, N0))
    results = Parallel(n_jobs=num_threads)(delayed(add_Gaussian_noise)
            (subset) for subset in iter_per_thread)
    
    out = sum_parallel(results, num_threads)
    return out

def init_uniform(init_num, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    N = init_num
    num_threads = sim_params["num_threads"]

    x_linspace = np.arange(-x_range, x_range, dx)
    y_linspace = np.arange(-x_range, x_range, dx)
    tt_len_x = len(x_linspace)
    tt_len_y = len(y_linspace)

    iter_per_thread = np.array_split(np.arange(0, N), num_threads)

    def add_Uniform_noise(subset):
        array = scipy.sparse.dok_matrix((tt_len_x, tt_len_y), dtype=int)

        for i in subset:
            x_index = np.random.choice(tt_len_x) 
            y_index = np.random.choice(tt_len_x) 
            array[x_index, y_index] += 1
        return array

    results = Parallel(n_jobs=num_threads)(delayed(add_Uniform_noise)
            (subset) for subset in iter_per_thread)
    out = sum_parallel(results, num_threads)
    return out

def init_trail_nh(params, sim_params):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    M = params["M"]
    Nh = int(params["Nh"])
    num_threads = sim_params["num_threads"]
    tau = params["tau"]
    v0 = params["v0"]
    dim = sim_params["ndim"]

    x_linspace = np.arange(-x_range, x_range, dx)
    tt_len_x = len(x_linspace)

    p_marg_x = trail_exp(x_linspace, 0, params, sim_params, prob=True)

    if dim == 1:
        array = np.zeros_like(x_linspace, dtype=int)
        for _ in range(M*Nh):
            x_ind = np.random.choice(tt_len_x, p = p_marg_x)
            array[x_ind] += 1
        return array
    
    #2D simulations
    y_linspace = np.arange(-x_range, x_range, dx)
    tt_len_y = len(y_linspace)
    p_marg_y = gaussian1D(x_linspace, 0, params, sim_params, direction="traverse", prob = True)

    iter_per_thread = np.array_split(np.arange(0, M*Nh), num_threads)

    def add_GaussianExptail_noise(subset):
        array = scipy.sparse.dok_matrix((tt_len_x, tt_len_y), dtype=int)

        for i in subset:
            x_index = np.random.choice(tt_len_x, p=p_marg_x) 
            y_index = np.random.choice(tt_len_y, p=p_marg_y) 
            array[x_index, y_index]+= 1
        return array

    results = Parallel(n_jobs=num_threads)(delayed(add_GaussianExptail_noise)
            (subset) for subset in iter_per_thread)
    # results = add_GaussianExptail_noise(np.arrange(0, N0))
    out = sum_parallel(results, num_threads)
    return out

def init_quarter_kernel(params, sim_params, ker_type = "coverage", exponent = 1): #Kernel is not parrallel
    if ker_type == "coverage" or ker_type == "r":
        kernel = 1./params["r"]
    elif ker_type == "Boltzmann" or ker_type == "beta":
        kernel = params["beta"]
    else:
        raise NotImplementedError
    
    conv_ker_size = sim_params["conv_size"]

    x_linspace = np.arange(0, conv_ker_size, 1)
    coordmap = np.array(np.meshgrid(x_linspace, x_linspace)).squeeze()

    radius = np.sqrt(np.sum((coordmap)**2, axis=0))
    exp_radius = np.power(radius, exponent)
    matrix_ker = np.exp(-exp_radius*kernel)
    return matrix_ker

def init_full_kernel(params, sim_params, ker_type = "coverage", exponent = 1): #Kernel is all four quadrants
    if ker_type == "coverage" or ker_type == "r":
        kernel = 1./params["r"]
    elif ker_type == "Boltzmann" or ker_type == "beta":
        kernel = params["beta"]
    else:
        raise NotImplementedError

    conv_ker_size = sim_params["conv_size"]

    x_linspace = np.arange(-conv_ker_size, conv_ker_size, 1)
    coordmap = np.array(np.meshgrid(x_linspace, x_linspace)).squeeze()

    radius = np.sqrt(np.sum((coordmap)**2, axis=0))
    exp_radius = np.power(radius, exponent)
    matrix_ker = np.exp(-exp_radius*kernel)
    return matrix_ker

def init_dict_kernel(params, sim_params, ker_type = "coverage", exponent = 1):
    if ker_type == "coverage" or ker_type == "r":
        kernel = 1./params["r"]
    elif ker_type == "Boltzmann" or ker_type == "beta":
        kernel = params["beta"]
    else:
        raise NotImplementedError

    conv_ker_size = sim_params["conv_size"]

    A = np.arange(0, conv_ker_size, 1)
    A_mesh, B_mesh = np.meshgrid(A, A)

    # Flatten the meshgrid arrays
    A_flat = A_mesh.ravel()
    B_flat = B_mesh.ravel()

    mask = A_flat <= B_flat
    unique_pairs = np.column_stack([A_flat[mask], B_flat[mask]])

    kernel_dict = {norm(key): np.exp(-norm(key)*kernel) for key in unique_pairs}
    return kernel_dict

def init_1D_kernel(params, sim_params, ker_type = "coverage", exponent = 1): #Kernel is all four quadrants
    if ker_type == "coverage" or ker_type == "r":
        kernel = 1./params["r"]
    elif ker_type == "Boltzmann" or ker_type == "beta":
        kernel = params["beta"]
    else:
        raise NotImplementedError
    
    dx =sim_params["dx"]
    conv_ker_size = sim_params["conv_size"]

    x_linspace = np.arange(-conv_ker_size, conv_ker_size, dx)

    exp_radius = np.power(np.abs(x_linspace), exponent)
    kernel_1D = np.exp(-exp_radius*kernel)
    return kernel_1D