import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
from joblib import Parallel, delayed
from formulas import calc_diff_const

def fill_parameters(params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    N = params["N"]
    Nh = params["Nh"]
    r = params["r"]
    v0 = 0
    sigma = 0

    params["D"] = D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    params["s"] = s = M*inv_v_tau
    params["tau"] = tau = M*Nh/N

    common_log = 24*np.log(N*np.power(D*np.power(s,2), 1/3))
    sigma = np.power(D/s, 1/3)*np.power(common_log, 1/6)
    v0 = np.power(s, 1/3)*np.power(D, 2/3)*np.power(common_log, 1/3)
    uc = s*np.power(sigma, 4)/(4*D)

    params["v0"] = v0
    params["sigma"] = sigma
    params["uc"] = uc    
    return params, sim_params

def init_cond(params, sim_params, out_print = False):
    Nh = params["Nh"]
    N0 = params["N0"]
    params["N"] = N0

    for i in range(100):
        params, sim_params = fill_parameters(params, sim_params)
        N0 = params["N"]
        uc = params["uc"]
        sigma = params["sigma"]
        if out_print:
            print(f"Phage Population: {N0:.4f}| Uc: {uc:.4f}| sigma: {sigma:.4f}")
        
        if np.isnan(uc) or np.isnan(sigma):
            raise(ValueError("You need >10E6 Nh or >10E3 N0"))
        
        N = Nh*(params["s"]*params["v0"])
        params["N"] = N
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
    sim_params["initial_var_n"] = sigma
    sim_params["initial_var_nh"] = np.sqrt(np.power(sigma, 2) + np.power(uc, 2))
    return params, sim_params

def init_guassian(init_num, sim_params, type = "n"):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    N0 = init_num
    initial_position = sim_params["initial_mean_"+type]
    initial_var = sim_params["initial_var_"+type]
    num_threads = sim_params["num_threads"]

    x_linspace = np.arange(-x_range+initial_position[0], x_range+initial_position[0], dx)
    y_linspace = np.arange(-x_range+initial_position[1], x_range+initial_position[1], dx)
    tt_len_x = len(x_linspace)
    tt_len_y = len(y_linspace)
    
    p_marg_x = np.exp(-x_linspace**2/(2*(initial_var**2)))
    p_marg_x = p_marg_x/np.sum(p_marg_x) # initial prob distribution for n: Gaussian dist

    p_marg_y = np.exp(-y_linspace**2/(2*(initial_var**2)))
    p_marg_y = p_marg_y/np.sum(p_marg_y) 

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
    out = np.sum(results, axis=0)

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
    out = np.sum(results, axis=0)

    return out

def init_exptail(init_num, params, sim_params, type = "nh"):
    x_range = sim_params["xdomain"] #Initialize the spaces
    dx = sim_params["dx"]
    M = params["M"]
    N0 = init_num
    initial_position = sim_params["initial_mean_"+type]
    initial_var = sim_params["initial_var_"+type]
    num_threads = sim_params["num_threads"]
    tau = params["tau"]
    v0 = params["v0"]
    axis = [1, 1]

    x_linspace = np.arange(-x_range+initial_position[0], 
                           x_range+initial_position[0], dx)
    y_linspace = np.arange(-x_range+initial_position[1], 
                           x_range+initial_position[1], dx)
    tt_len_x = len(x_linspace)
    tt_len_y = len(y_linspace)

    p_marg_x = np.exp(-x_linspace**2/(2*(initial_var**2)))
    p_marg_y = np.exp(-y_linspace**2/(2*(initial_var**2)))

    if axis[0] == 0:
        const = M/(v0*tau)
        p_marg_x = const*np.exp(-np.abs(x_linspace-initial_position[0])/(
            v0*tau))*np.heaviside(axis[1]*(x_linspace-initial_position[0]), 0)
        if np.sum(p_marg_x) < 0.99:
            print(f"you should consider increasingxdomain: {np.sum(p_marg_x)}")

    if axis[0] == 1:
        const = M/(v0*tau)
        p_marg_y = const*np.exp(-np.abs(y_linspace-initial_position[1])/(
            v0*tau))*np.heaviside(axis[1]*(y_linspace-initial_position[1]), 0)
        if np.sum(p_marg_y) < 0.99:
            print(f"you should consider increasing ydomain: {np.sum(p_marg_y)}")

    p_marg_y = p_marg_y/np.sum(p_marg_y) 
    p_marg_x = p_marg_x/np.sum(p_marg_x)
    iter_per_thread = np.array_split(np.arange(0, N0), num_threads)

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
    out = np.sum(results, axis=0)

    return out

def init_quarter_kernel(params, sim_params, type = "Radius", exponent = 1): #Kernel is not parrallel
    if type == "Radius" or type == "r":
        kernel = 1/params["r"]
    elif type == "Boltzmann" or type == "beta":
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

def init_full_kernel(params, sim_params, type = "coverage", exponent = 1): #Kernel is all four quadrants
    if type == "coverage":
        kernel = 1./params["r"]
    elif type == "Boltzmann":
        kernel = params["beta"]
    else:
        raise NotImplementedError
    
    kernel = params["r"]
    conv_ker_size = sim_params["conv_size"]

    x_linspace = np.arange(-conv_ker_size, conv_ker_size, 1)
    coordmap = np.array(np.meshgrid(x_linspace, x_linspace)).squeeze()

    radius = np.sqrt(np.sum((coordmap)**2, axis=0))
    exp_radius = np.power(radius, exponent)
    matrix_ker = np.exp(-exp_radius*kernel)
    return matrix_ker