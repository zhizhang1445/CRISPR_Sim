import numpy as np
import scipy
import matplotlib.pyplot as plt

def find_max_value_location(matrix):
    max_value = float('-inf')
    max_row, max_col = None, None

    for (row, col), value in matrix.items():
        if value > max_value:
            max_value = value
            max_row = row
            max_col = col

    return max_row, max_col

def find_mean_location(matrix):
    mean_row = 0
    mean_col = 0
    tt_value = 0

    for (row, col), value in matrix.items():
        mean_row += row*value
        mean_col += col*value
        tt_value += value
    
    mean_row = mean_row/tt_value
    mean_col = mean_col/tt_value

    return mean_row, mean_col

def compute_shift(nh, nh_prev, type = "max"):

    if type == "max":
        x_old, y_old= find_max_value_location(nh_prev)
        x_new, y_new= find_max_value_location(nh)
    elif type == "mean":
        x_old, y_old = find_mean_location(nh_prev)
        x_new, y_new = find_mean_location(nh)

    shift_x = np.rint(x_new - x_old).astype(int)
    shift_y = np.rint(y_new - y_old).astype(int)
    return shift_x, shift_y

def calculate_velocity(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = 24*np.log(N*np.power(D*np.power(s, 2), 1/3))
    v = np.power(s, 1/3)*np.power(D, 2/3)*np.power(common_log, 1/3)
    return v

def calculate_DlnND(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = np.log(N*np.power(D*np.power(s, 2), 1/3))
    v = np.power(D, 2/3)*np.power(common_log, 1/3)
    return v

def calculate_var(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = np.log(N*np.power(D*np.power(s, 2), 1/3))
    var = np.power(D/s, 1/3)*np.power(common_log, 1/6)
    return var

def calculate_FisherVelocity(N, params, sim_params):
    R0 = params["R0"]
    M = params["M"]
    r = params["r"]

    D = calc_diff_const(params, sim_params)
    inv_v_tau = (np.power(R0, 1/M)-1)/r
    s = M*inv_v_tau

    common_log = np.log(N*np.power(D*np.power(s, 2), 1/3))
    uc = (1/4)*np.power(D/s, 1/3)*(np.power(common_log, 2/3))
    v = 2*np.sqrt(s*uc*D)
    return v

def running_median_filter(signal, window_size, padding = 'symmetric'):
    pad_width = window_size // 2
    padded_signal = np.pad(signal, pad_width, mode=padding)
    filtered_signal = scipy.signal.medfilt(padded_signal, kernel_size=window_size)
    trimmed_signal = filtered_signal[pad_width:-pad_width]
    return trimmed_signal


def average_of_pairs(arr):
    averages = []
    for i in range(0, len(arr)-1, 1):
        average = (arr[i] + arr[i+1]) / 2.0
        averages.append(average)
    return np.array(averages)

def calc_diff_const(params, sim_params):
    dx = sim_params["dx"]
    shape = params["gamma_shape"]
    mu = params["mu"]

    mean = 2*dx
    scale = mean/shape
    gamma_var = shape*(scale**2)
    cos_uni_var = 1/2
    prod_var = (mean**2 + gamma_var)*(cos_uni_var)

    diff_const = mu*prod_var/2
    return diff_const


def guassian_diffusion(xspace, yspace, t, params, sim_params, print_flag = False):
    n_var = sim_params["initial_var_n"]
    diff_const = calc_diff_const(params, sim_params)
    a = 1/(2*n_var**2)
    b = 1/(4*diff_const*t)

    c = (1/a +1/b)

    coordmap = np.array(np.meshgrid(xspace, yspace)).squeeze()
    rsqrd = (coordmap[0]**2 + coordmap[1]**2)

    func = np.exp(-rsqrd/c)
    func = (func/np.sum(func))
    if print_flag:
        print("Diffusion Constant: ", diff_const)
        print("Current Normal Variance: ", np.sqrt(c/2))
    return func