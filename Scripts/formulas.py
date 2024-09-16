import numpy as np
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy
from supMethods import minmax_norm

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

def p_infection(p_coverage, M, Np, dc):
    p_infection = (1-p_coverage)**M

    for n in range(1, M+1):
        p_n_spacer = binomial_pdf(M, n, p_coverage)
        for d in range(0, dc+1):
            p_infection += binomial_pdf(Np, d, n/M)*p_n_spacer
    return p_infection

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

def p_infection(p_dense, params, sim_params):
    M = params["M"]
    Np = params["Np"]
    dc = params["dc"]
    n_order_spacer = params["n_spacer"]
    if n_order_spacer > M:
        n_order_spacer = M
    
    p_infection = 0
    for n in range(0, n_order_spacer+1):
        p_n_spacer = binomial_pdf(M, n, p_dense)
        for d in range(0, dc+1):
            p_infection += binomial_pdf(Np, d, n/M)*p_n_spacer
    return p_infection

def gaussian1D(x, t, params, sim_params, mu = 0, direction = "parallel", prob = False):
    if direction == "parallel":
        sigma = params["sigma"]
    else:
        sigma = np.sqrt(1.66)*params["sigma"]

    v0 = params["v0"]
    N = params["N"]
    
    A = (N / (sigma * np.sqrt(2 * np.pi))) 
    res = A * np.exp(-0.5 * ((x - mu - v0*t) / sigma) ** 2)

    if prob:
        return res/np.sum(res)
    else:
        return res

def trail_exp(x, t, params, sim_params, prob = False):
    A = params["A"]
    tau = params["tau"]
    N = params["N"]
    v = params["v0"]
    B = (v*t - x)/(v*tau)

    exp1 = np.exp(-1*B)
    heaviside = np.heaviside(v*t - x, 1)
    res = A*exp1*heaviside*(N/v)
    if prob:
        return res/np.sum(res)
    else:
        return res

def semi_exact_nh(x, t, params, sim_params):
    def n(x, t):
        return gaussian1D(x, t, params, sim_params, 0)
    
    def memory_ker(x, t):
        tau = params["tau"]
        return np.exp(-(t)/tau)
    
    A = params["A"]
    t_prime_low = t - params["tau"]*50
    dt_prime = 0.1
    t_prime_range = np.arange(t_prime_low, t, dt_prime)

    res = np.zeros_like(x)
    
    for t_prime in t_prime_range:
        res += A*n(x, t_prime)*memory_ker(x, t-t_prime)*dt_prime
    return res

def plot_wave_profiles(params, sim_params, ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    sigma_n = params["sigma"]

    x = np.arange(-6*sigma_n, 6*sigma_n, 0.1)
    ax.plot(x, gaussian1D(x, 1, params, sim_params), label = "Phage Population")
    ax.plot(x, trail_exp(x, 0, params, sim_params), label = "Projectile Aproximation")
    ax.plot(x, semi_exact_nh(x, 0, params, sim_params), label= "Exact Solution")

    ymax = max(trail_exp(x, 0, params, sim_params))
    ax.axvline([params["uc"]], 0,ymax, linestyle="--", color ='k', label = "Fittest Individual")
    ax.set_title("Traveling Wave Profiles")
    ax.set_xlabel("Antigenic Distance")
    ax.set_ylabel("Occupancy")
    ax.legend()

def theoretical_c(x,t, params, sim_params, direction = "Front", translation = 0):
    tau = params["tau"]
    v = params["v0"]
    r = params["r"]
    x = x-translation-v*t
    
    A = 1/(1+(v*tau/r))
    B = 1/(1-(v*tau/r))

    if direction == "Front":
        return A*np.exp(-x/r)
    
    if direction == "Back":
        exp1 = np.exp(-np.abs(x)/(v*tau))
        exp2 = np.exp(-np.abs(x)/r) - np.exp(-np.abs(x)/(v*tau))
        return A*exp1 + B*exp2
    
    if direction == "Full":
        front = A*np.exp(-x/r)*np.heaviside(x, 1)
        exp1 = np.exp(-np.abs(x)/(v*tau))
        exp2 = np.exp(-np.abs(x)/r) - np.exp(-np.abs(x)/(v*tau))
        back = (A*exp1 + B*exp2)*np.heaviside(-x, 1)
        return front+back

def semi_true_c(x, t, params, sim_params, how_true = "trail"):
    def nh(x, t):
        if how_true == "trail":
            return trail_exp(x, t, params, sim_params)
        return semi_exact_nh(x, t, params, sim_params)

    def coverage_ker(x,t):
        r = params["r"]
        return np.exp(-1*np.abs(x)/r)
    
    v = params["v0"]
    tau = params["tau"]
    x_prime_low = v*t - 10*params["tau"]*v
    dx_prime = 0.1
    x_prime_range = np.arange(x_prime_low, -1*x_prime_low, dx_prime)

    res = np.zeros_like(x)
    
    for x_prime in x_prime_range:
        res += (nh(x_prime, t)*coverage_ker(x-x_prime, t)*dx_prime)
        # print(np.sum(res))

    norm = params["M"]*params["Nh"]
    return res/norm

def plot_wave_coverage(params, sim_params, ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    sigma_n = params["sigma"]

    x= np.arange(-20*sigma_n, 6*sigma_n, 0.1)
    ax.plot(x, gaussian1D(x, 0, params,sim_params)/np.sum(params["N"]))
    ax.plot(x, trail_exp(x, 0, params,sim_params)/np.sum(params["Nh"]))
    ax.plot(x, theoretical_c(x, 0, params, sim_params, direction="Full"), label = "Proj. Approx")
    ax.plot(x, semi_true_c(x, 0, params, sim_params), label = "Num. Approx")
    # plt.plot(x, semi_true_c(x, 0, params, sim_params, "trail"), label = "Approx c")
    ax.axvline([0], 0, np.max(theoretical_c(x, 0, params, sim_params)), linestyle = "--", color = "k", label = "Wave Center")
    ax.set_title("Traveling Wave Coverage")
    ax.set_xlabel("Antigenic Distance")
    ax.set_ylabel("Coverage Probabiilty")
    ax.legend()


def fitness_simple_M(p_coverage, params, sim_params, M = None):
    if M is None:
        M = params["M"]
        
    def p_inf_M(M):
        params_temp = deepcopy(params)
        params_temp["M"] = M
        return p_infection(p_coverage, params_temp, sim_params)
    
    R0 = params["R0"]
    p_inf_M_vec = np.vectorize(p_inf_M)
    p_inf = p_inf_M_vec(M)
    return np.log(R0*p_inf)

def fitness_linear_approx(x, t, params, sim_params):
    s = params["s"]
    v = params["v0"]
    u = x-v*t
    return s*u

def plot_wave_fitness(params, sim_params, ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    sigma_n = params["sigma"]
    x= np.arange(-3*sigma_n, 3*sigma_n, 0.1)
    ax.plot(x, minmax_norm(gaussian1D(x, 0, params,sim_params)))
    # plt.plot(x, minmax_norm(trail_exp(x, 0, params,sim_params)))

    c = theoretical_c(x, 0, params, sim_params, direction="Full")
    # plt.plot(x, c, label = "Proj. Approx")
    ax.plot(x, fitness_linear_approx(x, 0, params, sim_params), color = "red", linestyle = "--",label = "Linear Approx")

    ax.plot(x, fitness_simple_M(c, params, sim_params), color = "red", label = "True Fitness")

    # plt.plot(x, fitness_simple_M(c, params, sim_params, M ), color = "orange", label = "M = 20")
    ax.axhline(0, np.min(x), np.max(x), linestyle= "--", color = "k")
    ax.set_title("Fitness Profile")
    ax.legend()

def plot_fitness_memory_dynamics(params, sim_params, ax = None):

    def weighted_avg_and_var(f, n):
    # Compute weighted average
        weighted_avg = np.sum(f * n) / np.sum(n)
        
        # Compute weighted standard error
        variance = np.sum(n * (f - weighted_avg)**2) / np.sum(n)  # Weighted variance
        weighted_var = np.sqrt(variance)
        
        return weighted_avg, weighted_var
    
    sigma_n = params["sigma"]
    if ax is None:
        fig, ax = plt.subplots()
    x = np.arange(-3*sigma_n, 3*sigma_n, 0.1)
    M_range = np.arange(1, 20, 1)

    n = gaussian1D(x, 0, params, sim_params)
    ind = np.where(n>=1)
    c = theoretical_c(x, 0, params, sim_params)
    c_restricted = c[ind]

    f = fitness_simple_M(c_restricted, params, sim_params)
    n_restricted = n[ind]
    avg_f, avg_var = weighted_avg_and_var(f, n_restricted)
    ax.errorbar(params["M"], avg_f,yerr= avg_var)
    ax.scatter(params["M"], avg_f)

    uc = params["uc"]
    c_uc = theoretical_c(uc, 0, params, sim_params)
    avg_fitness = []
    err_fitness = []

    for M in M_range:
        f = fitness_simple_M(c_restricted, params, sim_params, M)
        avg_f, avg_var = weighted_avg_and_var(f, n_restricted)
        avg_fitness.append(avg_f)
        err_fitness.append(avg_var)

    ax.scatter(M_range, avg_fitness, label = "Average Fitness")
    ax.errorbar(M_range, avg_fitness, yerr = err_fitness)
    ax.set_xticks([0, 5, 10, 15, 20])

    ax.plot(M_range, fitness_simple_M(c_uc, params, sim_params, M_range), label = "Most Fit Indivdual")

    ax.set_title("Fitness Distribution")
    ax.set_xlabel("Memory Size")
    ax.set_ylabel("Fitness")
    ax.legend()

def pred_value(params, sim_params):
    erf = scipy.special.erf
    var_nh = sim_params["initial_var_nh"]
    r = params["r"]
    const = np.exp(-var_nh**2/(2*np.power(r,2)))
    neg_exp = lambda x: (1/2)*np.exp(-x/r)
    pos_exp = lambda x: (1/2)*np.exp(x/r)

    div_const = 1/(var_nh*np.sqrt(2))
    erf_mean = var_nh**2/r
    neg_erf = lambda x: erf(div_const*(x-erf_mean))
    pos_erf = lambda x: erf(div_const*(x+erf_mean))

    def anal_result(x):
        return const*(neg_exp(x)*(neg_erf(x)+1)-pos_exp(x)*(pos_erf(x)-1))
    
    def anal_delay(x):
        return anal_result(x-erf_mean)
    return anal_delay
    