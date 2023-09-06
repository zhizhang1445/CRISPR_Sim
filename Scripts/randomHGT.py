import numpy as np
import numpy.ma as ma
import scipy
from joblib import Parallel, delayed, parallel_backend
from supMethods import timeit
from formulas import find_max_value_location

def get_time_next_HGT(current_t, params, sim_params):
    rate_event = params["rate_HGT"]
    
    scale_param = 1/rate_event
    next_time_reaction = current_t + np.random.exponential(scale_param, 1)
    return next_time_reaction
 