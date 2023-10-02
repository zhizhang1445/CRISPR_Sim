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

def HGT_logistic_event(t, n, params, sim_params, toprint = False):
    rate_recovery = params["rate_recovery"]
    time2nextevent = sim_params["time_next_event"]

    if t > time2nextevent:
        if toprint: print("HGT Occurs at time:", t)
        num_to_add = np.sum(n) + params["HGT_bonus_acquisition"]
        num_to_remove = np.sum(n)
        params["M"] = params["M"] + params["HGT_bonus_acquisition"]/params["Nh"]

        time2nextevent = get_time_next_HGT(t, params, sim_params)

    elif params["M"] > params["M0"]:
        if toprint: print("Lower Acquisition:", t)
        tt_spacers_current = int(params["M"]*params["Nh"])
        tt_spacers_lost = np.rint(params["Nh"]*(1-(params["M"]/params["M0"]))*params["M"]*rate_recovery)
        params["M"] = params["M"] + tt_spacers_lost/int(params["Nh"])

        num_to_add = np.sum(n) + tt_spacers_lost
        if num_to_add < 0:
            num_to_add = 0
        num_to_remove = tt_spacers_current - int(params["Nh"]*params["M"]) + num_to_add

    else:
        if toprint: print("Nothing happens at time:", t)

        num_to_add = np.sum(n)
        num_to_remove = np.sum(n)

    sim_params["time_next_event"] = time2nextevent
    return params, sim_params, num_to_add, num_to_remove

def HGT_discrete_event(t, n, params, sim_params, toprint = False):
    timesteps_to_recovery = np.ceil(1/(params["rate_recovery"]*sim_params["dt"]))
    counter_to_recovery = sim_params["counter_to_recovery"]
    time2nextevent = sim_params["time_next_event"]

    if t > time2nextevent:
        if toprint: print("HGT Occurs at time:", t)

        num_to_add = np.sum(n) + params["HGT_bonus_acquisition"]
        num_to_remove = np.sum(n)
        params["M"] = params["M"] + params["HGT_bonus_acquisition"]/params["Nh"]
        counter_to_recovery = counter_to_recovery + timesteps_to_recovery
        time2nextevent = get_time_next_HGT(t, params, sim_params)

    elif counter_to_recovery > 0:
        if toprint: print("Lower Acquisition:", t)
        num_to_add = np.sum(n) - params["HGT_bonus_acquisition"]*(1/timesteps_to_recovery)
        num_to_remove = np.sum(n)
        params["M"] = params["M"] - (params["HGT_bonus_acquisition"]/params["Nh"])*(1/timesteps_to_recovery)
        counter_to_recovery -= 1

    else:
        if toprint: print("Nothing happens at time:", t)

        num_to_add = np.sum(n)
        num_to_remove = np.sum(n)

    sim_params["counter_to_recovery"] = counter_to_recovery
    sim_params["time_next_event"] = time2nextevent
    return params, sim_params, num_to_add, num_to_remove