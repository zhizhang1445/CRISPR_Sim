import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from trajectory import *
from trajectoryVisual import *
from trajsTree import *
from supMethods import *
from formulas import *

def get_count_single(init_list, params, sim_params, start_index = 0):
    count_all_root = []

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            
            counts = np.array([x.count for x in trajs])
            if len(counts)>0:
                count_all_root.extend(counts)

    return  np.mean(count_all_root[start_index:]), np.var(count_all_root[start_index:])

def get_var_single(init_list, params, sim_params):
    var_all_root = []
    var_calc_all_root = []

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            
            var = np.array([np.linalg.det(x.cov) for x in trajs])
            if len(var)>0:
                var_all_root.extend(var)

            count = np.array([x.count for x in trajs])
            if len(count)>0:
                var_calc = calculate_var(count, params, sim_params)
                var_calc_all_root.extend(var_calc)

    return  np.mean(var_all_root), np.var(var_all_root), np.mean(var_calc_all_root), np.var(var_calc_all_root)

def plot_velocity_single(init_list, params, sim_params, show = False):
    velocity_obs = np.array([])
    velocity_calc = np.array([])
    velocity_Fisher = np.array([])

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            positions = [x.mean for x in trajs]
            time = [x.frame for x in trajs]
            time_avg = average_of_pairs(time)
            
            counts = np.array([x.count for x in trajs])
            counts_avg = average_of_pairs(counts)

            x_val, y_val = extract_xy(positions)
            if isinstance(x_val, np.ndarray):
                dt = np.diff(time)
                v_obs = np.sqrt(np.diff(x_val)**2 + np.diff(y_val)**2)/dt
                velocity_obs = np.concatenate([velocity_obs, v_obs], axis = 0)

                v_calc = calculate_velocity(counts_avg, params, sim_params)
                velocity_calc = np.concatenate([velocity_calc, v_calc], axis = 0)
                
                v_Fisher = calculate_FisherVelocity(counts_avg, params, sim_params)
                velocity_Fisher = np.concatenate([velocity_Fisher, v_Fisher], axis = 0)
                if show:
                    plt.plot(time_avg, v_obs, color = 'teal')
                    plt.plot(time_avg, v_calc, color = 'orange')
                    plt.plot(time_avg, v_Fisher, color = 'orange')
                    plt.xlabel("Time")
                    plt.ylabel("velocity")
    return velocity_obs, velocity_calc, velocity_Fisher

def plot_velocity(foldername_itr, limits = [0.07, 0.07]):
    v_obs_mean = []
    v_obs_var = []
    v_calc_mean = []
    v_calc_var = []
    v_Fisher_mean = []
    v_Fisher_var = []

    for foldername in foldername_itr:
        with open(foldername + "/params.json") as json_file:
            params = json.load(json_file)
        with open(foldername + "/sim_params.json") as json_file:
            sim_params = json.load(json_file)

        init_list = []
        tree_index = 0
        tree_path = foldername + f"/trajs_trees/tree{tree_index}.json"
        while(os.path.isfile(tree_path)):
            # print("tree loaded:", tree_index)
            init_list.append(TreeNode.load_tree(tree_path))
            tree_index += 1
            tree_path = foldername + f"/trajs_trees/tree{tree_index}.json"
    
        velocity_obs, velocity_calc, velocity_Fisher = plot_velocity_single(init_list, params, sim_params)

        v_obs_mean.append(np.mean(velocity_obs))
        v_obs_var.append(np.sqrt(np.var(velocity_obs)))
        v_calc_mean.append(np.mean(velocity_calc))
        v_calc_var.append(np.sqrt(np.var(velocity_calc)))
        v_Fisher_mean.append(np.mean(velocity_Fisher))
        v_Fisher_var.append(np.sqrt(np.var(velocity_Fisher)))

    plt.figure(figsize = [3,3])
    plt.errorbar(v_calc_mean, v_obs_mean, xerr= v_calc_var,
                yerr = v_obs_var, linestyle = "None", capsize = 1)
    # plt.scatter(population_mean, velocity_mean)
    # plt.scatter(v_Fisher_mean, v_obs_mean, color = "orange", linestyle = '--', label = "Fisher Velocity")

    line_x = np.linspace(0, limits[0], 100)
    line_y = np.linspace(0, limits[1], 100)

    # Plot the diagonal line
    plt.plot(line_x, line_y, color='red', linestyle='--', label = "Linear Fitness Velocity")
    plt.xlim(0, limits[0])
    plt.ylim(0, limits[1])
    plt.xlabel("Theoretical Velocity")
    plt.ylabel("Numerical Velocity")
    plt.legend()

def get_mean_count_single(init_list, params, sim_params):
    count_all_root = np.array([])

    for root_node in init_list:
        for trajs in root_node.get_all_traversals():
            positions = [x.mean for x in trajs]
            time = [x.frame for x in trajs]
            time_avg = average_of_pairs(time)
            
            counts = np.array([x.count for x in trajs])
            count_all_root = np.concatenate([count_all_root, np.mean(counts)], axis=0)

    return count_all_root 