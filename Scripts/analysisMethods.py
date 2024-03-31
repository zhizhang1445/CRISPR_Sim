from ast import Raise
import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import matplotlib.pyplot as plt
import sys
import os

from supMethods import load_last_output, read_json
from trajsTree import make_Treelist, link_Treelists, save_Treelist
from trajectory import fit_GMM_unknown_components, get_nonzero_w_repeats, fit_unknown_GMM, fit_GMM
from trajectoryVisual import make_frame, make_Gif, plot_Ellipses
from plotStuff import get_var_single, get_count_single, plot_velocity_single
from entropy import plot_entropy_change

def get_tdomain(foldername, to_plot=True, t0 = 0, margins = (-0.4, -0.4), dt = 0):
    params, sim_params = read_json(foldername)

    t0 = 0
    if dt == 0:
        try:
            dt = sim_params["dt_snapshot"]
        except KeyError:
            dt = sim_params["t_snapshot"]

    tf, n_final, nh_final = load_last_output(foldername)
    t_domain = np.arange(t0, tf, dt)

    if to_plot:
        fig = plt.figure()
        plt.contour(n_final.toarray().transpose(), cmap = "Reds")
        plt.contour(nh_final.toarray().transpose(), cmap = "Blues")
        plt.margins(margins[0], margins[1])
        plt.show()
        return t_domain
    return t_domain

def create_Tree(t_domain, foldername, GMM_flag = 0):
    params, sim_params = read_json(foldername)
    init_list = []

    for t in t_domain:
        try:
            n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{t}.npz").todok()
            indexes = get_nonzero_w_repeats(n_i)
            if GMM_flag == 0:
                means, covs, counts = fit_GMM_unknown_components(n_i, params, sim_params, indexes, scale = np.sqrt(2))
            else:
                means, covs, counts = fit_GMM(n_i, params, sim_params, n_components = GMM_flag)

            next_list = make_Treelist(t, means, covs, counts)
            if t == t_domain[0]:
                init_list = next_list
                prev_list = next_list
                continue

            prev_list = link_Treelists(prev_list, next_list)
            prev_list = next_list
        except ValueError:
            print(f"Failure to find GMM at t = {t}")
            break
    else:
        save_Treelist(foldername, init_list)
    return init_list

def create_both_Gifs(t_domain, foldername, margins, GMM_flag = 0):
    t_domain_no_error = []
    init_list = []

    for t in t_domain:
        make_frame(foldername, t, save = True, margins=margins)
        t_domain_no_error.append(t)
        plt.close("all")

    make_Gif(foldername, t_domain_no_error, typename = "time_plots")
    params, sim_params = read_json(foldername)
    print("time plots made for ", foldername)

    t_domain_no_error = []
    for t in t_domain:
        try:
            n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{t}.npz").todok()
            indexes = get_nonzero_w_repeats(n_i)

            if GMM_flag == 0:
                means, covs, counts = fit_GMM_unknown_components(n_i, params, sim_params, indexes, scale = np.sqrt(2))
            else:
                means, covs, counts = fit_GMM(n_i, params, sim_params, n_components = GMM_flag)

            next_list = make_Treelist(t, means, covs, counts)
            if t == t_domain[0]:
                init_list = next_list
                prev_list = next_list
                continue

            prev_list = link_Treelists(prev_list, next_list)
            prev_list = next_list
            try:
                plot_Ellipses(n_i, t, means, covs, save = True,
                        foldername = foldername, input_color = "teal", margins=margins)
            except TypeError:
                continue
            
            t_domain_no_error.append(t)

            plt.close('all')
        except ValueError:
            print(f"Failure to find GMM at t = {t}")
            break
    else:
        make_Gif(foldername, t_domain_no_error, typename = "GMM_plots")
        save_Treelist(foldername, init_list)
        print("GMM plots made for ", foldername)
    return init_list

def create_results_plots(tdomain, foldername, init_list = None):
    params, sim_params = read_json(foldername)

    resultsfolder = foldername+"/Results"
    if not os.path.exists(resultsfolder):
        os.mkdir(resultsfolder)

    if init_list is not None:
        get_var_single(init_list, params, sim_params, to_plot = True, to_save_folder = resultsfolder)
        get_count_single(init_list, params, sim_params, to_plot = True, to_save_folder = resultsfolder)
        plot_velocity_single(init_list, params, sim_params, to_plot=True, to_save_folder = resultsfolder)

    plot_entropy_change(tdomain, foldername, to_plot=True, to_save_folder = resultsfolder)
    return 1

def main(foldername, dt = 0, plot_flag = True, margin = -0.4, GMM_flag = 0):
    margins = (margin, margin)

    t_domain = get_tdomain(foldername,to_plot=False, t0 = 0, margins = margins, dt = dt)
    if len(t_domain) <= 1:
        return 0
    
    if plot_flag:
        root_list = create_both_Gifs(t_domain, foldername, margins, GMM_flag)
    else: 
        root_list = create_Tree(t_domain, foldername, GMM_flag)
    
    try:
        create_results_plots(t_domain, foldername, root_list)
    except FileNotFoundError:
        print("No Entropy made at ", foldername)
    return 1

if __name__ == "__main__":
    dt = 0
    GMM_flag = 0
    margin = 0.0
    input_flag = False
    subfolder_flag = False
    to_plot = False

    if len(sys.argv) > 1:
        foldername = sys.argv[1] #foldername is read out first
    
    if len(sys.argv) > 2:
        GMM_flag = int(sys.argv[2]) #analysis can be restricted to less timesteps

    if len(sys.argv) > 3:
        subfolder_flag = sys.argv[3] #analysis can be restricted to less timesteps
        if subfolder_flag in ["Superfolder", "1", "superfolder", "subfolder", "Subfolder", "true", "True"]:
            subfolder_flag = True
        else: subfolder_flag = False

    if len(sys.argv) > 4:
        to_plot = sys.argv[4]
        if to_plot in ["Plot", "to_plot", "1", "True", "plot", "true"]:
            to_plot = True

    if len(sys.argv) > 5:
        margin = float(sys.argv[5]) #analysis can be restricted to less timesteps

    if len(sys.argv) > 6:
        dt = int(sys.argv[6]) #analysis can be restricted to less timesteps

    if subfolder_flag:
        margin = float(sys.argv[4])
        subfolders = [f.path for f in os.scandir(foldername) if f.is_dir()]
        for folder in subfolders:
            main(folder, dt, plot_flag=to_plot, margin=margin, GMM_flag = GMM_flag)

    else:
        if to_plot:
            input_flag = True

        while input_flag:
            t_domain= get_tdomain(foldername,to_plot=True, t0 = 0, margins = (margin, margin), dt = dt)

            user_input = input("Please enter 'Yes[Y]', a float to resize margins (default -0.4) or 'No[N]' to exit \n").lower()
            if user_input in {'yes', 'y', 'Yes'}:
                input_flag = False

            elif user_input in {'no', 'n', 'No'}:
                print("Command To Exit")
                raise KeyboardInterrupt
            
            else:
                margin = float(user_input)
        main(foldername, dt, to_plot, margin, GMM_flag = GMM_flag)
        pass
    