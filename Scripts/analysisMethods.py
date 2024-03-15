import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import matplotlib.pyplot as plt
import sys
import os

from supMethods import load_last_output, read_json
from trajsTree import make_Treelist, link_Treelists, save_Treelist
from trajectory import fit_GMM_unknown_components, get_nonzero_w_repeats, fit_unknown_GMM
from trajectoryVisual import make_frame, make_Gif, plot_Ellipses

def get_tdomain(foldername, to_plot=True, t0 = 0, margins = (-0.4, -0.4), dt = 0):
    with open(foldername + "/params.json") as json_file:
        params = json.load(json_file)
    with open(foldername + "/sim_params.json") as json_file:
        sim_params = json.load(json_file)

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

def create_both_Gifs(t_domain, foldername, margins):
    t_domain_no_error = []

    for t in t_domain:
        make_frame(foldername, t, save = True, margins=margins)
        t_domain_no_error.append(t)
        plt.close("all")

    make_Gif(foldername, t_domain_no_error, typename = "time_plots")
    params, sim_params = read_json(foldername)
    print("time plots made")

    t_domain_no_error = []
    for t in t_domain:
        try:
            n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{t}.npz").todok()
            indexes = get_nonzero_w_repeats(n_i)
            means, covs, counts = fit_GMM_unknown_components(n_i, params, sim_params, indexes, scale = np.sqrt(2))

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
            print(t)
            break
    else:
        make_Gif(foldername, t_domain_no_error, typename = "GMM_plots")
        save_Treelist(foldername, init_list)
        print("GMM plots made")
    return 1

def main(foldername, dt = 0, input_flag = True, margin = -0.4):

    while input_flag:
        t_domain= get_tdomain(foldername,to_plot=input_flag, 
                                                                         t0 = 0, margins = (margin, margin), dt = dt)

        user_input = input("Please enter 'Yes[Y]', a float to resize margins (default -0.4) or 'No[N]' to exit \n").lower()
        if user_input in {'yes', 'y', 'Yes'}:
            break
        elif user_input in {'no', 'n', 'No'}:
            print("Command To Exit")
            return 0
        
        else:
            margin = float(user_input)

    margins = (margin, margin)

    if not input_flag:
        t_domain = get_tdomain(foldername,to_plot=input_flag, t0 = 0, margins = margins, dt = dt)
    create_both_Gifs(t_domain, foldername, margins)
    return 1

if __name__ == "__main__":

    if len(sys.argv) == 2:
        foldername = sys.argv[1]
        input_flag = True
        main(foldername, input_flag)

    if len(sys.argv) > 3:
        dt = int(sys.argv[3])
    else:
        dt = 0

    if len(sys.argv) > 2:
        foldername = sys.argv[1]
        margin = float(sys.argv[2])
        subfolders = [f.path for f in os.scandir(foldername) if f.is_dir()]
        for folder in subfolders:
            main(folder, dt=dt, input_flag=False, margin=margin)
                
    else:
        pass
    