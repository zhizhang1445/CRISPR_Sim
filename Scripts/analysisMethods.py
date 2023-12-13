import numpy as np
import matplotlib.pyplot as plt
import scipy
import json
import matplotlib.pyplot as plt
import sys
import os

from supMethods import load_last_output
from trajsTree import make_Treelist, link_Treelists, save_Treelist
from trajectory import get_nonzero_w_repeats, fit_unknown_GMM, reduce_GMM
from trajectoryVisual import make_frame, make_Gif, plot_Ellipses

def get_tdomain_and_show_last_output(foldername, to_plot=True, t0 = 0, margins = (-0.4, -0.4)):
    with open(foldername + "/params.json") as json_file:
        params = json.load(json_file)
    with open(foldername + "/sim_params.json") as json_file:
        sim_params = json.load(json_file)

    t0 = 0
    try:
        dt = sim_params["dt_snapshot"]
    except KeyError:
        dt = sim_params["t_snapshot"]

    tf, n_final, nh_final = load_last_output(foldername)

    if to_plot:
        fig = plt.figure()
        plt.contour(n_final.toarray().transpose(), cmap = "Reds")
        plt.contour(nh_final.toarray().transpose(), cmap = "Blues")
        plt.margins(margins[0], margins[1])
        plt.show()

    t_domain = np.arange(t0, tf, dt)
    return t_domain, foldername, margins

def create_both_Gifs(t_domain, foldername, margins):
    t_domain_no_error = []

    for t in t_domain:
        make_frame(foldername, t, save = True, margins=margins)
        t_domain_no_error.append(t)

    make_Gif(foldername, t_domain_no_error, typename = "time_plots")
    print("time plots made")

    t_domain_no_error = []
    for t in t_domain:
        try:
            n_i = scipy.sparse.load_npz(foldername+f"/sp_frame_n{t}.npz").todok()
            indexes = get_nonzero_w_repeats(n_i)
            means_gmm, covs_gmm, counts_gmm = fit_unknown_GMM(indexes,n_components=1, 
                                                            w = 1000, reg_covar=1e8)
            means, covs, counts = reduce_GMM(means_gmm, covs_gmm, counts_gmm)

            next_list = make_Treelist(t, means, covs, counts)
    
            if t == t_domain[0]:
                init_list = next_list
                prev_list = next_list
                continue

            prev_list = link_Treelists(prev_list, next_list)
            prev_list = next_list
            plot_Ellipses(n_i, t, means, covs, save = True,
                        foldername = foldername, input_color = "teal", margins=(-0.42, -0.42))
            
            t_domain_no_error.append(t)
        except ValueError:
            print(t)
            break
    else:
        make_Gif(foldername, t_domain_no_error, typename = "GMM_plots")
        save_Treelist(foldername, init_list)
        print("GMM plots made")
    return 1

def main(foldername, input_flag = True, margin = -0.4):

    while input_flag:
        t_domain, foldername, margins = get_tdomain_and_show_last_output(foldername,to_plot=input_flag, 
                                                                         t0 = 0, margins = (margin, margin))

        user_input = input("Please enter 'Yes[Y]', a float to resize margins (default -0.4) or 'No[N]' to exit \n").lower()
        if user_input in {'yes', 'y'}:
            break
        elif user_input in {'no', 'n'}:
            print("Command To Exit")
            return 0
        
        else:
            margin = float(user_input)

    margins = (margin, margin)

    if not input_flag:
        t_domain, foldername, margins = get_tdomain_and_show_last_output(foldername,to_plot=input_flag, 
                                                                         t0 = 0, margins = margins)
    create_both_Gifs(t_domain, foldername, margins)
    return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        foldername = sys.argv[1]
        input_flag = True
        main(foldername, input_flag)

    elif len(sys.argv) > 2:
        foldername = sys.argv[1]
        margin = float(sys.argv[2])
        subfolders = [f.path for f in os.scandir(foldername) if f.is_dir()]
        for folder in subfolders:
            main(folder, input_flag=False, margin=margin)
    else:
        print("Missing folder")
        pass
    