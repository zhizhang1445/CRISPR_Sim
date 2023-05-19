import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import json
import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

def get_nonzero_w_repeats(n_i):
    x_ind, y_ind = np.nonzero(n_i)
    nonzero_values = [n_i[index] for index in zip(x_ind, y_ind)]
    index_nonzero_w_repeats = []
    for value, index in zip(nonzero_values, zip(x_ind, y_ind)):
        for i in range(int(value)):
            index_nonzero_w_repeats.append(index)
    return index_nonzero_w_repeats

def make_ellipse(means, covariances, color = "navy"):
    if isinstance(covariances,float):
        covariances = np.eye(means.size)*covariances

    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        means, v[0], v[1], 180 + angle, color=color
    )
    return ell

def fit_unknown_GMM(index_nonzero_w_repeats, cov_type = "full",
                     n_components = 10, w = 10000):
    random_state = 2

    gaussian_estimator =  BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_distribution",
                n_components= 2*n_components,
                reg_covar=0,
                init_params="random",
                max_iter=1500,
                mean_precision_prior=0.8,
                covariance_type= cov_type,
                random_state=random_state,
                weight_concentration_prior = w
            )
    gaussian_estimator.fit(index_nonzero_w_repeats)

    covariances = gaussian_estimator.covariances_
    means = gaussian_estimator.means_
    clusters = gaussian_estimator.predict(index_nonzero_w_repeats)

    means_red = []
    covs_red = []
    for i in range(len(np.unique(clusters))):
        means_red.append(means[i])
        covs_red.append(covariances[i])

    _ , counts = np.unique(clusters, return_counts= True)
    return means_red, covs_red, counts

def fit_GMM(index_nonzero_w_repeats, cov_type = "full",
                     n_components = 10):

    gaussian_estimator =  GaussianMixture(
                n_components= n_components,
                reg_covar=0,
                init_params="kmeans",
                max_iter=1500,
                covariance_type= cov_type,
            )
    gaussian_estimator.fit(index_nonzero_w_repeats)

    covs = gaussian_estimator.covariances_
    means = gaussian_estimator.means_
    clusters = gaussian_estimator.predict(index_nonzero_w_repeats)

    _ , counts = np.unique(clusters, return_counts= True)
    return means, covs, counts