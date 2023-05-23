import numpy as np
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

def checkIfInEllipse(mean1, mean2, cov1, r = 1) -> bool:
    eigval, eigvec = np.linalg.eigh(cov1)
    # dist = np.linalg.norm(mean1-mean2)
    diff_in_eigev = np.matmul(eigvec, mean1-mean2)

    norm_dist = np.linalg.norm(diff_in_eigev/eigval)
    
    if norm_dist <= np.power(r, 2):
        return True

def fit_unknown_GMM(index_nonzero_w_repeats, cov_type = "full",
                     n_components = 10, w = 1/10):

    gaussian_estimator =  BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_process",
                n_components = 2*n_components,
                reg_covar = 0,
                init_params="k-means++",
                max_iter = 1500,
                mean_precision_prior = 0.3,
                covariance_type = cov_type,
                weight_concentration_prior = w,
                warm_start = True
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

def find_redudant(means, covs, counts):
    true_count = np.copy(counts)
    to_join = [[i] for i in range(len(counts))]

    for i in range(len(means)):
        for j in range(len(means)):
            if i == j: continue

            avg_cov = (covs[i]*counts[i]+covs[j]*counts[j])
            avg_cov = avg_cov/(counts[i]+counts[j])

            if checkIfInEllipse(means[i], means[j], avg_cov):
                true_count[i] += true_count[j]
                true_count[j] = 0
                to_join[i].extend(to_join[j])
                to_join[j] = []
    return to_join, true_count

def reduce_GMM(means, covs, counts):
    to_join, true_count = find_redudant(means, covs, counts)

    reduced_means = []
    reduced_covs = []
    reduced_counts = []

    for i in true_count.nonzero()[0]:
        if true_count[i] == 0:
            continue

        true_mean = 0
        for j in to_join[i]:
            true_mean += means[j]*counts[j]
        true_mean = true_mean/true_count[i]
        
        aprox_cov = np.zeros_like(covs[i])
        for j in to_join[i]:
            dist = np.diag(means[j]-true_mean)
            aprox_cov += (covs[j]+dist)*counts[j]
        aprox_cov = aprox_cov/true_count[i]

        reduced_means.append(true_mean)
        reduced_covs.append(aprox_cov)
        reduced_counts.append(true_count[i])

    return reduced_means, reduced_covs, reduced_counts

def find_links(means1, covs1, means2):
    to_join = [[] for i in range(len(means1))]
    to_remove = [i for i in range(len(means2))]
    dist_array = np.zeros((len(means1), len(means2)))

    for i in range(len(means1)):
        for j in range(len(means2)):
            dist_array[i, j] = np.linalg.norm(means1[i]-means2[j])

    while len(to_remove) != 0:
        num_iter = 0
        for i in range(len(means1)):
            min_j = np.argmin(dist_array)

            if min_j in to_remove and num_iter < 10:
                if checkIfInEllipse(means1[i], means2[min_j], covs1[i]):
                    to_join[i].append(min_j)
                    to_remove.remove(min_j)
        num_iter += 1
    
    return to_join

