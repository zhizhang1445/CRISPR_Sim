import numpy as np
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from trajectoryVisual import make_ellipse

def get_nonzero_w_repeats(n_i):
    x_ind, y_ind = np.nonzero(n_i)
    nonzero_values = [n_i[index] for index in zip(x_ind, y_ind)]
    index_nonzero_w_repeats = []
    for value, index in zip(nonzero_values, zip(x_ind, y_ind)):
        for i in range(int(value)):
            index_nonzero_w_repeats.append(index)
    return index_nonzero_w_repeats

def checkIfInEllipse(mean1, mean2, cov1, scale = 1) -> bool:
    eigval, eigvec = np.linalg.eigh(cov1)
    # dist = np.linalg.norm(mean1-mean2)
    diff_in_eigev = np.matmul(eigvec, mean1-mean2)

    norm_dist = np.linalg.norm(diff_in_eigev/eigval)
    
    if norm_dist <= np.power(scale, 2):
        return True

def fit_unknown_GMM(index_nonzero_w_repeats,
                     n_components = 10, w = 10):

    gaussian_estimator =  BayesianGaussianMixture(
                weight_concentration_prior_type="dirichlet_distribution",
                n_components = 2*n_components,
                reg_covar = 0,
                init_params="kmeans",
                max_iter = 2000,
                mean_precision_prior = 0.8,
                covariance_type = "full",
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

            if checkIfInEllipse(means[i], means[j], avg_cov, np.sqrt(2)):
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

def Multivar_Normal(x, mean, cov, count):
    diff = x-mean
    Mahalanobis = np.matmul(diff.transpose(), np.matmul(cov, diff))
    pdf = np.exp((-1/2)*Mahalanobis)
    norm = np.sqrt(np.power(2*np.pi, 2)*np.linalg.det(cov))
    pdf = pdf/norm
    return pdf*count

def Sum_Normal(n, means, covs, counts):
    sum_Normal = scipy.sparse.dok_matrix(n.shape)
    x_inds, y_inds = n.nonzero()

    for mean, cov, count in zip(means, covs, counts):
        for x_ind, y_ind in zip(x_inds, y_inds):
            sum_Normal[x_ind, y_ind] += Multivar_Normal([x_ind, y_ind], 
                                                    mean, cov, count)
        return sum_Normal
    
    return sum_Normal

def fit_GMM(n, index_nonzero_w_repeats, cov_type = "full",
                     n_components = 10):

    gaussian_estimator =  GaussianMixture(
                n_components= n_components,
                reg_covar=0,
                init_params="k-means++",
                max_iter=1500,
                covariance_type = cov_type,
            )
    gaussian_estimator.fit(index_nonzero_w_repeats)

    covs = gaussian_estimator.covariances_
    means = gaussian_estimator.means_
    clusters = gaussian_estimator.predict(index_nonzero_w_repeats)

    _ , counts = np.unique(clusters, return_counts= True)

    calc_data = Sum_Normal(n, means, covs, counts)
    x_inds, y_inds = n.nonzero()
    classification = gaussian_estimator.predict(np.array([x_inds, y_inds]).transpose())

    chi_sqr = 0
    for x_ind, y_ind, i in zip(x_inds, y_inds, classification):
        diff = n[x_ind, y_ind] - calc_data[x_ind, y_ind]
        variance = np.linalg.det(covs[i])

        chi_sqr += np.power(diff, 2)/variance

    deg_freedom = n_components*5

    red_chi_sqr = chi_sqr/deg_freedom
    print("Reduced ChiSqrd: ", red_chi_sqr)

    return red_chi_sqr, means, covs, counts

