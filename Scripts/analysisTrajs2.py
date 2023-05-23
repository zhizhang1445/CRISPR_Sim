import numpy as np
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

class TreeNode:
    def __init__(self, frame, mean, cov, count):
        self.frame = frame
        self.mean = mean
        self.cov = cov
        self.count = count
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)

    def remove_child(self, child):
        self.children.remove(child)

    def to_dict(self):
        node_dict = {
            'frame': self.frame,
            'mean': self.mean,
            'cov': self.cov,
            'count': self.count,
            'children': [child.to_dict() for child in self.children]
        }
        return node_dict

    def save_tree(self, filename):
        tree_dict = self.to_dict()
        with open(filename, 'w') as file:
            json.dump(tree_dict, file)

    @classmethod
    def load_tree(cls, filename):
        with open(filename, 'r') as file:
            tree_dict = json.load(file)
        return cls.from_dict(tree_dict)

    @classmethod
    def from_dict(cls, node_dict):
        node = cls(
            frame=node_dict['frame'],
            mean=node_dict['mean'],
            cov=node_dict['cov'],
            count=node_dict['count']
        )
        for child_dict in node_dict['children']:
            child = cls.from_dict(child_dict)
            node.add_child(child)
        return node

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