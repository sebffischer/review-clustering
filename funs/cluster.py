
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.cluster._kmeans import _k_init
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from coclust.clustering import SphericalKmeans
from funs.utils import get_centers


class SKMeans():
    """
    Implementation of Spherical K-Means ++
    """

    def __init__(self, n_clusters=5, n_init=1, max_iter=100, random_state=None,
                 tol=1e-4, init=None):
        """
        :param n_clusters: Number of clusters
        :param n_init: Number of different initializations
        :param max_iter: maximum number of iterations  
        :param random_state: seed for the initialization 
        :param tol: convergence tolerance 
        :param init: The initial centers; if None they are obtained using 
        _k_init from sklearn applied to the normalized vectors. Must be a list
        containing numpy arrays
        """
        assert isinstance(n_clusters, (int, np.int32, np.int64))
        assert isinstance(max_iter, (int, np.int32, np.int64))
        assert isinstance(random_state, (int, np.int32, np.int64)) \
            or random_state is None
        assert isinstance(tol, float)
        assert init is None or isinstance(init, np.ndarray)
        if init is not None: 
            assert n_clusters == init.shape[0] 
            print("when init is not None, random_state and n_init are \
                irrelevant")
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.inits = init

        if random_state is not None:
            np.random.seed(random_state)
        # we sample a seed for each iteration
        seeds = np.random.choice(np.arange(np.max([1000000, n_init * 1000])),
                                 size=n_init)
        seeds = [check_random_state(seed) for seed in seeds]
        self.seeds = seeds

    def fit(self, X):
        """
        :param X: numpy array containing the embeddings

        :return: the cluster labels 
        """
        assert isinstance(X, np.ndarray)
        self.X = X
        self.X_norm = normalize(X)
        # use the seeds to get an initialization for each run
        if self.inits is None: 
            self._get_initialization()
        else: 
            self.inits = [self.inits]
        self.inertias = list()  # contains the inertia of each run
        best_inertia = np.infty
        for i, init in enumerate(self.inits):
            inertia, labels, centers = self._fit_once(init=init)
            self.inertias.append(inertia)
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers
            print(f"Iteration {i + 1} done")

        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels

        return self.labels_

    def _fit_once(self, init):
        improvement = np.infty
        centers = init
        inertia = np.infty
        for _iter in range(self.max_iter):
            distances = pairwise_distances(X=self.X,
                                           Y=centers,
                                           metric="cosine")
            labels = np.apply_along_axis(func1d=np.argmin,
                                         axis=1,
                                         arr=distances)

            centers = list()
            for label in set(labels):
                # note that we use the normalized vectors for the calculation
                # of the means, for the calculation of the cosine distance above
                # it does not matter
                center = np.apply_along_axis(func1d=np.mean,
                                             axis=0,
                                             arr=self.X_norm[labels == label])
                centers.append(center)
            centers = np.vstack(centers)
            # select the minimum distances
            smallest_distances = distances[np.arange(len(distances)), labels]
            # smallest_distances = np.choose(labels, distances.T)
            new_inertia = np.sum(smallest_distances)
            improvement = inertia - new_inertia

            if improvement < self.tol:
                break

            inertia = new_inertia

        centers = normalize(centers)

        return inertia, labels, centers

    def _get_initialization(self):
        inits = list()

        def squared_euclidean(x):
            return np.inner(x, x)

        squared_norms = np.apply_along_axis(func1d=squared_euclidean,
                                            axis=1,
                                            arr=self.X_norm)
        for seed in self.seeds:
            init = _k_init(X=self.X_norm,
                           n_clusters=self.n_clusters,
                           x_squared_norms=squared_norms,
                           random_state=seed)
            inits.append(init)
        self.inits = inits


def merge_clusters(X, labels, which, normalize):
    """
    This function merges clusters and returns the new labels and centers. 

    :param X: Numpy array; rows contain embeddings
    :param labels: Numpy array containing the labels; Should be 0, 1, ...
    :param which: list of lists indicating which labels are to be merged, 
    e.g. [[1,2], [5,6]] means clusters 1 and 2 and clusters 5 and 6 are merged. 
    In case only one new cluster is formed it is also ok to set: which = [1,2]
    which would mean that clusters 1 and 2 are merged. New labels will 
    still be 0, 1, 2, ... and e.g. not 0, 2, ... 
    :param normalize: boolean indicating whether the centers should be 
    normalized with respect to euclidean length  

    :return: new labels and new centers    
    """
    assert isinstance(X, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(X) == len(labels)
    assert isinstance(which, list)
    # we check that which has the correct form
    assert all([isinstance(x, int) for x in which]) or \
        [isinstance(x, list) for x in which]
    if all([isinstance(x, int) for x in which]):
        which = [which]
    assert isinstance(normalize, bool)

    # check that all labels are assigned to only one cluster
    l_intermediate = [label for labels in which for label in labels]
    assert len(set(l_intermediate)) == len(l_intermediate)

    n_labels = len(set(labels))

    # first we create a 0-1 matrix of size n_cluster x n_clusters, where
    # each column is a 0-1 vector with exactly one 1. Those columns that
    # have their 1 in the same row will be in the same cluster

    assignment_matrix = np.eye(N=n_labels)

    for label_list in which:
        for j in label_list[1:]:  # the first entry in label_list decides which
            # row is used to store the current cluster in the assignment matrix
            assignment_matrix[label_list[0], j] = 1
            # remove the initial assigment
            assignment_matrix[j, j] = 0

    new_labels = np.zeros(len(labels), dtype=int)
    current_new_label = 0
    for assignments in assignment_matrix:
        current_old_labels = np.where(assignments == 1)[0]
        current_old_labels = current_old_labels.astype(int)
        if np.sum(assignments == 1) > 0:
            new_labels[np.isin(labels, current_old_labels)] = current_new_label
            current_new_label += 1

    new_centers = get_centers(X=X, labels=new_labels, normalize=normalize)

    return new_labels, new_centers


def do_em_selection(X, n_components_range, random_state, criterion="BIC",
                    max_iter=100, n_runs=1, cv_types=["spherical", "tied",
                                                      "diag", "full"]):
    """    
    Here we compare the BIC / AIC of various hyperparameter settings for 
    various hyperparameter settings of Gaussian Mixture Clustering. 
    :param X: the embeddings (rows containing observations)
    :param n_components_range: which values for n_components should be tested
    e.g. np.arange(5, 10) or [2, 6, 7]
    :param random_state: the seed 
    :param criterion: either "AIC" or "BIC"
    :param max_iter: maximum number of iterations for the EM-algorithm
    :param cv_types: which types of covariances matrices should be tried
    :param n_runs: number of iterations 

    :return: the best model and a dataframe containing the average BIC / AIC
    for each hyperparameter setting 
    """
    # input checking------------------------------------------------------------
    assert criterion in ["AIC", "BIC"]
    possible_cv_types = ["spherical", "tied", "diag", "full"]
    assert all([cv_type in possible_cv_types for cv_type in cv_types])
    n_combs = len(cv_types) * len(n_components_range)

    # draw seeds
    np.random.seed(random_state)
    m = max(1000000, n_runs * n_combs * 1000)
    seeds = np.random.choice(np.arange(m), n_runs * n_combs,
                             replace=False)

    # calculate AICs or BICs----------------------------------------------------
    lowest_measure = np.infty

    measures = np.empty((len(cv_types),
                         len(n_components_range),
                         n_runs))

    for run in range(n_runs):
        for i, cv_type in enumerate(cv_types):
            for j, n_components in enumerate(n_components_range):
                index = run * n_combs + i * len(n_components_range) + j
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type,
                                              random_state=seeds[index],
                                              max_iter=max_iter)

                new_labels = gmm.fit_predict(X)
                if criterion == "BIC":
                    measures[i, j, run] = gmm.bic(X)
                if criterion == "AIC":
                    measures[i, j, run] = gmm.aic(X)
                if measures[i, j, run] < lowest_measure:
                    lowest_measure = measures[i, j, run]
                    best_gmm = gmm
                    labels = new_labels

    best_gmm.labels_ = labels
    measure_matrix_mean = np.apply_along_axis(np.mean, 2, measures)
    measure_matrix_mean = pd.DataFrame(measure_matrix_mean,
                                       columns=n_components_range,
                                       index=cv_types)

    return best_gmm, measure_matrix_mean


def do_kmeans_selection(X, n_components_range, n_runs=1, random_state=None,
                        metric="cosine"):
    """
    This function calculates the inertia and the silhouette score for the 
    given number of clusters. 

    :param X: numpy array where rows contain the observations 
    :param n_components_range: which numbers of clusters are to be tried
    :param n_runs: how often is the experiment to be repeated (inertias and 
    silhouette scores will be mean over those trials)
    :param random_state: the seed
    :param metric: which metric to use: either "euclidean" or "cosine" 

    :return: dataframe containing the inertias and silhouette scores, figure
    displaying the silhouette scores and inertias 
    """
    # input checking------------------------------------------------------------
    assert metric in ["cosine", "euclidean"]

    if random_state is not None:
        np.random.seed(random_state)

    # create seeds
    m = max(1000000, n_runs * len(n_components_range) * 1000)
    seeds = np.random.choice(m, n_runs * len(n_components_range))

    # do the clustering---------------------------------------------------------
    inertia_matrix = np.empty((n_runs, len(n_components_range)))
    silhouette_matrix = np.empty((n_runs, len(n_components_range)))

    for run in range(n_runs):
        for j, k in enumerate(n_components_range):
            index = run * len(n_components_range) + j
            if metric == "cosine":
                model = SKMeans(n_clusters=k, random_state=seeds[index])
            else:
                model = KMeans(n_clusters=k, random_state=seeds[index])
            model.fit(X)
            inertia_matrix[run, j] = model.inertia_
            silhouette_matrix[run, j] = silhouette_score(X=X,
                                                         labels=model.labels_,
                                                         metric=metric)

    inertias_mean = np.apply_along_axis(np.mean,
                                        axis=0,
                                        arr=inertia_matrix)
    silhouettes_mean = np.apply_along_axis(np.mean,
                                           axis=0,
                                           arr=silhouette_matrix)

    df = pd.DataFrame({"k": list(n_components_range),
                       "inertia": inertias_mean,
                       "silhouette": silhouettes_mean})

    # create the plot-----------------------------------------------------------
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 5))
    axs[0, 0].boxplot(x=inertia_matrix)
    axs[0, 0].title.set_text("inertia boxplots")
    axs[0, 0].set_xticklabels(n_components_range, fontsize=12)
    axs[0, 1].boxplot(x=silhouette_matrix)
    axs[0, 1].set_xticklabels(n_components_range, fontsize=12)
    axs[0, 1].title.set_text("silhouette boxplots")
    axs[0, 1].set_xticklabels(n_components_range, fontsize=12)
    axs[1, 0].plot(n_components_range, inertias_mean)
    axs[1, 0].title.set_text("inertia line")
    axs[1, 0].set_xticklabels(n_components_range, fontsize=12)
    axs[1, 1].plot(n_components_range, silhouettes_mean)
    axs[1, 1].title.set_text("silhouette line")
    axs[1, 1].set_xticklabels(n_components_range, fontsize=12)
    plt.close()
    return df, fig


def em(X, **kwargs):
    """
    Simply wrapper for the EM algorithm because .fit does not create the 
    attribute labels_
    """
    model = GaussianMixture(**kwargs)
    labels = model.fit_predict(X)
    model.labels_ = labels
    return model


def skmeans(X, **kwargs):
    """
    RATHER USE SKMeans. 

    This function is only there to reproduce older results. 

    This is a simple wrapper around SphericalKmeans from coclust. 
    labels are transformed from a list to a numpy array. 
    In addition to that we calculate normalized cluster centers that 
    are not obtained out of the box. 

    :param embeddings: the sentence embeddings
    :param **kwargs: all parameters are passed to SphericalKmeans 

    :return: SphericalKmeans model with the additional attribute 
    cluster_centers_ and labels_ transformed to a numpy array 
    """
    model = SphericalKmeans(**kwargs)
    model.fit(X)
    X_norm = normalize(X)
    model.labels_ = np.array(model.labels_)

    model.cluster_centers_ = get_centers(X=X_norm,
                                         labels=model.labels_,
                                         normalize=True)

    return model
