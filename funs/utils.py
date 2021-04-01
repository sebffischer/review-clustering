import numpy as np
import pandas as pd
import pickle as pkl
import os
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from heapq import nlargest as _nlargest
from numpy.linalg import norm
import re
import itertools
from sklearn.metrics import adjusted_mutual_info_score


def get_centers(X, labels, normalize=False):
    """
    Calculates the centers of embeddings X for given labels. If normalize 
    is True the centers are normalized (for Spherical K-Means). 

    :param X: The embeddings 
    :param labels: The cluster labels. Must be numpy array and labels should be
    0, 1, 2, ... 
    :param normalize: Whether centers should normalized with respect to 
    euclidean norm.

    :return: numpy array containing the centers    
    """
    n_clusters = len(set(labels))
    centers = np.empty((n_clusters, X.shape[1]))

    for label in set(labels):
        center = np.apply_along_axis(np.mean,
                                     axis=0,
                                     arr=X[labels == label, :])
        if normalize:
            centers[label, :] = center / norm(center)
        else:
            centers[label, :] = center

    return centers


def get_central_indices(D, k, labels):
    """
    For each cluster center this function obtains the indices and distances of 
    the k embeddings that are closest to the respective center.

    :param D: the distance matrix (distance between embeddings to centers)
    rows are reviews, columns are clusters
    :param k: The k closest indices for each cluster are returned 
    :labels: A numpy array containing the cluster labels. Labels should be 
    0, 1, 2, ... 

    :return: a dictionary where each entry contains a pandas dataframe with the
    ids and distances of the k observations closest to the respective cluster
    center
    """
    assert len(D) == len(labels)

    output = dict()

    for label in set(labels):
        k_adj = min(k, np.sum(labels == label))
        # first we get the smallest k_adj indices and distances, they are not
        # necessarily sorted
        ids = np.argpartition(D[:, label], k_adj)[:k_adj]
        distances = D[ids, label]

        # sort them
        rearrange = np.argsort(distances)
        distances = distances[rearrange]
        ids = ids[rearrange]

        output[label] = pd.DataFrame({"id": ids,
                                      "distance": distances})
    return output


def get_distances(X, centers, labels=None, metric="cosine", precisions=None):
    """
    This function calculates the distance between the rows of X and the centers. 
    If the labels are passed it only calculates the distance between each row 
    and the center of the cluster to which it belongs.

    :param X: the embeddings 
    :param centers: Numpy array containing the cluster centers (as rows)
    :param labels: Numpy array containing the cluster labels must be 
    0, 1, 2, ...
    :param metric: the metric to be used for the distance calculation. 
    Was only used for "euclidean", "cosine" and "mahalanobis" during the project
    :param precision: the precision matrix in case metric is "mahalanobis" as 
    obtained from sklearn.GaussianMixture. It must have the shape 
    (n_)

    :return: In case labels are passed, a numpy vector containing the distance
    to the respective cluster center for each review. Otherwise a numpy array
    where each row contains the distances to each of the cluster centers. 
    """
    assert metric in ["euclidean", "cosine", "mahalanobis"]
    n_clusters = centers.shape[0]
    if metric == "mahalanobis":
        assert isinstance(precisions, np.ndarray)
        precisions = format_precisions(precisions=precisions, 
                                       n_clusters=n_clusters, 
                                       n_features=centers.shape[1])

    n_obs = X.shape[0]

    if labels is None:
        if metric == "mahalanobis":
            distances = np.empty((n_obs, n_clusters))
            for label in range(n_clusters):
                distances[:, label] = pairwise_distances(
                    X=X,
                    Y=centers[label, :][np.newaxis, :],
                    metric=metric,
                    VI=precisions[label, :, :]).squeeze()
        else:
            distances = pairwise_distances(
                X=X,
                Y=centers,
                metric=metric)
        return distances

    else:
        distances = np.empty(n_obs)
        for label in set(labels):
            if metric == "mahalanobis":
                distances[labels == label] = pairwise_distances(
                    X=X[labels == label, :],
                    Y=centers[label, :][np.newaxis, :],
                    metric=metric,
                    VI=precisions[label, :, :]).squeeze()
            else:
                distances[labels == label] = pairwise_distances(
                    X=X[labels == label, :],
                    Y=centers[label, :][np.newaxis, :],
                    metric=metric).squeeze()

    return distances


def has_key(d, key):
    return key in d.keys()


def path_creator(path):
    """
    When a folder .../name/ should be created but already exists it creates
    .../name(1)/, if this already exists it creates .../name(2)/ etc. 
    """
    symb = path[-1]  # whether the path ends with \ or /
    assert symb in ["/", "\\"]
    if os.path.exists(path):
        # check if path is of the sort ../foldername(n)/
        if re.match(".*\(\d{1,}\)", path[:-1]):
            # we make a list out of the string because strings are unmutuable
            path_list = np.array(list(path))
            # now we extract the old number
            open_brack_pos = int(np.where(path_list == "(")[-1])
            old_number = path_list[range(open_brack_pos + 1,
                                         len(path_list) - 2)]
            old_number = int("".join(old_number))
            new_number = old_number + 1
            # stack new number together with the rest of the path
            path_list = np.hstack([path_list[:open_brack_pos + 1],
                                   new_number,
                                   ")",
                                   symb])

            # make string out of the list
            path = "".join(path_list)
            path = path_creator(path)
        else:  # add (1) in the first iteration if ../foldername/ already exists
            path = path[:-1] + "(1)" + symb
            path = path_creator(path)
    return path


def sample_reviews(reviews, labels, k, random_state=None):
    """
    This function samples at max k reviews for each cluster (only less when the 
    cluster contains less observations) and returns a dictionary of lists with 
    the randomly sampled reviews 

    :param reviews: A pandas series containing the reviews. 
    :param labels: Numpy array containing the cluster labels.
    :param k: The number of reviews sampled for each cluster. 
    :param random_state: The seed. 

    :return: a dictionary containing a list with k (or less) reviews for each 
    of the clusters
    """
    if random_state is not None: 
        np.random.seed(random_state)
    output = dict()

    for label in set(labels):
        # get the review indices of the current cluster
        indices = np.where(labels == label)[0]
        # if cluster has less than k values we have to adjust k
        k_adj = min(len(indices), k)
        sampled_indices = np.random.choice(indices, size=k_adj, replace=False)
        current_list = list(reviews.iloc[sampled_indices])
        output[label] = current_list

    return output


def get_close_matches_indexes(word, possibilities, n=3, cutoff=0.6):
    """Use SequenceMatcher to return a list of the indexes of the best 
    "good enough" matches. 

    An simple example would be word = "hello", 
    possibilities = ["hello", "i", "am", "hulk"]. 
    This would return 0 

    The function is mostly taken from here: 
    https://stackoverflow.com/questions/50861237/is-there-an-alternative-to-
    difflib-get-close-matches-that-returns-indexes-l

    :param word: is a sequence for which close matches 
    are desired (typically a string).
    :param possibilities: is a list of sequences against which to match word
    (typically a list of strings).
    :param n: (default 3) is the maximum number of close matches to
    return.  n must be > 0.
    :param cutoff: (default 0.6) is a float in [0, 1].  Possibilities
    that don't score at least that similar to word are ignored.

    :return: a list with indices of the close matches 
    """

    if not n > 0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    sequence_matcher = SequenceMatcher()
    # we set the second sequence in sequence matcher (it has slots 1 and 2)
    sequence_matcher.set_seq2(word)
    for idx, possibility in enumerate(possibilities):
        # set the second slot for comparison
        sequence_matcher.set_seq1(possibility)
        # check whether they are close enough
        if sequence_matcher.real_quick_ratio() >= cutoff and \
           sequence_matcher.quick_ratio() >= cutoff and \
           sequence_matcher.ratio() >= cutoff:
            result.append((sequence_matcher.ratio(), idx))

    # Move the best scorers to head of list
    result = _nlargest(n, result)

    # Strip scores for the best n matches
    return [x for score, x in result]


def recycle(x, sentence_ids):
    """
    This function allows to recycle values that are on the sentence-level to the
    token/ngram, level 

    :param x: any vector on the sentence level (e.g. labels or weights)
    :param sentence_ids: the sentence_ids as obtained by get_complete_embedding

    :return: the recycled labels (numpy array of length len(sentence_ids))
    """
    x_extended = np.empty(len(sentence_ids), dtype=x.dtype)
    for sentence_id in set(sentence_ids):
        x_extended[sentence_ids == sentence_id] = x[sentence_id]
    return x_extended


def print_complete(data):
    """
    This prints a whole dataframe

    :param data: dataframe to be printed

    :return: None 
    """
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", 1000,
                           "display.max_colwidth", None):
        print(data)
    return None


def plot_distance_distribution(X):
    """
    This function creates the plot for the pairwise distance distributions 
    and returns it (euclidean and cosine distance).

    :param X: The embeddings

    :return: A matplotlib figure
    """
    D_euc = pairwise_distances(X, metric="euclidean")
    D_cos = pairwise_distances(X, metric="cosine")

    # subset upper triangle matrix (excluding the diagonal)
    vals_euc = D_euc[np.triu_indices(n=D_euc.shape[0], k=1)]
    vals_cos = D_euc[np.triu_indices(n=D_cos.shape[0], k=1)]
    # create the plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("pairwise distances", fontsize=16)
    axes[0].set_title("euclidean")
    axes[1].set_title("cosine")

    sns.kdeplot(vals_euc, ax=axes[0])
    sns.kdeplot(vals_cos, ax=axes[1])
    return fig


def load_pickle(path):
    """
    Simple wrapper that loads a pickle file from the specified path 
    and returns it 

    :param path: path to pickle object to be loaded

    :return: the loaded pickle object 
    """
    assert isinstance(path, str)
    if path[-4:] != ".pkl":
        path += ".pkl"
    with open(path, "rb") as f:
        data = pkl.load(f)

    return data


def save_pickle(object, path, overwrite=False):
    """
    Simple wrapper that saves the object to the provided path. 

    :param object: object to be saved
    :param path: path where that object is to be saved
    :param overwrite: if the path already exists, should it be overwritten

    :return: None 
    """
    assert isinstance(path, str)
    assert isinstance(overwrite, bool)
    if not path[-4:] == ".pkl":
        path += ".pkl"

    if overwrite is False and os.path.exists(path):
        raise Exception("file exists and overwrites equals False")

    if overwrite is True and os.path.exists(path):
        print("file exists but is overwritten")

    with open(path, "wb") as output:
        pkl.dump(object, output)

    return None


def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    This function is taken from here:
    https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib


    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2,
                         value_format.format(h), ha="center",
                         va="center")


def get_central_reviews(embeddings, centers, k, text, metric, labels,
                        precisions=None):
    """
    For center-based algorithms such as (Spherical) K-Means or the 
    EM-algorithm this function creates a dataframe with the id, distance and 
    text for the k closest embeddings. It does this for each label and creates 
    an output dict such that output[label] contains a dataframe of the form: 
    id | distance | text
    70     0.1      Really fast 
    61     0.2      Great speed 
    Note that this is ordered according to the distance. 

    It does so by first calling get_distances, then get_central_indices and then 
    subsetting the corresponding text. 

    :param embeddings: numpy array containing the embeddings 
    :param centers: a numpy array containing the centers as rows 
    :param k: the k closest values for each cluster are returned
    :param text: a list or pandas Dataframe containing the reviews
    corresponding to the embeddings
    :param metric: the metric that is used to calculate the distances 
    :param precisions: a 3d numpy array containing the precision matrix of the 
    i-th cluster in precisions[i,:,:] in case the metric is "mahalanobis"

    :return: 
    A dict of dataframes that contain the columns: id | distance | text
    in an ordered fasion. 
    """
    # input checking------------------------------------------------------------
    assert isinstance(embeddings, np.ndarray)
    assert isinstance(centers, np.ndarray)
    assert isinstance(k, (int, np.int32, np.int64))
    assert isinstance(text, (list, pd.core.series.Series,
                             pd.core.frame.DataFrame))
    assert metric in ["cosine", "euclidean", "mahalanobis"]
    assert isinstance(precisions, np.ndarray) or precisions is None
    text = pd.Series(text).reset_index(drop=True)
    assert len(embeddings) == len(text)

    # create the distance matrices----------------------------------------------
    distance_matrix = get_distances(X=embeddings,
                                    centers=centers,
                                    labels=None,
                                    metric=metric,
                                    precisions=precisions)

    central_indices = get_central_indices(D=distance_matrix,
                                          k=k,
                                          labels=labels)

    # now we add the corresponding text as a column for each of the labels
    for key in central_indices.keys():
        central_indices[key]["text"] = list(text[central_indices[key]["id"]])
    return central_indices


def generate_parameter_combination(names, values):
    """
    This function creates a list of dictionaries such that each dictonary
    can be passed as **kwargs to a clustering function.

    An example would be: 
    names = ["n_clusters", "n_init"]
    values = [[5, 10], [10, 20]]
    which would create 2 * 2 dictionaries with all the possible combinations 
    {"n_clusters" : 5, "n_init" : 10}, {"n_clusters" : 5, "n_init_ : 10} etc.
    """
    # first we create all possible value combinations via itertools.product
    value_combinations = []
    for comb in itertools.product(*values):
        value_combinations.append(dict(zip(names, comb)))

    return value_combinations


def do_benchmark(embeddings_list, model_list, parameter_list, true_labels,
                 random_state, n_runs=1, criterion=adjusted_mutual_info_score):
    """
    This function compares different embeddings and different clustering 
    algorithms with respect to how well they reconstruct the true labels. 
    in terms of adjusted mutual information

    :param embeddings_list: list of 2d numpy arrays containing the embeddings
    (rows are observations) 
    :param model_list: a list of the model generators: KMeans, SKMeans or 
    GaussianMixture was used fot the project
    :param parameter_list: a list of lists where the i-th entry contains
    a list where each entry contains a dictionary that can be used as **kwargs
    for the i-th model in model_list
    :param true labels: the true labels (can e.g. be a list or pd-Series)
    :paran n_runs: number of iterations 
    :param criterion: function that measures the agreement between the 
    estimated clustering and the passed true_labels
    :param seeds: must have the same length as n_par_comb * n_embeddings * n_runs

    :return: 
    a 3d numpy with dim (n_embeddings, n_algorithm_comb, n_runs) that contains
    the performance measures 
    """
    # initialize the performance_matrix
    par_lengths = [len(x) for x in parameter_list]
    n_col = np.sum(par_lengths)
    n_row = len(embeddings_list)
    performance_matrix = np.zeros(shape=(n_row, n_col, n_runs))

    # get seeds
    np.random.seed(random_state)
    seeds = np.random.choice(np.arange(1000000), n_row * n_col * n_runs)

    seed_counter = 0
    for run in range(n_runs):
        for i, embeddings in enumerate(embeddings_list):
            j = 0
            for k, model in enumerate(model_list):
                for par in parameter_list[k]:
                    current_model = model(
                        **par, random_state=seeds[seed_counter])
                    try:
                        # SphericalKmeans has no fit_predict function
                        # therefore we need this exception
                        if not hasattr(model, "fit_predict"):
                            current_model.fit(X=embeddings)
                            predicted_labels = current_model.labels_
                        else:
                            predicted_labels = current_model.fit_predict(
                                X=embeddings)

                        performance_matrix[i, j, run] = criterion(
                            predicted_labels,
                            true_labels)
                    except:
                        performance_matrix[i, j, run] = None
                    j += 1
                    seed_counter += 1

    return performance_matrix



def format_precisions(precisions, n_clusters=None, n_features=None): 
    """
    Depending on the structure of the covariance matrix the precicions_ 
    attribute has a different shape. Note that this function only works 
    as long as the embedding dimension is unequal to the cluster dimension. 
    
    1. full: (n_clusters, emedding_dim, embedding_dim)
    2. spherical: (n_clusters)
    3. diag: (n_clusters, embedding_dim)
    4. tied: (embeddings_dim, embedding_dim)
    
    This function unifies all these representations, i.e. brings it into the 
    form of 4. 
    
    :param precisions: the precicions_ attribute as obtained by GaussianMixture
    :param n_clusters: The number of clusters (only relevant in case of 
    spherical)
    
    :return: formatted precisions   
    """ 
    shape = precisions.shape
    if len(shape) == 3:  # 1. 
        return precisions
    if n_clusters == n_features: 
        raise Exception("in case n_clusters is the same as n_features the \
            precision matrix has to be formatted before as it is not possible \
                to infer the covariance type")
    if len(shape) == 1:  # 2.
        
        return \
            np.vstack([np.diag(v=np.repeat(x, n_features))[np.newaxis,:,:] \
                for x in precisions])
    if len(shape) == 2 and shape[0] == n_clusters: # 3. 
        return \
            np.vstack([np.diag(v=precisions[x,:])[np.newaxis, :, :] \
                for x in range(shape[0])])
    else: # 4. 
        return \
            np.vstack([precisions[np.newaxis,:,:] for x in range(n_clusters)])
        
                      
