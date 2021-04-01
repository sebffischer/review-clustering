import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score
from funs.utils import get_distances


def plot_distance_between_centers(centers, metric="cosine", figsize=(7, 7)):
    """
    This function creates a heatmap for the distances between cluster centers

    :param centers: a numpy array containing the centers (as rows)
    :param metric: string containing the metric (anything that is compatible)
    with pairwise_distances
    :param figsize: the figsize of the heatmap

    :return: figure
    """
    distance_matrix = pairwise_distances(X=centers, metric=metric)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(distance_matrix, ax=ax)
    plt.close()
    return fig


def plot_center_distance_density(X, centers, labels, precisions=None,
                                 metric="cosine", figsize=(5, 10)):
    """
    Creates density plots for the center distances 

    :param X: the embeddings
    :param centers: Numpy array with the cluster centers 
    :param labels: the cluster labels 
    :param precisions: the precisions matrices in case of mahalanobis distance 
    (as obtained by GaussianMixture)
    :param metric: the metric used to calculate the distances 
    :param figsize: the figsize of the plot 

    :return: figure 
    """
    label_list = list(set(labels))
    n_clusters = len(label_list)

    distances = get_distances(X=X,
                              centers=centers,
                              labels=labels,
                              metric=metric,
                              precisions=precisions)

    ncol = 2
    nrow = int(np.ceil(n_clusters / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    
    min_dist = min(distances)
    max_dist = max(distances)
    
    for i in range(nrow):
        for j in range(ncol):
            index = i * ncol + j
            if i * ncol + j + 1 <= n_clusters:
                label = label_list[index]
                title = f"label = {label}, n = {sum(labels == label)}"
                # min_dist = min(distances[labels == label])
                # max_dist = max(distances[labels == label])
                sns.kdeplot(data=distances[labels == label],
                            clip=(min_dist, max_dist),
                            ax=axs[i, j]).set_title(title)
            else:  # in case we have an uneven number of clusters
                fig.delaxes(axs[i, j])

    plt.tight_layout()
    plt.close()
    return fig


def plot_silhouettes(X, labels, metric="cosine", figsize=(10, 15)):
    """
    Creates a silhouette plot 
    this function is mostly taken from here: 
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    :param X: the embeddings
    :param labels: the cluster labels 
    :param metric: the metric used to calculate the silhouette scores 
    :param figsize: the figsize for the plot that is created 

    :return: fig 
    """
    fig, axs = plt.subplots(figsize=figsize)
    label_set = set(labels)
    n_clusters = len(label_set)
    silhouette_values = silhouette_samples(X, labels, metric=metric)

    axs.set_xlim([min(silhouette_values), max(silhouette_values)])
    axs.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(X, labels)
    y_lower = 10
    for label in label_set:
        current_silhouettes = silhouette_values[labels == label]
        current_silhouettes.sort()
        current_size = len(current_silhouettes)
        y_upper = y_lower + current_size

        color = cm.nipy_spectral(float(label) / n_clusters)
        axs.fill_betweenx(np.arange(y_lower, y_upper),
                          0, current_silhouettes,
                          facecolor=color, edgecolor=color, alpha=0.7)

        axs.text(-0.05, y_lower + 0.5 * current_size, str(label))

        y_lower = y_upper + 10

    # The vertical line for average silhouette score of all the values
    axs.axvline(x=silhouette_avg, color="red", linestyle="--")
    axs.set_title("Silhouette plot")
    axs.set_xlabel("Silhouette coefficient values")
    axs.set_ylabel("Cluster label")

    axs.set_yticks([])  # Clear the yaxis labels / ticks
    plt.close()
    return fig
