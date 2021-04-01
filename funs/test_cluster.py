import unittest
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import adjusted_mutual_info_score, pairwise_distances
from funs.cluster import merge_clusters, SKMeans


# to understand the idea:
# import matplotlib.pyplot as plt
# plt.scatter(X[:,0], X[:,1])
# plt.show()

class TestCluster(unittest.TestCase):

    def test_skmeans(self):
        radiants_means = [0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi]
        np.random.seed(98)
        radiants = np.repeat(radiants_means, 100) + np.random.randn(400) * 0.01

        true_labels = np.repeat(np.arange(4), 100)

        def helper(x):
            return [np.sin(x), np.cos(x)]

        X = [helper(radiant) for radiant in radiants]

        X = np.vstack(X)

        X[0:50, :] = X[0:50, :] * 10

        X[100:150, :] = X[100:150, :] * 10

        X[200:250, :] = X[200:250, :] * 10

        skmeans = SKMeans(n_clusters=4, n_init=10,
                          max_iter=100, random_state=42)
        skmeans.fit(X=X)
        # we have to pay attention to possible permutations
        skmeans = SKMeans(n_clusters=4, n_init=10,
                          max_iter=100, random_state=42)
        skmeans.fit(X=X)
        ami = adjusted_mutual_info_score(skmeans.labels_, true_labels)
        self.assertTrue(np.abs(ami - 1) < 0.01)

        center_norms = np.apply_along_axis(norm, 1, skmeans.cluster_centers_)

        center_1 = np.apply_along_axis(np.mean, axis=0, arr=X[0:100, :])
        center_1 = center_1 / norm(center_1)
        center_2 = np.apply_along_axis(np.mean, axis=0, arr=X[100:200, :])
        center_2 = center_2 / norm(center_2)
        center_3 = np.apply_along_axis(np.mean, axis=0, arr=X[200:300, :])
        center_3 = center_3 / norm(center_3)
        center_4 = np.apply_along_axis(np.mean, axis=0, arr=X[300:400, :])
        center_4 = center_4 / norm(center_4)
        centers = np.vstack([center_1, center_2, center_3, center_4])

        D = pairwise_distances(skmeans.cluster_centers_,
                               centers)
        minimal_distances = np.apply_along_axis(np.min, 1, D)

        self.assertTrue(np.sum(np.abs(minimal_distances)) < 0.01)

        self.assertTrue(np.sum(np.abs(center_norms -
                                      np.repeat(1, len(skmeans.cluster_centers_)))) < 0.01)

    def test_merge_clusters(self):
        labels = np.array([0, 1, 0, 2, 1])
        new_labels = np.array([0, 1, 0, 1, 1])
        X = np.array([[0, 0], [1, 1], [2, 2], [2, 2], [3, 3]])
        new_centers = np.vstack([[1, 1], [2, 2]])
        new_labels_calc, new_centers_calc = merge_clusters(X=X,
                                                           labels=labels,
                                                           which=[1, 2],
                                                           normalize=False)
        self.assertTrue(all(new_labels == new_labels_calc))
        self.assertTrue(np.all(new_centers == new_centers_calc))

        new_labels_calc, new_centers_calc = merge_clusters(X=X,
                                                           labels=labels,
                                                           which=[1, 2],
                                                           normalize=True)

        new_centers = np.repeat(1 / np.sqrt(2), 4).reshape(2, 2)
        self.assertTrue(all(new_labels == new_labels_calc))
        self.assertTrue(np.all(new_centers == new_centers_calc))

if __name__ == "__main__":
    unittest.main()

