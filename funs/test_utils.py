import unittest
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from funs.utils import get_central_indices, sample_reviews, recycle, \
    get_close_matches_indexes, get_central_reviews, get_centers, get_distances, \
    generate_parameter_combination, format_precisions


data = pd.DataFrame(["great", "great", "amazing"], columns=["text"])
labels = np.array([0, 0, 1])
sampled_dict = {0: ["great", "great"], 1: ["amazing"]}


class TestUtils(unittest.TestCase):

    def test_get_centers(self):
        embeddings = np.array([[1, 1],
                               [2, 2],
                               [3, 3],
                               [4, 4],
                               [5, 5],
                               [6, 6]])
        centers = np.array([[2, 2],
                            [5, 5]])
        centers_norm = np.array([[np.sqrt(0.5), np.sqrt(0.5)],
                                 [np.sqrt(0.5), np.sqrt(0.5)]])
        labels = np.array([0, 0, 0, 1, 1, 1])

        centers_calc = get_centers(X=embeddings, labels=labels)
        self.assertTrue(np.sum(np.abs(centers - centers_calc)) < 0.001)
        centers_calc_norm = get_centers(X=embeddings, labels=labels,
                                        normalize=True)
        self.assertTrue(
            np.sum(np.abs(centers_norm - centers_calc_norm)) < 0.001)

    def test_get_central_indices(self):
        distance_matrix = np.array([[1, 8],
                                    [2, 7],
                                    [3, 6],
                                    [4, 5]])
        labels = np.array([0, 1, 0, 1])
        central_indices = get_central_indices(D=distance_matrix,
                                              k=1,
                                              labels=labels)

        self.assertTrue(central_indices[0]["id"][0] == 0)
        self.assertTrue(central_indices[0]["distance"][0] == 1)

        self.assertTrue(central_indices[1]["id"][0] == 3)
        self.assertTrue(central_indices[1]["distance"][0] == 5)
        
    def test_get_central_reviews(self): 
        X = np.array([1, 2, 3, 4]).reshape(4, 1)
        centers = np.array([0.5, 3.5]).reshape(2, 1)
        labels = np.array([0, 0, 1, 1])
        central_reviews = get_central_reviews(embeddings=X, 
                                              labels=labels, 
                                              centers=centers, 
                                              k=1, 
                                              text=["a", "b", "c", "d"], 
                                              metric="euclidean")

        self.assertTrue(central_reviews[0]["text"][0] == "a")
        self.assertTrue(central_reviews[0]["id"] == "0")
        self.assertTrue(np.abs(central_reviews[0]["distance"][0] - 0.5) < 0.001)
        
        # hear the min is not unique but the first match is taken 
        self.assertTrue(central_reviews[1]["text"][0] == "c")
        self.assertTrue(central_reviews[1]["id"][0] == 2)
        self.assertTrue(np.abs(central_reviews[1]["distance"][0] - 0.5) < 0.001)

    def test_get_distances(self):
        X = np.array([[0, 1],
                      [0, 2]])

        centers = np.array([[0, 1], [1, 0]])
        labels = np.array([0, 1])

        precisions = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])

        distances_cosine = np.array([[0, 1], [0, 1]])
        distances_euclidean = np.array([[0, np.sqrt(2)], [1, np.sqrt(5)]])
        distances_mahalanobis = \
            np.array([[0, np.sqrt(2 * 2)], [1, np.sqrt(5 * 2)]])

        distances_euclidean_calc = get_distances(X=X,
                                                 centers=centers,
                                                 labels=None,
                                                 metric="euclidean")

        distances_cosine_calc = get_distances(X=X,
                                              centers=centers,
                                              labels=None,
                                              metric="cosine")

        distances_mahalanobis_calc = get_distances(X=X,
                                                   centers=centers,
                                                   labels=None,
                                                   metric="mahalanobis",
                                                   precisions=precisions)

        self.assertTrue(np.sum(np.abs(
            distances_euclidean_calc - distances_euclidean)) < 0.01)

        self.assertTrue(np.sum(np.abs(
            distances_cosine_calc - distances_cosine)) < 0.01)

        self.assertTrue(np.sum(np.abs(
            distances_mahalanobis_calc - distances_mahalanobis)) < 0.01)

        # now the same but with passing labels

        distances_cosine = np.array([0, 1])
        distances_euclidean = np.array([0, np.sqrt(5)])
        distances_mahalanobis = \
            np.array([0, np.sqrt(5 * 2)])

        distances_euclidean_calc = get_distances(X=X,
                                                 centers=centers,
                                                 labels=labels,
                                                 metric="euclidean")

        distances_cosine_calc = get_distances(X=X,
                                              centers=centers,
                                              labels=labels,
                                              metric="cosine")

        distances_mahalanobis_calc = get_distances(X=X,
                                                   centers=centers,
                                                   labels=labels,
                                                   metric="mahalanobis",
                                                   precisions=precisions)

        self.assertTrue(np.sum(np.abs(
            distances_euclidean_calc - distances_euclidean)) < 0.01)

        self.assertTrue(np.sum(np.abs(
            distances_cosine_calc - distances_cosine)) < 0.01)

        self.assertTrue(np.sum(np.abs(
            distances_mahalanobis_calc - distances_mahalanobis)) < 0.01)

    def test_sample_reviews(self):
        self.assertEqual(sample_reviews(reviews=data["text"],
                                        labels=labels,
                                        k=2,
                                        random_state=314),
                         sampled_dict)

    def test_get_close_matches_indexes(self):
        self.assertEqual(
            get_close_matches_indexes(word="great",
                                      possibilities=["super", "great", "nice"]),
            [1])
        self.assertEqual(
            get_close_matches_indexes(word="great",
                                      possibilities=["great", "great", "great"]),
            [2, 1, 0])  # are returned in reverse order

    def test_recycle(self):
        # note that we also test here that ir works when one review has no
        # sentence id
        sentence_ids = np.array([0, 0, 0, 1, 1, 3, 3, 3, 3])
        labels = np.array([0, 1, 1, 0])
        labels_recycled = [0, 0, 0, 1, 1, 0, 0, 0, 0]
        self.assertTrue(all(recycle(x=labels, sentence_ids=sentence_ids) ==
                            labels_recycled))

    def test_plot_distance_distribution(self):
        # actually only tests a subtask,  i.e. subsetting the upper triangular
        # matrix
        D = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        D_upper = D[np.triu_indices(n=D.shape[0], k=1)]
        self.assertTrue(all(D_upper == np.array([1, 2, 5])))

    def test_get_central_reviews(self):
        embeddings = np.array([[1, 1],
                               [2, 2],
                               [3, 3],
                               [4, 4],
                               [5, 5],
                               [6, 6]])
        centers = np.array([[2, 2],
                            [5, 5]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        reviews = ["this", "is", "just", "a", "test", "file"]
        central_embeddings = get_central_reviews(embeddings=embeddings,
                                                 centers=centers,
                                                 k=1,
                                                 text=reviews,
                                                 metric="euclidean",
                                                 labels=labels)
        self.assertTrue(central_embeddings[0]["text"][0] == "is")
        self.assertTrue(central_embeddings[0]["id"][0] == 1)
        self.assertTrue(central_embeddings[0]["distance"][0] == 0)

        self.assertTrue(central_embeddings[1]["text"][0] == "test")
        self.assertTrue(central_embeddings[1]["id"][0] == 4)
        self.assertTrue(central_embeddings[1]["distance"][0] == 0)

    def test_generate_parameter_combination(self):
        pc = generate_parameter_combination(names=["n_clusters", "n_init"],
                 values=[[5, 6], [3]])
        self.assertTrue(pc[0] == {"n_clusters": 5, "n_init": 3})
        self.assertTrue(pc[1] == {"n_clusters": 6, "n_init": 3})
        
    def test_format_precisions(self): 
        M1 = np.array(np.arange(500)).reshape(5, 10, 10)     
        M1_format = format_precisions(M1, n_clusters=5, n_features=10)
        assert_array_equal(M1, M1_format)
            
        M2 = np.array([1,2,3])
        M2_format = format_precisions(M2, n_features=2, n_clusters=3)
        assert_array_equal(M2_format, 
                           np.array([[[1, 0], [0, 1]], 
                                     [[2, 0], [0, 2]], 
                                     [[3, 0], [0, 3]]]))

        M3 = np.array([[1, 2], 
                       [2, 3]])
        M3_format = format_precisions(M3, n_clusters=2, n_features=3)
        assert_array_equal(M3_format,
                           np.array([[[1, 0], [0, 2]], 
                                     [[2, 0], [0, 3]]]))

        M4 = np.array(np.arange(4)).reshape(2, 2)
        M4_format = format_precisions(M4, n_clusters=3, n_features=2)
        assert_array_equal(M4_format, 
                           np.array([[[0, 1], [2, 3]], 
                                     [[0, 1], [2, 3]], 
                                     [[0, 1], [2, 3]]]))


if __name__ == "__main__":
    unittest.main()
