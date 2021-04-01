import unittest
import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from collections import Counter
from funs.preprocess import do_basic_correction, get_ngram_df, \
    get_term_frequency, is_asci, is_english, get_tf

# first we create synthetic reviews that are used for the testing

data = pd.DataFrame(["great machine", "loud machine", "great machine"],
                    columns=["text"])

unigrams = pd.DataFrame(
    {"unigram": ["great", "machine", "loud", "machine", "great", "machine"],
     "pos": ["ADJ", "NOUN", "ADJ", "NOUN", "ADJ", "NOUN"],
     "sentence_id": [0, 0, 1, 1, 2, 2],
     "unigram_stem": ["great", "machin", "loud", "machin", "great", "machin"]})

bigrams = pd.DataFrame(
    {"bigram": ["great machine", "loud machine", "great machine"],
     "sentence_id": [0, 1, 2],
     "bigram_stem": ["great machin", "loud machin", "great machin"]})

weights = [1.5, 2, 1.5]
labels = [0, 1, 0]

# correct outputs for term frequencies----------------------------------------

# unigrams

# without further restrictions
tf_unigram = Counter({"machine": 3, "great": 2, "loud": 1})

# without further restrictions and sorted
tf_unigram_sorted = pd.DataFrame(
    {"unigram": ["machine", "great", "loud"], "frequency": [3, 2, 1]})

# without further restrictions but weighted
tf_unigram_weighted = Counter({"machine": 5., "great": 3., "loud": 2.})

# nouns
tf_unigram_noun = Counter({"machine": 3})

# additional_stopwords (stopword is "loud")
tf_unigram_stopwords = Counter({"machine": 3,
                                "great": 2})

# with labels
tf_unigram_labelled = {
    0: Counter({"machine": 2, "great": 2}),
    1: Counter({"machine": 1, "loud": 1})}


# bigrams

# without further restrictions
tf_bigram = Counter({
    "great machine": 2,
    "loud machine": 1
})

# without further restrictions and sorted
tf_bigram_sorted = pd.DataFrame(
    {"bigram": ["great machine", "loud machine"],
     "frequency": [2, 1]})

# stopwords
tf_bigram_stopwords = Counter({
    "great machine": 2,
    "loud machine": 1
})

# labelled
tf_bigram_labelled = {
    0: Counter({"great machine": 2}),
    1: Counter({"loud machine": 1})
}


bigrams_calc = get_ngram_df(reviews=data["text"], n=2)
# base case
get_term_frequency(bigrams_calc)


class TestPreprocess(unittest.TestCase):

    def test_is_asci(self):
        self.assertTrue(is_asci("hallo"))
        self.assertFalse(is_asci("¿"))

    def test_is_english(self):
        self.assertTrue(np.abs(is_english("how are you") - 1) < 0.01)
        self.assertTrue(np.abs(is_english("Deutsche Sprache") - 0) < 0.01)

    def test_do_basic_correction(self):
        self.assertEqual(do_basic_correction("DOES IT WORK"),
                         "does it work")
        self.assertEqual(do_basic_correction("DOES  IT WORK"),
                         "does it work")
        self.assertEqual(do_basic_correction("DOES IT WORK?."),
                         "does it work.")
        self.assertEqual(do_basic_correction(" DOES IT WORK "),
                         "does it work")
        self.assertEqual(do_basic_correction(" DOES IT WOOORK "),
                         "does it work")
        self.assertEqual(do_basic_correction(" DOES IT WORK??? "),
                         "does it work?")
        self.assertEqual(do_basic_correction(" DOES.IT,WORK!THOUGH?HMM"),
                         "does. it, work! though? hmm")
        self.assertEqual(do_basic_correction(" IT WAS , GREAT"),
                         "it was, great")

    def test_get_ngram_df(self):
        assert_frame_equal(get_ngram_df(data["text"], n=1),
                           unigrams, check_dtype=False)
        assert_frame_equal(get_ngram_df(data["text"], n=2),
                           bigrams, check_dtype=False)

    def test_get_term_frequency(self):
        # unigrams
        # the calculated ungigrams
        unigrams_calc = get_ngram_df(data["text"], n=1)
        # base case
        self.assertEqual(get_term_frequency(unigrams_calc),
                         tf_unigram)
        # sorted
        assert_frame_equal(get_term_frequency(unigrams_calc, sorted_df=True),
                           tf_unigram_sorted)
        # weighted
        self.assertEqual(get_term_frequency(unigrams_calc, weights=weights),
                         tf_unigram_weighted)
        # nouns
        self.assertEqual(get_term_frequency(unigrams_calc, types="NOUN"),
                         tf_unigram_noun)
        # additional stopwords
        self.assertEqual(get_term_frequency(unigrams_calc,
                                            additional_stopwords="loud"),
                         tf_unigram_stopwords)
        # labelled
        self.assertEqual(get_term_frequency(unigrams_calc, labels=labels),
                         tf_unigram_labelled)

        # bigrams
        # the calculated bigrams
        bigrams_calc = get_ngram_df(reviews=data["text"], n=2)
        # base case
        self.assertEqual(get_term_frequency(bigrams_calc),
                         tf_bigram)
        # sorted
        assert_frame_equal(get_term_frequency(bigrams_calc, sorted_df=True),
                           tf_bigram_sorted)
        # stopwords
        self.assertEqual(get_term_frequency(bigrams_calc,
                                            additional_stopwords="loud"),
                         tf_bigram_stopwords),
        # labeled
        self.assertEqual(get_term_frequency(bigrams_calc, labels=labels),
                         tf_bigram_labelled)

        # this is a test for the weights, when setting the weights to 1,
        # it is the same result as without weights
        self.assertEqual(get_term_frequency(bigrams_calc,
                                            labels=labels,
                                            weights=np.array([1, 1, 1])),
                         tf_bigram_labelled)

    def test_get_tf(self):
        ngrams = ["hello", "what", "is", "hello"]
        weights = np.array([1, 2, 3, 4])

        wtf = get_tf(ngrams=ngrams,
                     name="ngram",
                     weights=weights,
                     sorted_df=True)

        assert_frame_equal(pd.DataFrame({"ngram": ["hello", "is", "what"],
                                         "frequency": [5, 3, 2]}),
                           wtf,
                           check_dtype=False)


if __name__ == "__main__":
    unittest.main()
