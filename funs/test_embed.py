import unittest
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas._testing import assert_frame_equal
from sentence_transformers import SentenceTransformer, models
import spacy
from funs.preprocess import get_ngram_df
from funs.embed import adjusted_argmin, get_embedding

sp = spacy.load("en_core_web_sm")

reviews = ["great machine", "love it"]
data = pd.DataFrame({"text": reviews})

tokens = ["[CLS]", "great", "machine", "[SEP]", "[CLS]", "love", "it", "[SEP]"]
sentence_ids = [0, 0, 0, 0, 1, 1, 1, 1]
pos_tags = [None, "ADJ", "NOUN", None, None, "VERB", "PRON", None]

unigram_df = get_ngram_df(reviews=reviews, n=1)

bert_token_df = pd.DataFrame(
    {"token": tokens, "sentence_id": sentence_ids, "pos": pos_tags}
)

model = SentenceTransformer("bert-base-nli-stsb-mean-tokens")

token_embeddings = model.encode(reviews,
                                output_value="token_embeddings",
                                batch_size=1)
token_embeddings = np.vstack(token_embeddings)

sentence_embeddings = model.encode(reviews,
                                   output_value="sentence_embedding",
                                   batch_size=1)

sentence_embeddings = np.vstack(sentence_embeddings)

reviews2 = ["love it. My greatest love. Really loving it "]
reviews2_sp = sp(reviews2[0])
pos_tags2 = [x.pos_ for x in reviews2_sp]

bert_raw = models.Transformer("bert-base-uncased")

# selecting the pooling operation isn't that important because we work
# with the token_embeddings anyway (see get_complete_embedding)
pooling = models.Pooling(bert_raw.get_word_embedding_dimension(),
                         pooling_mode_mean_tokens=False,
                         pooling_mode_cls_token=True,
                         pooling_mode_max_tokens=False)

bert = SentenceTransformer(modules=[bert_raw, pooling])


class TestEmbed(unittest.TestCase):

    def test_adjusted_argmin(self):
        self.assertEqual(adjusted_argmin(5, [2, 5], 20, 10),
                         0)
        self.assertEqual(adjusted_argmin(13, [10, 14], 100, 80),
                         0)
        self.assertEqual(adjusted_argmin(13, [2, 5], 100, 80),
                         1)

    def test_get_embedding(self):
        # here we test that our manual construction of the sentence embeddings
        # coincides with the SBert embeddings (i. e. whether the pooling was
        # done correctly) and also whether the bert_token_df is correctly
        # constructed
        sentence_embeddings1, token_embeddings1, bert_token_df1 = get_embedding(
            model=model, reviews=reviews, unigram_df=unigram_df,
            token_info=True
        )
        assert_allclose(sentence_embeddings, sentence_embeddings1, atol=0.0001)
        assert_allclose(token_embeddings, token_embeddings1, atol=0.0001)
        assert_frame_equal(bert_token_df, bert_token_df1, check_dtype=False)

        # now we check that the manual pooling for cls is also consistent with
        # the sbert inbuilt cls pooling
        sentence_embeddings2 = get_embedding(
            model=bert, reviews=reviews, unigram_df=unigram_df, token_info=False
        )
        sentence_embeddings2 = np.vstack(sentence_embeddings2)

        sentence_embeddings3 = get_embedding(
            model=bert, reviews=reviews, unigram_df=unigram_df, token_info=False
        )

        assert_allclose(sentence_embeddings2, sentence_embeddings3)


if __name__ == "__main__":
    unittest.main()
