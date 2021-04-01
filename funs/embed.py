import numpy as np
import pandas as pd
import sentence_transformers
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer
from funs.utils import get_close_matches_indexes


def get_embedding(model, reviews, batch_size=16, unigram_df=None,
                  token_info=False):
    """
    This function embeds the reviews using a Sentence Transformer model. 
    When token_info is True, the individual tokens embeddings and a 
    corresponding token_df is also returned. The token_df has the form
    "token"   | "sentence_id" | "pos"
    where "pos" is obtained using the spacy tokens (by finding the closest 
    unigram for the each token, if no match is found, "pos" is set to None)    

    Note that the sentence_embeddings slightly differ (magnitude of around 
    1e-07) depending on token_info is True, because the pooling is implemented 
    differently. 

    :param model: a SentenceTransformer model or a string that can be loaded
    via SentenceTransformer(model)
    :param reviews: a pandas Series or list containing the reviews 
    :param batch_size: The batchsize 
    :param unigram_df: When "token_df" is contained in components than the 
    unigram dataframe has to be passed (as obtained by get_ngram_df with n = 1)
    :param token_info: It true token_embeddings and token_df are also returned 

    :return: sentence_embeddings, (token_embeddings,  token_df) 
    """
    # input checking------------------------------------------------------------
    assert isinstance(
        model, (sentence_transformers.SentenceTransformer, str))
    if isinstance(model, str):
        try:
            model = SentenceTransformer(model)
        except:
            raise Exception("incorrect model string")
    assert isinstance(reviews, (list, pd.core.frame.DataFrame,
                                pd.core.series.Series))
    if isinstance(reviews, (pd.core.frame.DataFrame, pd.core.series.Series)):
        reviews = list(reviews)
    assert isinstance(batch_size, (int, np.int32, np.int64))
    assert isinstance(token_info, bool)
    if token_info:
        assert isinstance(unigram_df, pd.core.frame.DataFrame)
        assert "unigram" in unigram_df.columns
        assert "pos" in unigram_df.columns

    # get the pooling method that is used by the model
    options = ["pooling_mode_cls_token", "pooling_mode_mean_tokens",
               "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens"]

    chosen_option_logical = \
        [getattr(model._last_module(), option) for option in options]

    pooling_names = ["cls", "mean", "max", "mean_sqrt_len"]

    pooling = pooling_names[int(np.where(chosen_option_logical)[0])]

    # the token information can only be obtained for the mean and cls pooling
    # (could be extended though)
    if pooling not in ["cls", "mean"]:
        assert not token_info

    # early exit if only return sentence embeddings-----------------------------
    if not token_info:
        sentence_embeddings = model.encode(reviews,
                                           show_progress_bar=True,
                                           output_value="sentence_embedding",
                                           batch_size=batch_size)
        sentence_embeddings = np.vstack(sentence_embeddings)
        return sentence_embeddings

    # token_info is True--------------------------------------------------------

    # now we are in the case where not only the sentence_embeddings, but also
    # the information about the tokens is relevant. We first create the token
    # embeddings and then do the pooling manually
    token_embeddings = model.encode(reviews,
                                    show_progress_bar=True,
                                    output_value="token_embeddings",
                                    batch_size=batch_size)

    max_length = model.get_max_seq_length()

    # note that in python [0,1,2][0:100] gives back [0,1,2]
    bert_ids = [model.tokenize(text)[0:(max_length - 2)] for text in reviews]
    # (we have to add the tokens for [CLS] and [SEP] manually)
    bert_ids = [[101] + x + [102] for x in bert_ids]
    lengths = [len(x) for x in bert_ids]
    sentence_ids = np.repeat(np.arange(len(bert_ids)), lengths)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                              do_lower_case=True)

    # currently bert_ids is a list of lists that we have to flatten
    bert_ids = pd.Series([x for y in bert_ids for x in y])

    # decode ids to tokens
    tokens = pd.Series([tokenizer.decode([x]) for x in bert_ids])

    # now we have to obtain the POS-tags. the only tricky thing is that
    # the BERT-tokens and the spacy tokens might not necessarily conincide,
    # therefore we use the get_close_matches from the difflib module
    # that finds the matches between the bert tokens and the spacy unigrams
    # there are three possibilities:
    # 1. case: no close matches found -> pos set to None
    # 2. case: one close match found -> use that
    # 3. case: more than one close match found:
    #   e.g. indices [10, 45] (spacy tokens) while the bert token is 35
    #   we then want to obtain the token that is closest to the bert token.
    #   However we have to adjust to the different number of tokens / unigrams
    #   assume that the review contains 102 bert tokens but 50 spacy tokens
    #   when not adjusting for the length we would pick 35, because
    #   |35 - 45| = 10 and |10 - 25| = 20
    #   when adjusting for the length we calculate
    #   argmin(|10 * 100/50 - (35-1)|, |45 * 100/50 - (35 - 1)|) =
    #   argmin(14, 56) and we now pick the first one
    #   note that we use 100 instead of 102 because of the cls and sep token
    #   that don"t appear in the spacy tokens.

    bert_pos_list = list()

    for sentence_id in set(sentence_ids):  # only loop over every review once
        bert_tokens = list(tokens[sentence_ids == sentence_id])
        spacy_tokens = list(unigram_df["unigram"][unigram_df["sentence_id"] ==
                                                  sentence_id])
        spacy_pos = \
            list(unigram_df["pos"][unigram_df["sentence_id"] == sentence_id])

        n_bert_tokens = len(bert_tokens)
        n_spacy_tokens = len(spacy_tokens)

        # remove the padded tokens from the token embeddings
        # note that token_embeddings is a list such that token_embeddings[i]
        # contains a numpy array with the (padded) token_embeddings
        # the padding originates from the batching when calcualting the
        # embeddings
        token_embeddings[sentence_id] = \
            token_embeddings[sentence_id][:n_bert_tokens]
        for i, bert_token in enumerate(bert_tokens):
            if bert_token in ["[CLS]", "[SEP]"]:
                bert_pos_list.append(None)
                continue
            # get the closest token matches between the bert token and the
            # spacy tokens from the current review
            matches = get_close_matches_indexes(bert_token, spacy_tokens)
            if len(matches) == 0:
                bert_pos_list.append(None)
            elif len(matches) == 1:
                bert_pos_list.append(spacy_pos[matches[0]])
            else:  # see explanation above
                # in case of multiple matches the first on is chosen
                # (this behaviour is inherited from np.argmin in adjusted_argmin)
                # note that the index of the current bert token is i - 1 and
                # not i (as we don"t consider the CLS and SEP token)
                closest_index = adjusted_argmin(index=i - 1,
                                                spacy_matches=matches,
                                                n_bert=n_bert_tokens - 2,
                                                n_spacy=n_spacy_tokens)
                bert_pos_list.append(spacy_pos[matches[closest_index]])

    # calculate the sentence embeddings from the token embeddings
    # note that the padded tokens were already removed in the previous loop

    if pooling == "mean":
        sentence_embeddings = [np.apply_along_axis(np.mean, 0, x)
                               for x in token_embeddings]
    else:  # "cls"
        sentence_embeddings = [x[0, :] for x in token_embeddings]

    sentence_embeddings = np.vstack(sentence_embeddings)

    token_embeddings = np.vstack(token_embeddings)

    bert_token_df = pd.DataFrame({"token": tokens,
                                  "sentence_id": sentence_ids,
                                  "pos": bert_pos_list})

    return sentence_embeddings, token_embeddings, bert_token_df


def adjusted_argmin(index, spacy_matches, n_bert, n_spacy):
    """
    Does the adjustment as explained above.

    :param index: the index of the bert sequence (without CLS, and SEP )
    :param spacy_matches: the indices in the spacy_sequence that are close 
    matches (output of get_close_matches_indices)
    :param n_bert: number of bert tokens (should be without cls and sep)
    :param n_spacy: number of spacy tokens 

    :return: the index with the closest match     
    """
    x = np.argmin(np.abs(index * (n_spacy / n_bert) - np.array(spacy_matches)))
    return x
