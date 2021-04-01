import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from collections import Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import textacy
from nltk.stem.porter import PorterStemmer
from langdetect import detect
from funs.utils import has_key, recycle

# correction and verification---------------------------------------------------


def is_asci(sent):
    """
    This function checks whether all the characters in a string are ASCI

    :param sent: string to be checked 

    :return: bool
    """
    for letter in sent:
        if not 0 <= ord(letter) <= 127:
            return False

    return True


def is_english(sent):
    """
    This function checks whether a given string is english

    :param sent: the string to be checked 

    :return: 1 if english, 0 if not, 2 if unclear
    """
    try:
        if detect(sent) == "en":
            return 1
        else:
            return 0
    except:
        return 2


def do_basic_correction(sentence):
    """
    This function does some basic corrections. 

    Example for each operation: 
    - "Great Machine" -> "great machine"
    - " great machine " -> "great machine"
    - "great  machine" -> "great machine"
    - "greeeat machine" -> "great machine" 
      (but "greeat machine" stays the same)
    - "great machine!!!" -> "great machine"
    - "great machine!." -> "great machine."
    - "great.Machine" -> "great. Machine"
    - "great , machine" -> "great, machine"

    :param sentence: a string that is to be corrected  

    :return: corrected string     
    """
    # lowercase
    sentence = sentence.lower()
    # remove whitespace at the end and beginning
    sentence = sentence.strip()
    # more than one whitespace to one whitespace
    sentence = re.sub("\s+", " ", sentence)
    # letter repetition (if more than 2) is reduced to one letter
    sentence = re.sub(r"([a-z])\1{2,}", r"\1", sentence)
    # remove non-word repetition (eg. "hallo!!!" -> "hallo!")
    sentence = re.sub(r"([\W+])\1{1,}", r"\1", sentence)
    # normalization 8: xxx[?!]. -- > xxx.
    sentence = re.sub(r"\W+?\.", ".", sentence)
    # missing whitespace after punctuation
    sentence = re.sub(r"(\.|\?|\!|\,)(\w)", r"\1 \2", sentence)
    # "great , despite" -> "great, despite"
    sentence = re.sub(r'\s([?.!,;:"](?:\s|$))', r"\1", sentence)

    return sentence


# ngrams------------------------------------------------------------------------

def get_ngram_df(reviews, n, stem=True, spacy_model="en_core_web_sm", **kwargs):
    """
    This function creates a dataframe that contains ngrams
    If stem == True the ngrams are also stemmed (as an additional column). 

    Two cases are considered: 
    - n = 1 -> unigrams
    - n > 2 -> bigrams, trigrams etc. 
    In case n == 1 (and stem is True) the output has the following form: 
    "unigram" | "sentence_id" | "pos"  | "unigram_stem"
    ---------------------------------------------------
    "great    |       0       | "ADJ"  |    "great"
    "washing" |       0       | "NOUN" |    "wash"
    "machine" |       0       | "NOUN" |   "machin"
    "I"       |       1       |  ...
    "love"    |       1       |  ...
    ...

    - the column "unigram" simply contains the unigrams
    - the column "sentence_id" contains a number that indicates to which review
    the unigram belongs
    - the column "pos" is the pos-tag as obtained from the spacy pos-tagger
    - the "unigram_stem" is only included if stem is True 

    in case n == 2 (and stem is True), the output has the following form: 
    "bigram"          | "sentence_id" | "bigram_stem"
    -------------------------------------------------
    "great washing"   |       0       | "great wash"
    "washing machine" |       0       | "wash machin"
    "I love"          |       1       | ...


    again the "bigram_stem" column is only contained if stem is True 

    in case n >= 3 the column is named "ngram" but otherwise works as n == 2

    :param reviews: a pandas Series or list containing the reviews 
    :param stem: logical indicating whether ngrams should be stemmed
    :param spacy_model: the model string that is loaded with spacy.load
    see here: https://spacy.io/usage/models
    :param **kwargs: additional parameters that can be passed to 
    textacy.extract.ngrams

    :return: a dataframe as described above 

    """
    assert isinstance(reviews,
                      (list, pd.core.frame.Series))
    assert isinstance(n, (int, np.int32, np.int64))
    assert isinstance(stem, bool)
    assert isinstance(spacy_model, str)

    if n == 1:
        return get_unigram_df_main(reviews=reviews,
                                   stem=stem,
                                   spacy_model=spacy_model)
    else:
        return get_ngram_df_main(reviews=reviews,
                                 n=n,
                                 stem=stem,
                                 spacy_model=spacy_model,
                                 **kwargs)


def get_unigram_df_main(reviews, stem, spacy_model):
    """
    This function is the worker that creates the unigram dataframe with pos 
    tags and the respective sentence labels. 

    For further information see the doc of get_ngram_df 
    """
    # load the spacy model (this is a CNN)
    try:
        sp = spacy.load(spacy_model)
    except:
        raise Exception("model string not available")

    n_reviews = len(reviews)
    print("tokenize words using spacy")

    # apply the model to all the reviews
    doc_list = [sp(review) for review in tqdm(reviews)]

    # create a df with the text and pos-tag
    df = pd.DataFrame(
        [[token.text, token.pos_] for doc in doc_list for token in doc],
        columns=["unigram", "pos"])

    # create the column "sentence_id"
    sentence_lengths = [len(doc) for doc in doc_list]
    sentence_ids = np.repeat(np.arange(n_reviews), sentence_lengths)
    df["sentence_id"] = sentence_ids

    # stem the tokens
    if stem:
        stemmer = PorterStemmer()
        unigrams_stemmed = [stemmer.stem(unigram) for unigram in df["unigram"]]
        df["unigram_stem"] = unigrams_stemmed

    return df


def get_ngram_df_main(reviews, n, stem, spacy_model, **kwargs):
    """
    This function is the worker that creates a pandas dataframe that 
    contains the ngram dataframe and respective sentence labels 

    For further documentation see the doc of get_ngram_df
    """
    # load the spacy model (a CNN)
    try:
        sp = spacy.load(spacy_model)
    except:
        raise Exception("model string not available")
    n_reviews = len(reviews)
    print("tokenize words using spacy")

    # apply the model to the reviews
    doc_list = [sp(review) for review in tqdm(reviews)]

    # create the ngrams (list of lists)
    ngrams_list = [list(textacy.extract.ngrams(doc, n, **kwargs)) for
                   doc in doc_list]

    # get the number of ngrams in each review:
    ngram_lengths = [len(ngram) for ngram in ngrams_list]

    # unpack the ngrams
    ngrams = [str(ngram) for ngrams in ngrams_list for ngram in ngrams]

    sentence_ids = np.repeat(np.arange(n_reviews), ngram_lengths)

    # create name for the column that contains the ngrams
    if n == 1:
        name = "unigram"
    if n == 2:
        name = "bigram"
    else:
        name = "ngram"

    output = pd.DataFrame(
        {name: ngrams,
         "sentence_id": sentence_ids})

    # stem the tokens
    if stem:
        stemmer = PorterStemmer()

        def stem_ngram(ngram):
            words = ngram.split()
            return " ".join([stemmer.stem(word) for word in words])

        output[name + "_stem"] = output[name].apply(stem_ngram)

    return output

# term frequency tables---------------------------------------------------------


def get_term_frequency(df, name=None, types=None, labels=None, weights=None,
                       additional_stopwords=None, sorted_df=False):
    """
    This function creates a term frequency table for a dataframe that is created 
    by get_ngram_df. Either one for the whole df is created (when labels is None) 
    when one term frequency dict is created for each 
    cluster, i.e. the return is a dict
    where output[k] contains the term-frequencies for label k. 
    If weights are passed each objects is weighed according to it's weight and 
    not as 1 

    :param df: a ngram-dataframe as obtained by the function 
    get_ngram_df 
    :param name: the name of the column in df that contains the ngrams
    e.g. "unigram", "bigram" or "unigram_stem", if kept at None it is assigned 
    to "unigram", "bigram" or "ngram" if the name exists in df.columns
    :param types: df has a column "pos" that can be used to e.g. subset 
    words. So only the data for which data["pos"] is in types are considered
    when creating the term frequencies. If None all types are accepted 
    :param labels: cluster labels as numpy array or list 
    :param weights: a numpy array or list that contains the weights, if None
    simple counting is done
    :param additional_stopwords: basic stopwords from spacy are removed by 
    default. in some cases it is useful to remove further task-specific stopwords
    that can be passed as a set or list; None means no additional stopwords 
    :param sorted_df: If true the term-frequencies are a sorted dataframe, 
    otherwise they are a Counter object 

    :return: the term frequencies 
    """
    # input checking------------------------------------------------------------
    assert isinstance(df, pd.core.frame.DataFrame)
    assert name is None or isinstance(name, str)
    if name is not None:
        assert name in df.columns
    else:
        if "unigram" in df.columns:
            name = "unigram"
        elif "bigram" in df.columns:
            name = "bigram"
        elif "ngram" in df.columns:
            name = "ngram"
        else:
            raise Exception("name is none und was could not be identified")
    assert types is None or isinstance(types, (list, str, set))
    if isinstance(types, str):
        types = [types]
    if isinstance(types, list):
        assert "pos" in df.columns
    assert labels is None or isinstance(labels, (list, np.ndarray))
    # we have to ensure that weights and labels are numpy arrays because
    # if they remain a list we cannot subset via e.g. [2,5,7]
    if isinstance(labels, list):
        labels = np.array(labels)
    assert weights is None or isinstance(weights, (list, np.ndarray))
    if isinstance(weights, list):
        weights = np.array(weights)

    if additional_stopwords is None:
        additional_stopwords = set()
    assert isinstance(additional_stopwords, (set, list, str))
    if isinstance(additional_stopwords, str):
        additional_stopwords = [additional_stopwords]

    assert isinstance(sorted_df, bool)
    assert "sentence_id" in df.columns

    # recycle weights and labels if required------------------------------------
    if labels is not None and len(labels) != len(df):
        labels = recycle(labels, df["sentence_id"])
    if weights is not None and len(weights) != len(df):
        weights = recycle(weights, df["sentence_id"])

    # subset relevant ngrams----------------------------------------------------
    if name == "unigram":
        # in case of unigrams we include the standard stopwords from spacy
        stopwords = STOP_WORDS.union(additional_stopwords)
    else:
        stopwords = additional_stopwords

    relevant = df[name].apply(lambda x: x not in stopwords)

    if types is not None:
        relevant_type = df["pos"].apply(lambda x: x in types)
        relevant = relevant & relevant_type

    relevant_df = df[[name, "sentence_id"]][relevant]

    if labels is not None:
        labels = labels[relevant]
    if weights is not None:
        weights = weights[relevant]

    # calculate the term frequencies--------------------------------------------
    if labels is None:
        return get_tf(ngrams=relevant_df[name], name=name, weights=weights,
                      sorted_df=sorted_df)
    else:
        output = dict()
        current_weights = None
        for label in set(labels):
            # ATTENTION:
            # not that it is important here that we subset the dataframe
            # with a logical vector and not the indices, because then
            # one would have to pay attention to resetting the indices of the
            # pandas dataframe when constructing the relevant_df
            if weights is not None:
                current_weights = weights[labels == label]
            output[label] = get_tf(ngrams=relevant_df[name][labels == label],
                                   name=name,
                                   weights=current_weights,
                                   sorted_df=sorted_df)

    return output


def get_wtf_only(ngrams, weights):
    """
    Gets Weighted Term Frequency 
    """
    if isinstance(weights, (pd.core.frame.DataFrame, pd.core.series.Series)):
        weights = weights.to_numpy()
    output = dict()
    for i, word in enumerate(ngrams):
        if has_key(output, word):
            output[word] += weights[i]
        else:
            output[word] = weights[i]
    return output


def get_tf(ngrams, name, weights, sorted_df):
    """
    gets term frequency or weighted term frequency and convertes to sorted
    dataframe if wanted
    """
    if weights is not None:
        output = get_wtf_only(ngrams, weights)
    else:
        output = Counter(ngrams)

    if sorted_df:
        output = pd.DataFrame(sorted(output.items(), key=lambda x: x[1])[::-1])
        output.columns = [name, "frequency"]

    return output
