import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from coclust.clustering import SphericalKmeans
from funs.utils import format_precisions, save_pickle, path_creator
from funs.cluster import SKMeans
from funs.interpret import plot_ngram, plot_wordcloud, \
    write_cluster_text
from funs.diagnostics import plot_center_distance_density, \
    plot_distance_between_centers, plot_silhouettes


def save_topic_model(model, path, reviews, metric, unigram_df, ngram_df, 
                     sentence_embeddings, random_state=None, sentiments=None,
                     wordcloud_params=None, ngram_params=None,
                     text_params=None, diagnostic_params=None, labels=None,
                     centers=None, precisions=None):
    """
    This function stores the model as a pkl file and creates plots and txt-files
    that enable the interpretation of the clusters. 

    A list of the files that can be saved: 
    - the model
    - wordclouds 
    - ngram barplots 
    - random sentences of each cluster 
    - central reviews for each cluster
    - heatmap for distances between cluster centers 
    - kernel density estimates for the distribution of center-distances for each 
    cluster 
    - silhouette plot   
    Note that when the metric is "mahalanobis" in case of Gaussian Mixture 
    clustering, some of these functions (e.g. silhouette plots) take some time
    and are disabled by default     

    :param model: The model object (either of class KMeans from sklearn.cluster, 
    SphericalKmeans from coclust, SKMeans from funs.clustering.cluster 
    or GaussianMixture from sklearn.cluster)
    :param path: path to a folder where the files will be stored 
    :param reviews: e.g. a list or pandas Series containing the reviews 
    :param metric: "cosine", "euclidean" or "mahalanobis" 
    :param unigram_df: the unigram_df as obtained by 
    funs.preprocess.get_ngram_df (for the wordclouds)
    :param ngram_df: the ngram_df as obtained by funs.preprocess.get_ngram_df
    (for the ngram barplots)
    :param sentence_embeddings: the sentence embeddings from 
    funs.embed.get_embedding
    :param random_state: the seed (when sampling the sentences for a cluster)
    :param sentiments: the sentiments of the reviews: numpy array or pandas 
    Series 
    :param wordcloud_params: a dictionary containing the entries: 
        - "names" : default is ["unigram", "unigram_stem"]. when only unigrams 
                    should be created it would be ["unigram"]
        - "figsize" : default is (15, 30)
        - "types" : a list of lists containing the types that are to be used
        the default is [["NOUN", "PROPN], ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]]
        which means that one wordcloud uses only nouns and proper nouns, and the 
        other one nouns, verbs, adjectives, adverbs and proper nouns. 
        the wordclouds for all possible combinations in "names" and "types" are 
        then created. When one e.g. only wants to change the figsize but leave
        the rest as the defaults, it is only necessary to specify the figsize
        wordcloud_params = {"figsize" = (5, 10)}, the rest of the arguments 
        still uses the defaults. 
        - "max_words" the maximum number of words that is display in the 
        wordcloud; default is 20 
        - "additional_stopwords" : a dictionary containing unigrams that are 
            removed when creating the wordclouds (in addition to STOP_WORDS from 
            spacy); e.g. {"unigram" : ["amazon"], "bigram" : ["amazon web"]} 
            would mean that "unigram" has the additional stopword "amazon"
            and the bigrams "amazon web" 
    :param ngram_params: the arguments that drive the creation of the ngram 
    plots, same principle as wordcloud_params. 
        - "names" : column names in ngram_df that are used to create the 
                    barplots, e.g. ["bigram", "bigram_stem"]
        - "figsize" : the figsize for the barplots 
        - "k" : the k most frequent ngrams are plotted 
        - "additional_stopwords" : see above at "wordcloud_params"
    The function was so far only used for bigrams, but should in principle also 
    be useable for trigrams etc. 
    :param text_params: A dict containing the entries 
         - "random" : an integer indicating how many reviews are sampled for 
         each cluster ; default is 10; 0 means no text file is created
        - "central" : default is 10, meaning for each cluster the 10 reviews 
            that are close to the respective center are chosen. 0 means no 
            text file is created 
    :param diagnostic_params: a dict containing the entries: 
        - "center_distance_density" : figsize for center distance density plots; 
            default is (7, 15); None means no plot is created
        - "distance_between_clusters" : figsize for heatmap for distance 
            between centers; default is (7, 15); None means no plot is created
        - "silhouettes" : figsize for silhouette plot; default is (7, 15); 
            None means no plot is created 
    :param labels: if model has no attribute labels_, the labels can be 
    passed via this argument (e.g. if labels were manually changed via 
    the function merge_clusters)
    :param centers: the centers of the cluster (if e.g. labels were changed
    manually)
    :precisions: The precision matrices for the clusters in case of EM 
    clustering

    :return: None 
    """
    # input checking------------------------------------------------------------
    assert isinstance(
        model, (KMeans, SKMeans, SphericalKmeans, GaussianMixture))
    assert path[-1] in ["/", "\\"]
    path = path_creator(path)  # this creates e.g. ./folder(1)/ when ./folder/
    # already exists
    assert metric in ["cosine", "euclidean", "mahalanobis"]
    assert isinstance(random_state, (int, np.int32, np.int64)) or \
        random_state is None 

    if labels is None:
        try:
            labels = model.labels_
        except:
            raise Exception("centers is None and model has no attribute \
                            labels_")
    if not hasattr(model, "labels_"):
        model.labels_ = labels
    label_set = set(labels)
    n_clusters = len(label_set)

    # labels have to be 0, 1, ..., g otherwise some functions might be failing
    assert label_set == set([i for i in range(n_clusters)])
    if centers is None:
        if isinstance(model, (KMeans, SKMeans, SphericalKmeans)):
            try:
                centers = model.cluster_centers_
            except:
                raise Exception("centers is None and model has no attribute \
                                cluster_centers_")
        if isinstance(model, GaussianMixture):
            try:
                centers = model.means_
            except:
                raise Exception("no centers passed and model has no attribute \
                                 means_")

    if metric == "mahalanobis":
        assert isinstance(model, GaussianMixture)
        if precisions is None:
            precisions = model.precisions_

    # now we check that the all the dimensions fit together
    # reviews, labels, sentence_embeddings, sentiments
    assert len(labels) == len(reviews)
    assert len(sentence_embeddings) == len(labels)
    assert sentiments is None or len(sentiments) == len(reviews)
    assert centers.shape[0] == n_clusters
    assert sentence_embeddings.shape[1] == centers.shape[1]
    if precisions is not None:
        precisions = format_precisions(precisions=precisions, 
                                       n_clusters=n_clusters, 
                                       n_features=centers.shape[1])
        assert precisions.shape[0] == n_clusters
        assert precisions.shape[1] == sentence_embeddings.shape[1]
        assert precisions.shape[2] == sentence_embeddings.shape[1]

    # test and fill with defaults
    ngram_params, wordcloud_params, text_params, diagnostic_params = \
        check_params(ngram_params=ngram_params,
                     wordcloud_params=wordcloud_params,
                     text_params=text_params,
                     diagnostic_params=diagnostic_params,
                     metric=metric)
    assert set(wordcloud_params["names"]).issubset(set(unigram_df.columns))
    assert set(ngram_params["names"]).issubset(set(ngram_df.columns))

    os.mkdir(path)

    # save the model
    save_pickle(object=model, path=path + "model.pkl")

    # wordclouds and barplots for ngrams----------------------------------------

    # create and save the ngram plots
    for nam in ngram_params["names"]:
        if nam in ngram_params["additional_stopwords"].keys():
            additional_stopwords = ngram_params["additional_stopwords"][nam]
        else: 
            additional_stopwords = None 
            
        ngram_plot = plot_ngram(
            ngram_df=ngram_df,
            name=nam,
            k=ngram_params["k"],
            labels=labels,
            sentiments=sentiments,
            figsize=ngram_params["figsize"],
            additional_stopwords=additional_stopwords)
        file_path = path + nam + ".pdf"
        file_path = file_path.lower()
        ngram_plot.savefig(file_path)

    # create and save the wordcloud plots
    for nam in wordcloud_params["names"]:
        for types in wordcloud_params["types"]:
            if nam in wordcloud_params["additional_stopwords"].keys():
                additional_stopwords = \
                    wordcloud_params["additional_stopwords"][nam]
            else: 
                additional_stopwords = None 
                
            wordcloud_plot = plot_wordcloud(
                df=unigram_df,
                name=nam,
                labels=labels,
                sentiments=sentiments,
                additional_stopwords=additional_stopwords,
                types=types,
                figsize=wordcloud_params["figsize"])

            if types is None:
                file_path = path + "wordcloud_" + nam + ".pdf"
            else:
                file_path = path + "wordcloud_" + nam + "_" +  \
                    "_".join(types) + ".pdf"
            file_path = file_path.lower()
            wordcloud_plot.savefig(file_path)

    # write the text files------------------------------------------------------

    write_cluster_text(text_params=text_params,
                       path=path,
                       reviews=reviews,
                       centers=centers,
                       random_state=random_state,
                       metric=metric,
                       sentence_embeddings=sentence_embeddings,
                       labels=labels,
                       sentiments=sentiments,
                       precisions=precisions)

    # diagnostic plots----------------------------------------------------------
    if diagnostic_params["center_distance_density"] is not None:
        center_distance_density = plot_center_distance_density(
            X=sentence_embeddings,
            centers=centers,
            labels=labels,
            precisions=precisions,
            metric=metric,
            figsize=diagnostic_params["center_distance_density"]
        )
        center_distance_density.savefig(path + "center_distance_density.pdf")

    if diagnostic_params["distance_between_clusters"] is not None:
        distance_between_clusters = plot_distance_between_centers(
            centers=centers,
            metric=metric,
            figsize=diagnostic_params["distance_between_clusters"]
        )

        distance_between_clusters.savefig(
            path + "distance_between_clusters.pdf")

    if diagnostic_params["silhouettes"] is not None:
        silhouettes = plot_silhouettes(
            X=sentence_embeddings,
            labels=labels,
            metric=metric,
            figsize=diagnostic_params["silhouettes"]
        )

        silhouettes.savefig(path + "silhouettes.pdf")

    return None


def check_params(ngram_params, wordcloud_params, text_params,
                 diagnostic_params, metric):
    """
    This function takes care of checking the parameters and handling the 
    defaults for ngram_params, wordcloud_params, text_params and diagnostic 
    params. 
    """
    ngram_params_default = {"names": ["bigram", "bigram_stem"],
                            "figsize": (15, 30),
                            "k": 10,
                            "additional_stopwords": dict()}

    wordcloud_params_default = {"names": ["unigram", "unigram_stem"],
                                "figsize": (15, 30),
                                "types": [["NOUN", "VERB", "ADJ", "ADV",
                                           "PROPN", "PART"]],
                                "additional_stopwords": dict(),
                                "max_words": 20}

    text_params_default = {"random": 10,
                           "central_sentences": 10}
    if metric == "mahalanobis":  # in this case it takes quite some time,
        # therefore the default is to not create the plots
        diagnostic_params_default = {"center_distance_density": None,
                                     "distance_between_clusters": None,
                                     "silhouettes": None}
    else:
        diagnostic_params_default = {"center_distance_density": (7, 15),
                                     "distance_between_clusters": (7, 7),
                                     "silhouettes": (7, 15)}

    if ngram_params is None:
        ngram_params = ngram_params_default
    else:
        # check that all the manually passed parameters are actual parameters
        assert set(ngram_params.keys()).issubset(
            set(ngram_params_default.keys()))
        for key in ngram_params_default.keys():
            # fill those parameters that were not passed with the defaults
            if not key in ngram_params.keys():
                ngram_params[key] = ngram_params_default[key]

    if wordcloud_params is None:
        wordcloud_params = wordcloud_params_default
    else:
        assert set(wordcloud_params.keys()).issubset(
            set(wordcloud_params_default.keys()))
        for key in wordcloud_params_default.keys():
            if not key in wordcloud_params.keys():
                wordcloud_params[key] = wordcloud_params_default[key]

    if text_params is None:
        text_params = text_params_default
    else:
        assert set(text_params.keys()).issubset(
            set(text_params_default.keys()))
        for key in text_params_default.keys():
            if not key in text_params.keys():
                text_params[key] = text_params_default[key]

    if diagnostic_params is None:
        diagnostic_params = diagnostic_params_default
    else:
        assert set(diagnostic_params.keys()).issubset(
            set(diagnostic_params_default.keys()))
        for key in diagnostic_params_default.keys():
            if not key in diagnostic_params.keys():
                diagnostic_params[key] = diagnostic_params_default[key]

    return ngram_params, wordcloud_params, text_params, diagnostic_params
