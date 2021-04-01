import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from funs.preprocess import get_term_frequency
from funs.utils import sample_reviews, recycle, get_central_reviews


def plot_ngram(ngram_df, name=None, k=10, labels=None, specific_label=None,
               sentiments=None, figsize=(15, 15), weights=None,
               additional_stopwords=None, make_title=True):
    """
    This function creates barplots for the frequency of ngrams in three possible
    ways: 
    - if labels are passed but specific_label is None one big plot is created 
    containing ngram barplots for each cluster 
    - if labels are passed and specific_label is passed only the ngram plot for 
    the specific_label is created
    - if labels and specific_label is None an ngram-plot for the complete 
    ngram_df is created 

    :param ngram_df: a data-frame containing at least the columns name and 
    "sentence_id" 
    :param name: the column name where the ngrams are stored in ngram_df, e.g.
    "unigram", "bigram" or "unigram_stem"
    :param k: the k most frequent ngrams are displayed in the barplot 
    :param labels: cluster labels 
    :param specific_label: plot for this label is created, if None plot for all 
    clusters is created 
    :param sentiments: the sentiments belonging to the reviews (if passed as an 
    argument they are included in the titles)
    :param figsize: the figsize 
    :param weights: weights used for the term frequencies, if None the term 
    frequencies are simply the counts. 
    :param additional_stopwords: any specific ngrams that should be removed 
    :param make_title: whether to add a title 

    :return: the barplot figure 
    """
    if name is None:
        if "unigram" in ngram_df.columns:
            name = "unigram"
        elif "bigram" in ngram_df.columns:
            name = "bigram"
        elif "ngram" in ngram_df.columns:
            name = "ngram"
        else:
            raise Exception("name not passed and could not be inferred")

    if labels is None and specific_label is None:
        df = get_term_frequency(df=ngram_df,
                                name=name,
                                weights=weights,
                                sorted_df=True,
                                additional_stopwords=additional_stopwords)
        fig, axs = plt.subplots(figsize=figsize)

        title = f"n = {len(set(ngram_df['sentence_id']))}"
        if sentiments is not None:
            average_sent = np.round(sentiments, 2)
            title = title + f", average sentiment = {average_sent}"

        sns.barplot(data=df[0:k], x="frequency", y=name, ax=axs)
        if make_title: 
            axs.set_title(title)
        axs.set_xlabel("")
        axs.set_ylabel("")

        return fig

    # labels are passed---------------------------------------------------------

    # create labels on ngram-level from labels on the review label
    ngram_labels = recycle(x=labels,
                           sentence_ids=ngram_df["sentence_id"])

    if specific_label is not None:
        df = get_term_frequency(df=ngram_df[ngram_labels == specific_label],
                                name=name,
                                weights=weights,
                                sorted_df=True,
                                additional_stopwords=additional_stopwords)
        fig, axs = plt.subplots(figsize=figsize)

        cluster_size = np.sum(labels == specific_label)
        title = f"label = {specific_label}, n = {cluster_size}"

        if sentiments is not None:
            av_sent = np.round(sentiments[labels == specific_label], 2)
            title = title + f", average sentiment = {av_sent}"

        sns.barplot(data=df[0:k], x="frequency", y=name, ax=axs)
        if make_title: 
            axs.set_title(title)
        axs.set_xlabel("")
        axs.set_ylabel("")

        return fig

    # plots for all clusters are created----------------------------------------
    df_list = list()
    label_list = list(set(labels))

    # first we create the term frequency-dfs for each of the labels
    for label in label_list:
        current_df = get_term_frequency(df=ngram_df[ngram_labels == label],
                                        name=name,
                                        weights=weights,
                                        sorted_df=True)
        df_list.append(current_df)

    n_clusters = len(label_list)
    ncol = 2
    nrow = int(np.ceil(n_clusters / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)

    for i in range(nrow):  # loop over the axes and create the plots
        for j in range(ncol):
            if i * ncol + j + 1 <= n_clusters:
                index = i * ncol + j
                label = label_list[index]
                k_adj = min(k, len(df_list[index]))
                title = f"label = {label}, n = {sum(labels == label)}"
                if sentiments is not None:
                    av_sent = np.round(np.mean(sentiments[labels == label]), 2)
                    title = title + f", average sentiment = {av_sent}"

                sns.barplot(data=df_list[index][0:k_adj], x="frequency", y=name,
                            ax=axs[i, j])
                
                if make_title: 
                    axs[i, j].set_title(title)
                axs[i, j].set_xlabel("")
                axs[i, j].set_ylabel("")
            else:  # creates an empty plot in case n_clusters is uneven
                fig.delaxes(axs[i, j])
    plt.tight_layout()
    plt.close()
    return fig


def plot_wordcloud(df, name="unigram", labels=None, specific_label=None,
                   sentiments=None, additional_stopwords=None, types=None,
                   figsize=(15, 15), max_words=20, weights=None, 
                   make_title=True):
    """
    This function creates wordclouds
    - when labels are passed and a specific label, only a wordcloud for this one 
    cluster is created 
    - when only labels are passed, a wordcloud for each cluster is created 
    - when neither labels nor specific_label is passed a wordcloud for the 
    complete df is created 
    Note that the labels and the sentiments need not be of the length of the 
    passed df, but can simply be corresponding to the original reviews 

    :param df: the unigram dataframe 
    :param name: the column name where the unigrams are stored 
    :param labels: the cluster labels 
    :param specific_labels: A specific label (if only a wordcloud for this 
    cluster should be created)
    :param sentiments: sentiments belonging to the reviews  
    :param additional_stopwords: A set (or list) containing additional stopwords 
    that are not already in the spacy stopwords
    :param types: Specify a list with word-types that are allowed in the wordcloud 
    see https://spacy.io/api/annotation; can be e.g. "NOUN" or "VERB" or "ADJ"
    :param figsize: the figsize 
    :param max_words: the maximal number of words that are displayed in one 
    wordcloud 
    :weights: whether the term frequencies used for the wordcloud are weighted
    :param make_title: whether to include a title 

    :return: figure
    """
    if additional_stopwords is None:
        additional_stopwords = set()

    stopwords = STOP_WORDS.union(additional_stopwords)

    # wordcloud for whole df----------------------------------------------------

    if labels is None and specific_label is None:
        term_frequency = get_term_frequency(
            df=df,
            name=name,
            types=types,
            additional_stopwords=additional_stopwords,
            weights=weights)

        wordcloud = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=max_words
        ).generate_from_frequencies(frequencies=term_frequency)

        fig, axs = plt.subplots(figsize=figsize)
        title = f"n = {max(df['sentence_id'])}"
        if sentiments is not None:
            average_sentiment = np.round(np.mean(sentiments), 2)
            title = title + f", average sentiment = {average_sentiment}"
        if make_title:
            axs.set_title(title)
        axs.imshow(wordcloud)
        # remove x and y ticks
        axs.set_xticks([])
        axs.set_yticks([])
        return fig

    # wordcloud for a specific label--------------------------------------------

    if specific_label is not None:
        recycled_labels = recycle(labels, df["sentence_id"])
        term_frequency = get_term_frequency(
            df=df[recycled_labels == specific_label],
            name=name,
            types=types,
            additional_stopwords=additional_stopwords,
            weights=weights)

        wordcloud = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=max_words
        ).generate_from_frequencies(frequencies=term_frequency)

        fig, axs = plt.subplots(figsize=figsize)
        cluster_size = sum(labels == specific_label)
        title = f"Wordcloud for cluster {specific_label}, n = {cluster_size}"
        if sentiments is not None:
            average_sentiment = round(np.mean([labels == specific_label]))
            title = title + f", average sentiment = {average_sentiment}"
        if make_title: 
            axs.set_title(title)
        axs.imshow(wordcloud)
        # remove x and y ticks
        axs.set_xticks([])
        axs.set_yticks([])
        return fig

    # wordclouds for all clusters-----------------------------------------------

    # term frequency is a list with counter objects, one for each cluster
    term_frequency_list = get_term_frequency(
        df=df,
        name=name,
        types=types,
        labels=labels,
        additional_stopwords=additional_stopwords,
        weights=weights)

    def do_wc(term_frequency):
        wordcloud = WordCloud(
            background_color="white",
            max_words=max_words
        ).generate_from_frequencies(frequencies=term_frequency)

        return wordcloud

    # create wordcloud for each cluster
    wordclouds = [do_wc(term_frequency_list[i]) for i in \
                  range(len(term_frequency_list))]

    label_list = list(set(labels))
    n_clusters = len(label_list)
    ncol = 2
    nrow = int(np.ceil(n_clusters / ncol))

    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)

    plt.subplots_adjust(hspace=0.3)

    for i in range(nrow):
        for j in range(ncol):
            if i * ncol + j + 1 <= n_clusters:  # if n_clusters is uneven one subplot
                # is empty
                index = i * ncol + j
                label = label_list[index]
                title = f"label = {label}, " \
                    f"n = {sum(labels == label)}"
                if sentiments is not None:
                    average_sentiment = round(
                        np.mean(sentiments[labels == label]), 2)
                    title = title + \
                        f", average_sentiment = {average_sentiment}"
                if make_title: 
                    axs[i, j].set_title(title)
                axs[i, j].imshow(wordclouds[index])
                # remove x and y ticks
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
            else:
                fig.delaxes(axs[i, j])  # this is potentially the empty plot
                # if number of plots is uneven
    plt.tight_layout()
    plt.close()
    return fig


def write_cluster_text(text_params, path, reviews, centers, random_state,
                       metric, sentence_embeddings, labels, sentiments=None,
                       precisions=None):
    """
    This function allows to create the following text-files: 
    - random reviews for a cluster -> path/random_sentences.txt
    - the central reviews for a cluster -> path/central_sentences.txt

    :param text_params: This parameter is a dictionary with the keys 
    "random_reviews" and "central_reviews" containing an integer that 
    indicates how many random sentences are to be printed for each cluster
    (if 0, the file is not created). 
    text_params = {"random_reviews" : 5, "central_reviews" : 0} means that 
    a text file is created that contains 5 randomly sampled reviews for the 
    cluster and no text file is created for the central reviews
    :param path: path to where the files are created 
    :param reviews: the reviews as pandas Series or list 
    :param centers: Numpy array with cluster centers  in the rows 
    :param random_state: the random_state to make it reproducible 
    :param metric: the metric for the distances ("euclidean", 
    "cosine" or "mahalanobis" were used in the project)
    :param sentence_embeddings: the sentence-level embeddings
    :param labels: the cluster labels; must be 0, 1, 2, ... 
    :param sentiments: the sentiments, are printed above each review if they 
    are passed 
    :param precisions: the precisions matrices (in case metric = "mahalanobis"; 
    as obtained by GaussianMixture from sklearn.mixture)

    :return: None     
    """
    label_list = list(set(labels))

    # create subtititles for the clusters---------------------------------------
    cluster_sizes = [np.sum(labels == label) for label in label_list]
    if sentiments is None:
        average_sentiments = ["NA" for i in label_list]
    else:
        average_sentiments = [round(np.mean(sentiments[labels == label]), 2) \
                              for label in set(labels)]

    subtitle_dict = {"label": label_list,
                     "n": cluster_sizes,
                     "sentiment": average_sentiments}

    if text_params["random"] > 0:
        text_list = sample_reviews(reviews=reviews,
                                   labels=labels,
                                   k=text_params["random"],
                                   random_state=random_state)
        write_text(text_list=text_list,
                   path=path + "random_sentences.txt",
                   sep="\n \n",
                   title="randomly sampled reviews",
                   subtitle_dict=subtitle_dict)

    if text_params["central_sentences"] > 0:
        similar_sentences = get_central_reviews(
            embeddings=sentence_embeddings,
            centers=centers,
            k=text_params["central_sentences"],
            text=reviews,
            metric=metric,
            precisions=precisions, 
            labels=labels)
        
        write_text(text_list=similar_sentences,
                   path=path + "central_sentences.txt",
                   sep="\n \n",
                   title=f"sentences that are close to the cluster centers",
                   subtitle_dict=subtitle_dict)

    return None


def write_text(text_list, path, sep, title, subtitle_dict, overwrite=False):
    """
    This functions writes text_list to a text file (where text list contains
    a list with sentences for each cluster)

    :param text_list: this is a list that contains an entry for each cluster. 
    In case the random reviews are printed, this is a list of strings. In case 
    the central sentences are printed this is a pandas DataFrame containing 
    the columns "distance" and "text"
    :param path: the path to the file to which the text data is written
    :param sep: how is text within a cluster seperated when writing 
    to the text file e.g. "\newline"
    :param title: Main title of the file 
    :param subtitle_dict: Each cluster can have it's own subtitle, 
    subtitle_dict can e.g. be {"sentiment" : [0.3, 0.5, 0.9], 
                               "n" : [10, 20, 25], 
                               "label" : [0, 1, 2]}
    this would mean the subtitle of the first list of text_list, i.e. text_list[0]
    would have the header: sentiment = 0.3, n = 10, label = 0,
    :param overwrite: Whether the file should be overwritten when it already
    exists

    :return: None                                
    """
    if path[-4:] != ".txt":
        path += ".txt"
    # ensure that we do not accidentally overwrite other files
    if os.path.exists(path) and not overwrite:
        raise Exception("File exists and overwrite is False")
    if os.path.exists(path) and overwrite: 
        print("File exists but is overwritten")

    with open(path, "w+", encoding="utf-8") as file:
        if title is not None:
            file.write(title + "\n \n \n")

        for i, _ in enumerate(text_list):
            text = text_list[i]
            if isinstance(text, pd.core.frame.DataFrame):
                # here we have the columns "text" and "distance"
                # (this is the case when we create the central sentences)
                text = list(text["distance"].apply(
                    lambda x: str(np.round(x, 2))) + ", " + text["text"])

            # write the subtitle to the file
            n_keys = len(subtitle_dict.keys())
            for j, key in enumerate(subtitle_dict.keys()):
                file.write(f"{key} = {subtitle_dict[key][i]}")
                if j < n_keys - 1: 
                    file.write(", ")
                    
            file.write("\n \n")

            for entry in text:
                file.write(entry)
                file.write(sep)

            file.write("\n \n")
    return None


def plot_topics(true_labels, labels=None, specific_label=None, figsize=(20, 15), 
                k=15, sentiments=None):    
    if labels is None or specific_label is not None: 
        if labels is None:
            term_frequency = Counter(true_labels)
        else: 
            term_frequency = Counter(true_labels[labels == specific_label])           
        term_frequency = \
            pd.DataFrame(sorted(term_frequency.items(), 
                                key=lambda x: x[1])[::-1])
        term_frequency.columns = ["true_label", "frequency"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.barh(np.arange(k), term_frequency["frequency"][0:k], 
                align="center")
        ax.set_yticks(np.arange(k))
        ax.set_yticklabels(term_frequency["true_label"][0:k])
        ax.set_xlabel("frequency")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.close() 
        
        return fig 

    # create plots for all labels 
    
    term_frequency_list = list() 
    label_list = list(set(labels))
    n_clusters = len(label_list)
    
    for label in label_list: 
        term_frequency = Counter(true_labels[labels == label])
        term_frequency = \
            pd.DataFrame(sorted(term_frequency.items(), 
                                key=lambda x: x[1])[::-1])
        term_frequency.columns = ["true_label", "frequency"]
        term_frequency_list.append(term_frequency)
        

        
    fig, axs = plt.subplots(n_clusters, 1, figsize=figsize)
    
    for i in range(n_clusters):
        if i < n_clusters:                  
            term_frequency = term_frequency_list[i]
            label = label_list[i]
            k_adj = min(k, len(term_frequency))
            title = f"label = {label}, " \
                f"n = {sum(labels == label)}"
            if sentiments is not None:
                average_sent = np.round(sentiments[labels == label], 2)
                title = title + f", average sentiment = {average_sent}" 
                
            axs[i].barh(np.arange(k_adj), 
                            term_frequency["frequency"][0:k_adj], 
            align="center")
            axs[i].set_yticks(np.arange(k_adj))
            axs[i].set_yticklabels(term_frequency["true_label"][0:k_adj])
            axs[i].set_xlabel("frequency")
            axs[i].invert_yaxis()

            axs[i].set_title(title)
        else:
            fig.delaxes(axs[i, j])  # this is potentially the empty plot
            # if number of plots is uneven
    plt.tight_layout()
    plt.close()
    
    return fig 
        
