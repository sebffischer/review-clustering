_Authors_: Sebastian Fischer

## Summary

This repository implements easy-to-use functionality for topic clustering
of customer reviews, consisting of the following steps:

1. Embed customer reviews using [S-BERT embeddings](https://arxiv.org/abs/1908.10084)
2. Partition the embeddings using clustering algorithms
3. Interpret the resulting clusters via Word-Clouds, N-grams, and the
   central reviews of the respective partition

It was my part of the statistical consulting project during my Statistics
master at **LMU Munich** in 2020.

## Functionality

- **utils** - Utility functions that are used in the other files
- **preprocess** - Functions for preprocessing the reviews. This includes some
  functions for input correction, as well as the creation of dat aframes that
  contain n-grams for the customer reviews. For unigrams this also includes the
  part of speech (pos) tag, that can later be used when interpreting the results.  
  The ngrams and the pos tags are obtained using the [spacy](https://spacy.io/)
  and [textacy](https://pypi.org/project/textacy/) libraries.
- **embed** Includes functionality to obtain the embeddings for the reviews.
  It is mostly built upon the
  [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
  repository. The only thing that it adds, is that it is possible to obtain the
  pos tags for BERT tokens. These are obtained by matching the BERT tokens with
  the spacy tokens and using the spacy pos tag. When the individual token
  embeddings are not needed one can also directly use the Sentence-Transformer
  library.
- **cluster** This includes functions for hyperparameter selection using the
  elbow criterion and silhouette plots for (Spherical) K-Means and AIC/
  BIC-selection for Gaussian Mixtures. In addition to that it contains an
  implementation of Spherical K-Means and a function to manually merge clusters.
  We implemented Spherical K-Means ourselves as the implementations we found were
  optimized for sparse vectors and therefore quite slow when using them on dense
  vectors.
- **interpret** This file contains functionality to create n-ngram plots and
  wordclouds for existing clusters and write random reviews or the most central
  reviews to text files. It also contains a function that plots the distribution
  of one set of labels against another set of labels.
- **diagnostics** Contains functions that create diagnostic plots.
- **save** Once a clustering algorithm has been applied to a data set the
  function _save_topic_model_ is a convenient way to store the topic model. It
  mostly calls the functions in **interpret** and **diagnostics**
