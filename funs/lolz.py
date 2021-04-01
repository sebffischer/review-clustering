from numpy.core.defchararray import add
from funs.interpret import plot_ngram, plot_wordcloud, plot_topics
from funs.utils import load_pickle
import pandas as pd 
from funs.interpret import plot_topics
from funs.preprocess import get_term_frequency
import numpy as np 
from numpy.linalg import norm 

model = load_pickle("./results/clustering/amazon_skmeans_30/model.pkl")

unigram_df = pd.read_csv("./data/processed/amazon_unigram.csv")
bigram_df = pd.read_csv("./data/processed/amazon_bigram.csv")

wordcloud_plot = plot_wordcloud(unigram_df, 
                                labels=model.labels_, 
                                specific_label=29, make_title=False, 
                                figsize=(4, 3), 
                                types=["NOUN", "ADJ", "ADV", "PROPN", "VERB"])

wordcloud_plot.savefig("./results/clustering/report/graphics/amazon_skmeans_30_wordcloud_29.png", 
                       dpi=400, 
                       bbox_inches="tight")

ngram_plot = plot_ngram(bigram_df, 
                        labels=model.labels_, 
                        specific_label=29, 
                        figsize=(4, 3), 
                        additional_stopwords="amazon", 
                        make_title=False)


ax = ngram_plot.gca()
ax.set_xlabel("Frequency")
ax.set_xticks(ticks = [5, 10, 15])
ax.set_xticklabels(["5", "10", "15"])


ngram_plot.savefig("./results/clustering/report/graphics/amazon_skmeans_30_ngram.png", 
                   dpi=400, 
                   bbox_inches="tight")

