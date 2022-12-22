from __future__ import print_function

import sys, os, glob, re, time, math, pickle
import numpy as np
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
from decimal import*
import scipy.sparse as sp
from scipy.stats import chi2
import lda
import glob

def OpTop(doc_word_proportions, theta, beta, cutoff, n_docs):
    # Model-based document-word frequencies
    X = np.matmul(theta, beta)
    XSorted = np.sort(X, axis=1)
    sorted_indices = X.argsort(axis=1)
    XSortedCumSum = XSorted.cumsum(axis=1)
    OpTopStat = 0
    P = 0
    # Loop over documents
    for j in range(0, n_docs):
        # Indices of unimportant words having low frequencies in LDA model
        cutoff_indices = np.where(XSortedCumSum[j, :] < cutoff)

        # Cumulative model-based frequency of unimportant words in document j
        Xmin = XSorted[j, cutoff_indices[0]].sum()

        # Indices of relatively important words having higher frequencies in LDA model
        include_indices = np.where(XSortedCumSum[j, :] >= cutoff)

        # Number of higher-frequency words in document j
        Pj = len(include_indices[0])

        # Vector of model-based frequencies of relatively important words in document j
        Xj = XSorted[j, include_indices[0]]

        # Cumulative observed frequency of unimportant words in document j
        Dmin = (doc_word_proportions[j, :].toarray())[0, sorted_indices[j, cutoff_indices[0]]].sum()
        # Note: unimportant words are identified using model-based frequencies

        # Vector of observed frequencies of relatively important words in document j
        Dj = (doc_word_proportions[j, :].toarray())[0, sorted_indices[j, include_indices[0]]]

        # Compute summand of OpTop statistic for document j
        Smin = (Dmin - Xmin) ** 2 / Xmin
        Sj = np.divide(np.power((Dj - Xj), 2), Xj).sum()

        # Add jth summand to OpTop statistic
        OpTopStat += (Pj + 1) * (Sj + Smin)

        # Add number of high-frequency words in jth document to degrees of freedom for OpTop statistic
        P += Pj

    # THIS IS WRONG, BUT THIS HOW LEWIS & GROSSETTI IMPLEMENT IT
    OpTopStat /= (P + n_docs)
    chsq = 1 - chi2.cdf(OpTopStat, df=1)

    # Return OpTop statistic and degrees of freedom
    return OpTopStat, P, chsq


def calculate_OpTop(count_data, models, min_k, max_k, cutoff=0.05):
    '''
    Return the values of the optop statistics applying the chosen cut-off.
    '''

    # Compute document-word proportions
    doc_word_proportions = count_data.multiply(sp.csr_matrix(1 / count_data.sum(axis=1)))
    doc_word_proportions.sorted_indices()

    # Number of documents
    n_docs = count_data.shape[0]

    # Number of words in vocabulary
    n_words = count_data.shape[1]

    # Minimal and maximal number of topics
    min_n_topics = min_k
    max_n_topics = max_k

    # Step size for number of topics
    step = 1

    # Initialize an array of OpTop stats
    OpTopStats = np.zeros(shape=(math.ceil((max_n_topics - min_n_topics + 1) / step), 3))

    # Frequency cutoff value
    # Reported cutoff in Lewis and Grossetti (2022) is 0.05
    # Default cutoff the implementation is 0.20

    # Significance level. Default is alpha=0.05
    # If alpha=0, then the optimal number of topics is selected
    # using the minimum of OpTop statistic

    # Number of restarts
    restarts_number = 1

    t0 = time.time()

    # Loop over number of topics from min to max to estimate lda models
    n_iter = 1000
    for i, model_fitted in enumerate(models):
        t001 = time.time()
        k = model_fitted.components_.shape[0]
        # Loop over restarts
        for restarts_index in range(0, restarts_number):
            # print('restarts_index=',restarts_index)
            theta = model_fitted.doc_topic_.copy()
            beta = model_fitted.components_.copy()
        t002 = time.time()
        # print('Time to estimate model with '+str(k)+' topics = ',t002-t001)

        OpTopStats[i, :] = OpTop(doc_word_proportions, theta, beta, cutoff, n_docs)

    t01 = time.time()
    print('Time to process all models = ',t01-t0)

    return OpTopStats[:, 0]


