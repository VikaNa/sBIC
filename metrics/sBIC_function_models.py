import time, math, pickle, os
import numpy as np
from decimal import *

# Import from sklearn.decomposition module the LDA algorithm
from sklearn.decomposition import LatentDirichletAllocation
# Import lda module: https://pypi.org/project/lda/
import lda

# Import logsumexp function
from scipy.special import logsumexp
# Import sparse matrix package
import scipy.sparse as sp


def learning_coefficient(n_docs, n_words, super_model, sub_model):
    """
    Compute learning coefficient and its multiplicity
    using formulas from Hayashi (2021):

    N=n_docs: number of documents
    M=n_words: vocabulary size
    H=super_model: number of topics in a model of a higher order
    r=sub_model: number of topics in a model of a lower order

    Output:
    learn_coef = \lambda_{Hr}
    m = multiplicity of \lambda_{Hr}

    """
    N = n_docs
    M = n_words
    H = super_model
    r = sub_model

    if (N + r + 1) <= (M + H) and (M + r + 1) <= (N + H) and (H + r + 1) <= (M + H):
        if (M + N + H + r) % 2 != 0:
            learn_coeff = (2 * (H + r + 1) * (M + N) - (M - N) ** 2 - (H + r + 1) ** 2) / 8 - N / 2;
            m = 1
        else:
            learn_coeff = (2 * (H + r + 1) * (M + N) - (M - N) ** 2 - (H + r + 1) ** 2 + 1) / 8 - N / 2;
            m = 2
    elif (M + H) < (N + r + 1):
        learn_coeff = (M * H + N * (r + 1) - H * (r + 1) - N) / 2;
        m = 1
    elif (N + H) < (M + r + 1):
        learn_coeff = (N * H + M * (r + 1) - H * (r + 1) - N) / 2;
        m = 1
    else:
        learn_coeff = (M * N - N) / 2;
        m = 1

    return learn_coeff, m


# Function to compute loglikelihood using count data and parameter estimates
def loglikelihood(X, theta, beta):
    is_sparse_x = sp.issparse(X)
    n_samples, n_features = X.shape
    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    score = 0
    for idx_d in range(n_samples):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]: X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]: X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]
        temp = np.dot(theta[idx_d, :, np.newaxis].T, beta[:, ids])
        norm_phi = np.log(temp)
        score += np.dot(cnts, norm_phi.T)

    return score.item(0)


def calculate_sBIC_models(count_data, models, min_n_topics=10, max_n_topics=100, sampling='var_inf',
                          likelihood='variational_loglik', restarts_number=1, n_iter=100,
                          steps=1, precision_value=50000):
    '''
    Calculates sBIC for estimated models.
    :param: models: list containing estimated models in the range.
    :param sampling: sampling method. "var_inf" and "gibbs" are possible values.
    :param likelihood: if sampling="var_inf", there are two different methods to calculate likelihoods: "loglik" (old) and "variational_loglik" (new).
    :param restarts: number of restarts.
    :param n_iter: number of iterations.
    :param steps: an integer number specifying the incrementation in the given topics range.
    :param path: define the directory the models should be stored in.
    :return: sBIC values.
    '''

    # Set computational precision
    getcontext().prec = 28
    getcontext().Emin = -999999999
    getcontext().Emax = 999999999

    # Number of documents
    n_docs = count_data.shape[0]

    # Number of words in vocabulary
    n_words = count_data.shape[1]

    # Number of observations = sum of counts as in Hayashi (2021):
    # "the sample size n is the number of words in all of the given documents"
    n_obs = np.sum(count_data)

    # Initialize list of log-likelihoods
    LogLik = []

    # Initialize list of BICs
    BIC = []

    # Initialize list of log-likelihoods for restarts
    RESTARTSLIK = np.zeros(shape=(restarts_number, max_n_topics - min_n_topics + 1))
    topic_range = [i for i in range(min_n_topics, max_n_topics + 1, steps)]

    if sampling == 'var_inf':
        if likelihood == 'loglik':
            print('sampling var inf old')
            t0 = time.time()
            # Loop over number of topics from min to max to compute log-likelihoods
            for i, model_fitted in enumerate(models):
                t001 = time.time()
                # print('number of topics=', k)
                # Loop over restarts
                k=model_fitted.n_components
                for restarts_index in range(0, restarts_number):
                    print('restarts_index=', restarts_index)
                    RESTARTSLIK[restarts_index, i] = model_fitted.loglik(count_data)
                t002 = time.time()
                # print('Time to estimate model with ' + str(k) + ' topics = ', t002 - t001)
                # Use median log-likelihood
                LogLik.append(np.median(RESTARTSLIK[:, i]))
                BIC.append(LogLik[i] - ((n_docs + n_words - 1) * k - n_docs) / 2 * math.log(n_obs))

            t01 = time.time()
            print('Time to estimate all models = ', t01 - t0)
            # Compute LogL_ij
            LogLij = []
            for i, topic in enumerate(topic_range):
                LogLi = []
                for j in topic_range[:i + 1]:
                    lc = learning_coefficient(n_docs, n_words, topic, j)
                    LogLi.append((LogLik[i] - lc[0] * math.log(n_obs) + (lc[1] - 1) * math.log(math.log(n_obs))))
                LogLij.append(LogLi)

            mn = max(max(LogLij))
        elif likelihood == 'variational_loglik':
            print('sampling var inf new')
            t0 = time.time()
            for i, model_fitted in enumerate(models):
                t001 = time.time()
                # print('number of topics=', k)
                # Loop over restarts
                k = model_fitted.n_components
                for restarts_index in range(0, restarts_number):
                    print('restarts_index=', restarts_index)
                    RESTARTSLIK[restarts_index, i] = model_fitted.variational_loglik(count_data)
                t002 = time.time()
                # print('Time to estimate model with ' + str(k) + ' topics = ', t002 - t001)
                # Use median log-likelihood
                LogLik.append(np.median(RESTARTSLIK[:, i]))
                BIC.append(LogLik[i] - ((n_docs + n_words - 1) * k - n_docs) / 2 * math.log(n_obs))
            t01 = time.time()
            print('Time to estimate all models = ', t01 - t0)
            # Compute LogL_ij
            LogLij = []
            for i, topic in enumerate(topic_range):
                LogLi = []
                for j in topic_range[:i + 1]:
                    lc = learning_coefficient(n_docs, n_words, topic, j)
                    LogLi.append((LogLik[i] - lc[0] * math.log(n_obs) + (lc[1] - 1) * math.log(math.log(n_obs))))
                LogLij.append(LogLi)

            mn = min(min(LogLij))
    elif sampling == 'gibbs':
        print('sampling gibbs')
        t0 = time.time()
        # Loop over number of topics from min to max to compute log-likelihoods
        for i, model_fitted in enumerate(models):
            t001 = time.time()
            # print('number of topics=', k)
            # Loop over restarts
            k = model_fitted.components_.shape[0]
            for restarts_index in range(0, restarts_number):
                print('restarts_index=', restarts_index)
                theta = model_fitted.doc_topic_.copy()
                beta = model_fitted.components_.copy()
                RESTARTSLIK[restarts_index, i] = loglikelihood(count_data, theta, beta)
            t002 = time.time()
            # print('Time to estimate model with ' + str(k) + ' topics = ', t002 - t001)
            LogLik.append(np.median(RESTARTSLIK[:, i]))
            BIC.append(LogLik[i] - ((n_docs + n_words - 1) * k - n_docs) / 2 * math.log(n_obs))

        t01 = time.time()
        print('Time to load all models = ', t01 - t0)

        # Compute LogL_ij
        LogLij = []
        for i, topic in enumerate(topic_range):
            LogLi = []
            for j in topic_range[:i + 1]:
                lc = learning_coefficient(n_docs, n_words, topic, j)
                LogLi.append((LogLik[i] - lc[0] * math.log(n_obs) + (lc[1] - 1) * math.log(math.log(n_obs))))
            LogLij.append(LogLi)

        mn = max(max(LogLij))

    # Normalization and exponentiation
    print('Normalization and exponentiation')
    Lij = []
    for i in range(len(range(0, max_n_topics - min_n_topics + 1, steps))):
        Li = []
        for j in range(0, len(LogLij[i])):
            Li.append(Decimal(LogLij[i][j] - mn).exp())
        Lij.append(Li)

    # sBIC for minimal model
    L = [Lij[0][0]];
    sBIC = [Lij[0][0].ln() + Decimal(mn)]

    # Loop over topics to compute sBIC.
    print('Loop over topics to compute sBIC.')
    # Automatically increase precision if Infinity values are encountered.
    for i in range(1, len(range(0, max_n_topics - min_n_topics + 1, steps))):
        b = -Lij[i][i] + sum(L[0:i])
        c = Decimal(0)
        for j in range(0, i):
            c -= Lij[i][j] * L[j]
        getcontext().prec = precision_value
        Temp = (-b + (b ** Decimal(2) - Decimal(4) * c).sqrt()) / Decimal(2)
        getcontext().prec = 28
        while (max(Temp, Decimal(0))).ln() + Decimal(mn) == Decimal('-Infinity'):
            precision_value += 50000
            getcontext().prec = precision_value  # increase this value if infinities are encountered
            Temp = (-b + (b ** Decimal(2) - Decimal(4) * c).sqrt()) / Decimal(2)
            getcontext().prec = 28
        L.append(Temp)
        sBIC.append((max(L[i], Decimal(0))).ln() + Decimal(mn))

    # Select optimal number of topics
    optimal_n = topic_range[np.argmax(sBIC)]
    print(f"The optimal number of topics: {optimal_n}")
    return sBIC
