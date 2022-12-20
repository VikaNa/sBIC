
def generate_text_from_lda(text_length, doc_topic_weights, topic_word_weights, vocabulary, rng):
    '''
    :param corpus_length: a list containing the length of texts in the corpus.
    :param doc_topic_weights: a data frame, where the rows correspond to the documents and the columns - to the topics
    :param topic_word_weights: a data frame, where the rows correspond to the topics and the columns - to the vocabulary words
    :param vocabulary: list of vocabulary words
    :param rng: a random number generator object to use for this function.
    :return: generated corpus of texts
    '''

    import numpy as np
    from tqdm import tqdm
    #np.random.seed(random_seed)
    corpus_gen = []

    word_weights_cum = np.cumsum(topic_word_weights, axis=1)
    topic_weights_cum = np.cumsum(doc_topic_weights, axis=1)

    for j, text in enumerate(text_length):
        # define the length of a new abstract
        abstract_length = text
        # cumulative sum of topic probabilities of the abstract
        topic_prob_cum = topic_weights_cum.iloc[j]
        # generate uniformly distributed floats for topic and words selection
        deltas = rng.uniform(0, 1, abstract_length)
        delta2s = rng.uniform(0, 1, abstract_length)

        abstract_gen = []

        for k in range(0, abstract_length):
            # select randomly a topic
            ii, = np.where(topic_prob_cum >= deltas[k])
            topic = min(ii)
            # select randomly a word
            ii, = np.where(word_weights_cum.iloc[topic] >= delta2s[k])
            new_word_index = min(ii)
            abstract_gen.append(vocabulary[new_word_index])
        corpus_gen.append(' '.join(abstract_gen))
    return corpus_gen

