<h1 align="center">sBIC</h1>
<p align="center">singular Bayesian information criterion</p>

Selecting the right number of topics in LDA models is considered to be a difficult task, for which alternative approaches have been proposed. The performance of the recently developed singular Bayesian information criterion (sBIC) is evaluated and compared to the performance of alternative model selection criteria. The sBIC is a generalization of the standard BIC that can be implemented to singular statistical models. The comparison is based on Monte Carlo simulations and carried out for several alternative settings, varying with respect to the number of topics, the number of documents and the size of documents in the corpora. Performance is measured using different criteria which take into account the correct number of topics, but also whether the relevant topics from the DGPs are identified. Practical recommendations for LDA model selection in applications are derived.

<h2 align="center">Requirements</h2>

```python
git clone https://github.com/VikaNa/sBIC
cd sBIC
pip install -r requirements.txt
```

<h2 align="center">Data</h2>

In the current project, we consider three different data generation processes (DGPs):
- Scientific papers published in the [Journal of Economics and Statistics (JES)](https://www.degruyter.com/view/journals/jbnst/jbnst-overview.xml).
- Abstracts submitted to [European Research Consortium for Informatics and Mathematics (ERCIM)](https://www.ercim.eu/) and [Computational and Financial Econometrics (CFE)](http://www.cfenetwork.org/) conferences
- Newsticker items from [heise online](https://www.heise.de/)

The **data** folder contains document-topic and topic-word matrices used to generate new corpora for each DGP.

To load data use:

```python
import pickle
import numpy as np
# abstracts data
doc_topic = pickle.load(open('data/abstracts/BoA_data_topicweights', 'rb'))
topic_word = pickle.load(open('data/abstracts/BoA_data_word_weights', 'rb'))
text_length = np.load('ata/abstracts/text_length.npy')
wordlist = np.load("data/abstracts/BoA_data_wordlist.npy")
```

<h2 align="center">Text generation</h2>

```python
from text_generation.generate_text_from_lda import *
# Set Random State
from numpy.random import SeedSequence, default_rng
ss = SeedSequence(123)
corpus_gen = generate_text_from_lda(text_length, 
                                    doc_topic_weights = doc_topic, 
                                    topic_word_weights= topic_word,
                                    vocabulary = wordlist, 
                                    rng = default_rng(ss))
```

<h2 align="center">Model estimation</h2>

```python
count_data = pickle.load(open('data/abstracts/BoA_count_data', 'rb'))
min_k = 2
max_k = 32
topic_range = range(min_k, max_k+1, 1)
for k in topic_range:
    t001 = time.time()
    print('number of topics=', k)
    model_fitted = lda.LDA(n_topics=k, 
                            alpha=1 / k, 
                            eta=1 / k, 
                            n_iter=1000, 
                            random_state=1,
                            refresh=1000).fit(count_data)
    pickle.dump(model_fitted, open(f'{path}/abstract/lda_{k}Topics', 'wb'))
    t002 = time.time()
    print('Time to estimate model with ' + str(k) + ' topics = ', t002 - t001)
```

<h2 align="center">Evaluation metrics</h2>


<h2 align="center">Results</h2>
