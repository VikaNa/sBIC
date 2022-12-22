import numpy as np
import pandas as pd

from tmtoolkit.topicmod.evaluate import metric_cao_juan_2009, metric_arun_2010, metric_coherence_mimno_2011


def calculate_tm_metrics(models, count_data):
    cao_juan_2009 = []
    coherence_mimno_2011 = []
    for model_fitted in models:
        cao_juan_2009.append(metric_cao_juan_2009(model_fitted.topic_word_))
        coherence_mimno_2011.append(metric_coherence_mimno_2011(model_fitted.topic_word_, count_data, return_mean=True))

    return cao_juan_2009, coherence_mimno_2011