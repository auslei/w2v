#%%
import tensorflow as tf
import numpy as np
import itertools
from collections import Counter

#%%
sentences = [
    'I love machine learning',
    'Deep learning is a subset of machine learning',
    'Natural language processing is a part of artificial intelligence',
    'Word embeddings are useful for NLP tasks',
    'TensorFlow is a powerful library for machine learning'
]