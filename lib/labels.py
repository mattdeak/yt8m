import os
import time
import string
import itertools
import functools
import numpy as np
import pandas as pd
import multiprocessing
from .util import imputer_dict, timed

def file_reader(name, num):
    """Kernel for GLoVE parallel loading. Parses a single file for its contents.

    Parameters
    ----------
    num : int
        The number suffix for the file this process is meant to parse.

    Returns
    -------
    list of tuple of (str, np.array)
        The pairs of words and vectors found in the file.
    """
    if num < 10:
        num = f'0{num}'
    elif num >= 90:
        num = 9000 + num - 90
    out = []
    with open(f'/home/data/{name}/x{num}', 'rb') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0].decode()
            embedding = np.array([float(val) for val in splitLine[1:]])
            out.append((word, embedding))
    return out


@timed
def load_embedding(name='glove', default_text='the'):
    """Loads dictionary mapping individual words to their GLoVE vectors."""
    start_time = time.time()
    p = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
    results = p.map(functools.partial(file_reader, 'glove'),
                    range(len(os.listdir(f'/home/data/{name}'))),
                    )
    embedding = [pair for result in results for pair in result]
    embedding = imputer_dict(default_text, embedding)
    return embedding


def load_text_labels():
    return dict(pd.read_csv('/home/data/label_names_2018.csv').values)


def load_int_labels(subset, limit):
    filename = f'/home/data/labels/{subset}_labels.txt'
    out = {}
    with open(filename, 'r') as file:
        for i,line in enumerate(file):
            if limit >= 0 and i >= limit:
                break
            data = line.split(' ')
            out[data[0]] = [int(datum) for datum in data[1:]]
    return out


def text_to_embedding(text, embedding_dict):
    punc_remover = str.maketrans('', '', string.punctuation)
    words = text.replace('-', ' ')
    words = words.split(' ')
    words = [word.lower().translate(punc_remover) for word in words]
    if len(words) > 1:
        out = np.array([embedding_dict[word] for word in words]).mean(axis=0)
    else:
        out = embedding_dict[words[0]]
    return out


@timed
def embedding_matrix(subset, limit, embedding):
    int_to_text = load_text_labels()
    X = load_int_labels(subset, limit)
    X = sorted(list(set(val for obs in X.values() for val in obs)))
    X = [int_to_text[x] if x in int_to_text and isinstance(int_to_text[x], str) else 'the'
         for x in X]
    X = np.array(list(map(lambda x: text_to_embedding(x, embedding), X)))
    return X


@timed
def adjacency_matrix(subset, limit, identity=False):
    lists = load_int_labels(subset, limit)
    label_indices = sorted(list(set(val for obs in lists.values() for val in obs)))
    label_indices = dict(zip(label_indices, range(len(label_indices))))
    adj_matrix = np.zeros([len(label_indices)]*2)
    for obs in lists.values():
        for label1, label2 in itertools.product(obs, obs):
            if label1 != label2 or (identity and label1 == label2):
                coords = (label_indices[label1], label_indices[label2])
                adj_matrix[coords] += 1
    return adj_matrix
