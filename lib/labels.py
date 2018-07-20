import os
import time
import string
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from .util import timed, imputer_dict


def file_reader(num):
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
    with open(f'/home/data/glove/x{num}', 'rb') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0].decode()
            embedding = np.array([float(val) for val in splitLine[1:]])
            out.append((word, embedding))
    return out


class LabelHandler:
    """Responsible for loading labels and converting them to any other format, e.g. glove vectors.

    New formats for labels should be added as methods to this class.

    Parameters
    ----------
    subset : {'train', 'validate', 'test'}
        Name of dataset to load labels for.
    default_text_label : str
        In the event that a label int is not associated with a str, default to this.
    default_glove_word : str
        In the event that the GLoVE embedding does not include a word, default to this.

    Attributes
    ----------
    punc_remover : dict
        Utility for removing punctuation from labels to pass them to GLoVE.
    default_text_label : str
        In the event that a label int is not associated with a str, default to this.
    default_glove_word : str
        In the event that the GLoVE embedding does not include a word, default to this.
    video_to_int : dict of {str: list of int}
        Maps video ID to its integer labels.
    int_to_name : dict of {int: str}
        Maps integer label to text label, e.g. "video game".
    word_to_glove : dict of {str: np.array}
        Maps singular word from label, e.g. "video", to its 300-D GLoVE vector.
    """
    def __init__(self, subset, limit=-1,
                 default_text_label='undefined',
                 default_glove_word='the'):
        self.punc_remover = str.maketrans('', '', string.punctuation)
        self.filename = f'/home/data/labels/{subset}_labels.txt'
        self.limit = limit
        self.default_text_label = default_text_label
        self.default_glove_word = default_glove_word
        self.video_to_int = None
        self.int_to_name = None
        self.word_to_glove = None

    @timed
    def _load_labels(self, filename):
        """Loads labels from file as dictionary.

        Parameters
        ----------
        filename : str
            Full or relative filepath of the file to be loaded from.

        Returns
        -------
        dict of {str: list of int}
            Mapping of video ID to list of integer labels.
        """
        out = {}
        with open(filename, 'r') as file:
            for i,line in enumerate(file):
                if self.limit >= 0 and i >= self.limit:
                    break
                data = line.split(' ')
                out[data[0]] = [int(datum) for datum in data[1:]]
        return out

    @timed
    def _load_names(self):
        """Loads dictionary mapping label integers to text names."""
        return dict(pd.read_csv('/home/data/label_names_2018.csv').values)

    @timed
    def _load_glove(self):
        """Loads dictionary mapping individual words to their GLoVE vectors."""
        start_time = time.time()
        p = multiprocessing.Pool(multiprocessing.cpu_count()-2)
        results = p.map(file_reader, range(len(os.listdir('/home/data/glove'))))
        glove = [pair for result in results for pair in result]
        if self.default_text_label:
            glove = imputer_dict(self.default_glove_word, glove)
        else:
            glove = dict(glove)
        return glove

    def get(self):
        if not self.video_to_int:
            self.video_to_int = self._load_labels(self.filename)
        return self.video_to_int

    def get_text(self):
        """Converts loaded integer labels to their text names."""
        if not self.video_to_int:
            self.video_to_int = self._load_labels(self.filename)
        if not self.int_to_name:
            self.int_to_name = self._load_names()

        text_labels = {}
        for video, labels in self.video_to_int.items():
            try:
                video_labels = [self.int_to_name[label] for label in labels]
                video_labels = [label if isinstance(label, str) else self.default_text_label
                                for label in video_labels]
                text_labels[video] = video_labels
            except KeyError:
                continue
        return text_labels

    def get_glove(self):
        """Converts loaded integer labels to their GLoVE vectors."""
        if not self.video_to_int:
            self.video_to_int = self._load_labels(self.filename)
        if not self.word_to_glove:
            self.word_to_glove = self._load_glove()
        text_labels = self.to_text()
        punc_remover = str.maketrans('', '', string.punctuation)
        text_to_glove = {}
        for key, text_group in text_labels.items():
            for text in text_group:
                if text in text_to_glove:
                    continue
                text = text.replace('-', ' ') # e.g. "action-adventure game" is three words
                words = text.split(' ')
                words = [word.lower().translate(punc_remover) for word in words]
                if len(words) > 1:
                    text_to_glove[text] = np.array([self.word_to_glove[word]
                                                      for word in words]
                                                     ).mean(axis=0)
                else:
                    text_to_glove[text] = self.word_to_glove[words[0]]
        return text_to_glove

    def get_graph(self, identity=False):
        """Get adjacency matrix for graph of labels, where edges are common observations.

        Parameters
        ----------
        identity : bool
            Whether self-loops are included in the matrix; i.e. the addition of the identity.

        Returns
        -------
        adj_matrix : np.array
            A square 2D Numpy array that represents the adjacency matrix of the present labels.
        """
        if not self.video_to_int:
            self.video_to_int = self._load_labels(self.filename)

        uniques = list(set(val for obs in self.video_to_int.values() for val in obs))
        adj_matrix = np.zeros([len(uniques)]*2)
        for obs in self.video_to_int.values():
            for label1, label2 in itertools.product(obs, obs):
                if label1 != label2:
                    coords = (uniques.index(label1), uniques.index(label2))
                    adj_matrix[coords] = 1
        return adj_matrix
