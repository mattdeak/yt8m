import pandas as pd
import tensorflow as tf

def _parse_labels(tf_example):
    """Extracts the labels from a yt8m tf.Example object.

    Parameters
    ----------
    tf_example : tf.Example
        An example object loaded from a tf.record.

    Returns
    -------
    str
        A string representation of the labels. Each label is separated by
        whitespace (E.g "1 23 45").
    """
    # Extract labels
    labels = tf_example.features.feature['labels'].int64_list.value

    # Convert to string
    labels = str(labels)

    # Remove unecessary punctuation
    labels = labels.replace('[', "").replace("]", "").replace(',', "")
    return labels


def extract_labels(video_files, outfile='labels.txt'):
    """Extracts labels from the yt8m video files and organizes them in a text file.

    Parameters
    ----------
    video_files : list of str
        List of video files by directory.
    outfile : str
        Output destination file (default: 'labels.txt').
    """
    with open(outfile, 'w') as output:
        for video_file in video_files:
            for record in tf.python_io.tf_record_iterator(video_file):
                record = tf.train.Example.FromString(record)
                id_ = record.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
                labels = _parse_labels(record)
                output.write(id_ + " " + labels)
                output.write('\n')


def load_labels(filename):
    """Loads labels from file as dictionary, with keys as video IDs and values a list of labels.

    Parameters
    ----------
    filename : str
        Full or relative filepath of the file to be loaded from

    Returns
    -------
    dict of {str: list of int}
        Mapping of video ID to list of integer labels.
    """
    out = {}
    with open(filename, 'r') as file:
        for line in file:
            data = line.split(' ')
            out[data[0]] = [int(datum) for datum in data[1:]]
    return out
