import pandas as pd
import tensorflow as tf

def _parse_labels(tf_example):
    """Extracts the labels from a yt8m tf.Example object.

    Arguments:
        tf_example {tf.Example} -- An example object loaded from a tf.record

    Returns:
        [str] -- A string representation of the labels. Each label is separated by
        whitespace (E.g "1 23 45")
    """
    # Extract labels
    labels = tf_example.features.feature['labels'].int64_list.value

    # Convert to string
    labels = str(labels)

    # Remove unecessary punctuation
    labels = labels.replace('[', "").replace("]", "").replace(',', "")
    return labels


def extract_labels(video_files, outfile='labels.txt'):
    """Extracts labels from the yt8m video files and organizes them
    in a text file.

    Arguments:
        video_files {list} -- List of video files by directory

    Keyword Arguments:
        outfile {str} -- Output destination file (default: {'labels.txt'})
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

    Arguments:
        filename {str} -- Full or relative filepath of the file to be loaded from

    Returns:
        [dict{str: list{int}}] -- mapping of video ID to list of integer labels
    """
    out = {}
    with open(filename, 'r') as file:
        for line in file:
            data = line.split(' ')
            out[data[0]] = [int(datum) for datum in data[1:]]
    return out


def labels_to_text(labels):
    """Takes output of load_labels and converts each label index to its text.

    Arguments:
        labels {list} -- list of integer labels to be converted

    Returns:
        [dict{str: list{str}}] -- mapping of video ID to list of string labels

    Notes:
        - some labels appear in the data but not in the label list; skipped
        - some labels in the label list have no text equivalent; relabelled 'undefined'
    """
    names = pd.read_csv('data/label_names_2018.csv')
    label_to_name = dict(names.values)


    text_labels = {}
    for video, labels in labels.items():
        try:
            video_labels = [label_to_name[label] for label in labels]
            video_labels = ['undefined' if isinstance(label, float) else label
                           for label in video_labels]
            text_labels[video] = video_labels
        except KeyError:
            continue
    return text_labels
