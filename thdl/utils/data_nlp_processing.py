# -*- coding: utf-8 -*-

import re
import numpy as np
from .random import get_dtype


def one_hot(labels, nb_classes=None, dtype=get_dtype()):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if nb_classes is None:
        nb_classes = classes.size
    one_hot_labels = np.zeros((labels.shape[0], nb_classes), dtype=dtype)
    for i, c in enumerate(classes):
        one_hot_labels[labels == c, i] = 1
    return one_hot_labels


def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string


def remove_punctuation(string):
    string = re.sub(r"[^A-Za-z0-9'-]", " ", string)
    return string


def get_split(length, folder_num, test_idx):
    """
    Get the splits, according to the total length and the folder_num.
    :param length:
    :param folder_num:
    :param test_idx:
    :return:
    """
    # get the quotient, and the remainder
    gap, mod = divmod(length, folder_num)

    # get the start of folder 'folder_idx'
    start = gap * test_idx
    start += min(test_idx, mod)

    # get the end  of folder 'folder_idx'
    if test_idx < mod:
        end = start + gap + 1
    else:
        end = start + gap

    # return
    return start, end


def yield_item(iterable):
    """
    yield the text list. Maybe the list is a multiple dimension list.

    For examples:
        ['a', 'b', 'c', 'd'] ---> ['a', 'b', 'c', 'd']

        and

        [['a', 'b'], 'c', 'd'] ---> ['a', 'b', 'c', 'd']

        and

        [[['a', 'b'], 'c'], 'd'] ---> ['a', 'b', 'c', 'd']

    :param iterable:
    """
    for item in iterable:
        if type(item) in [list, tuple]:
            for elem in yield_item(item):
                yield elem
        else:
            yield item


def _item_list2index_list(item_iter, index_iter, item2index, unknown='UNKNOWN'):
    """
    Transform the item list to the index list, by using item2index.
    :param item_iter: the item_list
    :param index_iter: the index list
    :param item2index:
    :param unknown: if item not in the item2index, using unknown instead.
    :return:
    """
    for item in item_iter:
        if type(item).__name__ in ('list', 'tuple'):
            item_list = []
            _item_list2index_list(item, item_list, item2index, unknown)
            index_iter.append(item_list)
        else:
            if item in item2index:
                index_iter.append(item2index[item])
            else:
                index_iter.append(item2index[unknown])


def item_list2index_list(item_iter, item2index, unknown='UNKNOWN'):
    """
    Transform the item list to the index list, by using item2index.
    :param item_iter: the item_list
    :param item2index:
    :param unknown: if item not in the item2index, using unknown instead.
    :return:
    """
    index_iter = []
    _item_list2index_list(item_iter, index_iter, item2index, unknown)
    return index_iter


def sen2seqs(sen, seq_len):
    """
    Make the sentence indices into sequence indices.
    For examples:

        >>> sen2seqs([0, 1], 4)
        >>> [[0], [0, 1]]

        and

        >>> sen2seqs([0, 1, 2, 3, 4, 5], 4)
        >>> [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]

    :param sen:
    :param seq_len:
    """
    seqs = [sen[max((0, i - seq_len)): i] for i in range(1, len(sen) + 1)]
    return seqs


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    Copied from Keras library.

    Pads each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or the end of the sequence.

    :param sequences: list of lists where each element is a sequence
    :param maxlen: int, maximum length
    :param dtype: type to cast the resulting sequence.
    :param padding: 'pre' or 'post', pad either before or after each sequence.
    :param truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
    :param value: float, value to pad the sequences to the desired value.
    :return: numpy array with dimensions (number_of_sequences, maxlen)
    """

    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.asarray(np.ones((nb_samples, maxlen) + sample_shape) * value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
