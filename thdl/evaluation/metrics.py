# -*- coding: utf-8 -*-

"""
Adopted from Keras Version 1.2.1.
"""

from theano import tensor


_epsilon = 10e-8


def binary_accuracy(outputs, targets):
    return tensor.mean(tensor.eq(outputs, tensor.round(targets)))


def categorical_accuracy(outputs, targets):
    return tensor.mean(tensor.eq(tensor.argmax(outputs, axis=-1), tensor.argmax(targets, axis=-1)))


def sparse_categorical_accuracy(outputs, targets):
    return tensor.mean(tensor.eq(tensor.max(outputs, axis=-1),
                                 tensor.cast(tensor.argmax(targets, axis=-1), tensor.config.floatX)))


def precision(outputs, targets):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = tensor.sum(tensor.round(tensor.clip(outputs * targets, 0, 1)))
    predicted_positives = tensor.sum(tensor.round(tensor.clip(targets, 0, 1)))
    res = true_positives / (predicted_positives + _epsilon)
    return res


def recall(outputs, targets):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = tensor.sum(tensor.round(tensor.clip(outputs * targets, 0, 1)))
    possible_positives = tensor.sum(tensor.round(tensor.clip(outputs, 0, 1)))
    recall = true_positives / (possible_positives + _epsilon)
    return recall


def fbeta_score(outputs, targets, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if tensor.sum(tensor.round(tensor.clip(outputs, 0, 1))) == 0:
        return 0

    p = precision(outputs, targets)
    r = recall(outputs, targets)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + _epsilon)
    return fbeta_score


def fmeasure(outputs, targets):
    """Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(outputs, targets, beta=1)

