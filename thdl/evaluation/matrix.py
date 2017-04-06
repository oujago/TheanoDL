# -*- coding: utf-8 -*-

import sys

import numpy as np

from .other import divide


def get_confusion_matrix(predictions, origins, y_num):
    """
    Get the value matrix, according to the predicted and original data.

    :param predictions:
    :param origins:
    :param y_num:
    """
    assert predictions.shape == origins.shape

    res_matrix = np.zeros((y_num, y_num), dtype='int32')
    for i in range(len(predictions)):
        res_matrix[origins[i], predictions[i]] += 1
    return res_matrix


def get_evaluation_matrix(confusion_matrix, beta=1):
    """
    Get the evaluation matrix, according to the value matrix

    :param confusion_matrix: A confusion matrix
    :param beta: beta value
    :return:
        evaluation matrix ——
                                    precision, recall, F1
        1st label :                 P         R        F1
        2nd label :                 P         R        F1
        ……                        P         R        F1
        the last but one line :     macro-P   macro-R  macro-F1
        the last line :             micro-P   micro-R  micro-F1
    """

    y_num = confusion_matrix.shape[0]
    res_matrix = np.zeros((y_num + 2, 3))

    # calculate each element precision, recall, F1
    for i in range(confusion_matrix.shape[0]):
        precision = divide(confusion_matrix[i, i], np.sum(confusion_matrix[:, i]))
        recall = divide(confusion_matrix[i, i], np.sum(confusion_matrix[i, :]))
        f1 = divide((1 + beta ** 2) * precision * recall, beta ** 2 * precision + recall)
        res_matrix[i, 0] = precision
        res_matrix[i, 1] = recall
        res_matrix[i, 2] = f1

    # calculate macro precision, recall, F1
    res_matrix[-2, :] = np.mean(res_matrix[:-2, :], axis=0)

    # calculate micro precision, recall, F1
    res_matrix[-1, :] = divide(np.sum(np.diag(confusion_matrix)), np.sum(confusion_matrix))

    return res_matrix


def print_matrix(matrix, rows, columns, file=sys.stdout):
    """
    Print the value matrix into the file.
    
    :param matrix:
    :param rows:
    :param columns:
    :param file:
    """
    assert len(rows) == len(matrix) and len(columns) == len(matrix[0])
    gap = max([len(row) for row in rows] + [len(column) for column in columns]) + 1

    # print header
    runout = ' ' * gap
    for column in columns:
        runout += (" " * (gap - len(column)) + column)
    print(runout, file=file)

    # print each row
    for i in range(len(rows)):
        runout = ' ' * (gap - len(rows[i])) + rows[i]
        for value in matrix[i]:
            value = ("%s" % value)[:gap - 1]
            runout += (" " * (gap - len(value)) + value)
        print(runout, file=file)


def format_runout_matrix(matrix, rows, columns):
    output_lines = []

    # check
    assert len(rows) == len(matrix) and len(columns) == len(matrix[0])
    gap = max([len(row) for row in rows] + [len(column) for column in columns]) + 1

    # header
    header = ' ' * gap
    for column in columns:
        header += (" " * (gap - len(column)) + column)
    output_lines.append(header)

    # each row
    for i in range(len(rows)):
        row_runout = ' ' * (gap - len(rows[i])) + rows[i]
        for value in matrix[i]:
            value = ("%s" % value)[:gap - 1]
            row_runout += (" " * (gap - len(value)) + value)
        output_lines.append(row_runout)

    return output_lines
