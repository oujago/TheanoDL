# -*- coding: utf-8 -*-


import math
import os

from datetime import datetime


def now():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def today():
    return datetime.now().strftime('%Y-%m-%d')


def check_duplicate_path(filepath):
    """
    If there is a same filepath, for example 'file/test.txt',
    Then this path will be 'file/test(1).txt'

    If 'file/test(1).txt' also exists, then this path will be 'file/test(2).txt'

    :param filepath:
    :return:
    """
    if not os.path.exists(os.path.join(os.getcwd(), filepath)):
        return filepath

    i = 1
    splits = filepath.split(".")
    head, tail = '.'.join(splits[:-1]), splits[-1]
    while os.path.exists(os.path.join(os.getcwd(), "%s(%d).%s" % (head, i, tail))):
        i += 1

    return "%s(%d).%s" % (head, i, tail)


def time_format(total_time):
    if total_time > 3600:
        return "%f h" % (total_time / 3600)
    elif total_time > 60:
        return "%f min" % (total_time / 60)
    else:
        return "%f s" % total_time


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    from thdl.data import processing
    print("Please use thdl.nlp_data.pad_sequences()!")
    return processing.pad_sequences(sequences, maxlen, dtype, padding, truncating, value)

