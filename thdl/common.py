# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2016/10/26

@notes:

"""

import math
import os
from datetime import datetime
from collections import OrderedDict
import numpy as np
import xlwt

from six.moves import cPickle as pickle


def pickle_dump(data, path):
    """
    Given the DATA, then pickle it into the PATH.
    :param data: the f_data to pickle
    :param path: the path to store the pickled f_data
    """
    pickle.dump(data, open(os.path.join(os.getcwd(), path), 'wb'))


def pickle_load(path):
    """
    From the PATH get the DATA.
    :param path: the path that store the pickled f_data
    """
    data = pickle.load(open(os.path.join(os.getcwd(), path), 'rb'))
    return data


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
    from . import nlp_data
    print("Please use thdl.nlp_data.pad_sequences()!")
    return nlp_data.pad_sequences(sequences, maxlen, dtype, padding, truncating, value)


def ceil(number):
    return math.ceil(number)


def floor(number):
    return math.floor(number)


def yield_item(iterable):
    print("Please use thdl.nlp_data.yield_item()!")
    for item in iterable:
        if type(item).__name__ == 'list':
            for elem in yield_item(item):
                yield elem
        else:
            yield item



def write_xls(contents, filepath):
    if isinstance(contents, dict):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('sheet 1')

        if not isinstance(contents, OrderedDict):
            print("contents should be a instance of 'OrderedDict'.")

        for i, (head, content) in enumerate(contents.items()):
            ws.write(0, i, head)
            ws.write(1, i, content)

        wb.save(os.path.join(os.getcwd(), filepath))

    elif isinstance(contents, list) and isinstance(contents[0], dict):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('sheet 1')

        if not isinstance(contents[0], OrderedDict):
            print("contents should be a instance of 'OrderedDict'.")

        for i, (head, _) in enumerate(contents[0].items()):
            ws.write(0, i, head)

        i = 0
        for content in contents:
            i += 1
            for j, (_, value) in enumerate(content.items()):
                ws.write(i, j, value)

        wb.save(os.path.join(os.getcwd(), filepath))

    else:
        raise ValueError("Unknown format.")
