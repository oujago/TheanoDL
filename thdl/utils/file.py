# -*- coding: utf-8 -*-

import os
from collections import OrderedDict

import xlwt
from six.moves import cPickle as pickle


def write_xls(contents, filepath):
    """Write the contents into the xls tables.
    
    :param contents: an instance of :class:`dict` or an instance of :class:`list`
        If `isinstance(contents, dict) == True`, there is only one line in xls table.
        If contents is a list, then there are several lines in xls table.
    :param filepath: :class:`str` instance
        The path to save.
    """
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


def pickle_dump(data, path):
    """
    Given the DATA, then pickle it into the PATH.
    :param data: the f_data to pickle
    :param path: the path to store the pickled f_data
    """
    pickle.dump(data, open(os.path.join(os.getcwd(), path), 'wb'))


def pickle_load(path):
    """From the PATH get the DATA.
    
    :param path: the path that store the pickled f_data
    """
    data = pickle.load(open(os.path.join(os.getcwd(), path), 'rb'))
    return data


def check_duplicate_path(filepath):
    """Check the duplicate path.
    
    If there is a same filepath, for example 'file/test.txt',
    Then this path will be 'file/test(1).txt'

    If 'file/test(1).txt' also exists, then this path will be 'file/test(2).txt'

    :param filepath: the file path to save.
    :return: the file path can be used to save path.
    """
    if not os.path.exists(os.path.join(os.getcwd(), filepath)):
        return filepath

    i = 1
    splits = filepath.split(".")
    head, tail = '.'.join(splits[:-1]), splits[-1]
    while os.path.exists(os.path.join(os.getcwd(), "%s(%d).%s" % (head, i, tail))):
        i += 1

    return "%s(%d).%s" % (head, i, tail)
