# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""

import os
from collections import OrderedDict

import xlwt
from six.moves import cPickle as pickle


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


