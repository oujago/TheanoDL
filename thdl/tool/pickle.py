# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""

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


