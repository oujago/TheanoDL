# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""

import numpy as np
from theano import shared
from theano import tensor

from .variables import dtype


def get_regularization(regularizer, params, scale):
    if scale == 0.:
        return tensor.sum(shared(value=np.array(0., dtype=dtype)))

    regularizer = regularizer.lower()

    if regularizer == 'l1':
        return tensor.sum([tensor.sum(tensor.abs_(ps)) for ps in params]) * scale

    elif regularizer == 'l2':
        return tensor.sum([tensor.sum(tensor.sqr(ps)) for ps in params]) * scale

    else:
        raise ValueError("Unknown regularizer: %s" % regularizer)
