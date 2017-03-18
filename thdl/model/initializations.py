# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""

import numpy as np
from theano import shared

from .variables import dtype


def get_shared(rng, size, init, dim_ordering='th', borrow=True, name=None, **kwargs):
    """
    Initialization getting method

    :param rng:
    :param size:
    :param init:
    :param dim_ordering:
    :param borrow:
    :param name:
    :param kwargs:
    """
    if init == 'zero':
        return shared(value=np.asarray(np.zeros(size), dtype=dtype), name=name, borrow=borrow)

    if init == 'one':
        return shared(value=np.asarray(np.ones(size), dtype=dtype), name=name, borrow=borrow)

    ##########################################
    #               Get Fans                 #
    # adapted from keras/initializations.py  #
    ##########################################
    if len(size) == 2:
        fan_in = size[0]
        fan_out = size[1]

    elif len(size) == 4 or len(size) == 5:
        # assuming convolution kernels (2D or 3D)
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering.lower() == 'th':
            respective_field_size = np.prod(size[2:])
            fan_in = size[1] * respective_field_size
            fan_out = size[0] * respective_field_size
        elif dim_ordering.lower() == 'tf':
            respective_field_size = np.prod(size[:-2])
            fan_in = size[-2] * respective_field_size
            fan_out = size[-1] * respective_field_size
        else:
            raise ValueError("Invalid dim_ordering: ", dim_ordering)

    else:
        fan_in = np.sqrt(np.prod(size))
        fan_out = np.sqrt(np.prod(size))

    ##########################################
    #               Get Values               #
    # adapted from keras/initializations.py  #
    ##########################################
    if init == 'uniform':
        scale = kwargs.get('scale', 0.05)
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'normal':
        scale = kwargs.get('scale', 0.05)
        value = rng.normal(loc=0.0, scale=scale, size=size)

    elif init == 'lecun_uniform':
        ''' Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf '''
        scale = np.sqrt(3. / fan_in)
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'glorot_normal':
        ''' Reference: Glorot & Bengio, AISTATS 2010 '''
        scale = np.sqrt(2. / (fan_in + fan_out))
        value = rng.normal(loc=0.0, scale=scale, size=size)

    elif init == 'glorot_uniform':
        scale = np.sqrt(6. / (fan_in + fan_out))
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'he_normal':
        ''' Reference:  He et al., http://arxiv.org/abs/1502.01852 '''
        scale = np.sqrt(2. / fan_in)
        value = rng.normal(loc=0.0, scale=scale, size=size)

    elif init == 'he_uniform':
        scale = np.sqrt(6. / fan_in)
        value = rng.uniform(low=-scale, high=scale, size=size)

    elif init == 'orthogonal':
        ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120 '''
        flat_shape = (size[0], np.prod(size[1:]))
        a = rng.normal(loc=0.0, scale=1.0, size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        value = q.reshape(size)

    elif init == 'rand':
        value = rng.rand(*size) * 0.2 + 0.1

    else:
        raise ValueError("unknown init type: %s" % init)

    value = np.asarray(value, dtype=dtype)
    return shared(value=value, name=name, borrow=borrow)
