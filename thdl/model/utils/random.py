# -*- coding: utf-8 -*-

import numpy as np
import theano

_rng = np.random
_dtype = theano.config.floatX


def set_seed(seed):
    global _rng
    _rng = np.random.RandomState(seed=seed)


def set_rng(rng):
    global _rng
    _rng = rng


def get_rng():
    return _rng


def set_dtype(dtype):
    global _dtype
    _dtype = dtype


def get_dtype():
    return _dtype
