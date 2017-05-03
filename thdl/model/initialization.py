# -*- coding: utf-8 -*-


import numpy as np
import theano

from thdl.base import ThdlObj
from thdl.utils.random import get_dtype
from thdl.utils.random import get_rng


def shared(value, borrow=True):
    return theano.shared(value=value, borrow=borrow)


class Initializer(ThdlObj):
    def __call__(self, size, get_shared=True):
        value = self.call(size)
        return shared(value=value) if get_shared else value

    def call(self, size):
        raise NotImplementedError()

    def to_json(self):
        return {}


class Zero(Initializer):
    def call(self, size):
        value=np.zeros(size, dtype=get_dtype())
        return value


class One(Initializer):
    def call(self, size):
        value = np.ones(size, dtype=get_dtype())
        return value


class Uniform(Initializer):
    def __init__(self, scale=0.05):
        self.scale = scale

    def call(self, size):
        value = get_rng().uniform(-self.scale, self.scale, size=size)
        value = value.astype(get_dtype())
        return value

    def to_json(self):
        return {'scale': self.scale}


class Normal(Initializer):
    def __init__(self, scale=0.05):
        self.scale = scale

    def call(self, size):
        value = get_rng().normal(loc=0.0, scale=self.scale, size=size)
        value = value.astype(get_dtype())
        return value

    def to_json(self):
        return {'scale': self.scale}


class LecunUniform(Initializer):
    """
    Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """

    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(3. / fan_in))(size)


class GlorotUniform(Initializer):
    """
    Reference: Glorot & Bengio, AISTATS 2010
    """

    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(6 / (fan_in + fan_out)))(size)


class GlorotNormal(Initializer):
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Normal(np.sqrt(2 / (fan_out + fan_in)))(size)


class HeNormal(Initializer):
    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Normal(np.sqrt(2. / fan_in))(size)


class HeUniform(Initializer):
    """
    Reference:  He et al., http://arxiv.org/abs/1502.01852
    """

    def call(self, size):
        fan_in, fan_out = _decompose_size(size)
        return Uniform(np.sqrt(6. / fan_in))(size)


class Orthogonal(Initializer):
    """
    From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """

    def call(self, size):
        flat_shape = (size[0], np.prod(size[1:]))
        a = get_rng().normal(loc=0., scale=1., size=flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        value = q.reshape(size).astype(get_dtype())
        return value


def _decompose_size(size):
    if len(size) == 2:
        fan_in = size[0]
        fan_out = size[1]

    elif len(size) == 4 or len(size) == 5:
        respective_field_size = np.prod(size[2:])
        fan_in = size[1] * respective_field_size
        fan_out = size[0] * respective_field_size

    else:
        fan_in = fan_out = int(np.sqrt(np.prod(size)))

    return fan_in, fan_out


_zero = Zero()
_one = One()
