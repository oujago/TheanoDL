# -*- coding: utf-8 -*-


from theano import tensor

from .base import Layer


class Flatten(Layer):
    def __init__(self, ndim=2):
        self.ndim = ndim

    def forward(self, input, **kwargs):
        return input.flatten(self.ndim)

    def to_json(self):
        config = {
            'ndim': self.ndim
        }
        return config


class Reshape(Layer):
    def __init__(self, newshape):
        self.newshape = newshape

    def forward(self, input, **kwargs):
        return tensor.reshape(input, newshape=self.newshape)

    def to_json(self):
        config = {
            "newshape": self.newshape
        }
        return config


class Mean(Layer):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, input, **kwargs):
        return tensor.mean(input, axis=self.axis)

    def to_json(self):
        config = {
            "axis": self.axis
        }
        return config


class Dimshuffle(Layer):
    def __init__(self, pattern):
        self.pattern = pattern

    def forward(self, input, **kwargs):
        return input.dimshuffle(self.pattern)

    def to_json(self):
        config = {
            'pattern': self.pattern
        }
        return config
