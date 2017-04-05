# -*- coding: utf-8 -*-


from theano import tensor

from .base import Layer


class Flatten(Layer):
    def __init__(self, ndim=2):
        self.ndim = ndim

    def connect_to(self, pre_layer=None):
        self.output_shape = pre_layer.output_shape

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

    def connect_to(self, pre_layer=None):
        self.output_shape = self.newshape

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

    def connect_to(self, pre_layer=None):
        input_shape = pre_layer.output_shape

        self.output_shape = input_shape[:self.axis] + input_shape[self.axis:]

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

    def connect_to(self, pre_layer=None):
        input_shape = pre_layer.output_shape
        output_shape = []

        for p in self.pattern:
            if p == 'x':
                output_shape.append(1)
            else:
                output_shape.append(input_shape[p])

        self.output_shape = output_shape

    def forward(self, input, **kwargs):
        return input.dimshuffle(self.pattern)

    def to_json(self):
        config = {
            'pattern': self.pattern
        }
        return config
