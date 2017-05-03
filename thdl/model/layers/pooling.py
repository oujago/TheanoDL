# -*- coding: utf-8 -*-

from theano.tensor.signal import pool

from .base import Layer


class Pooling(Layer):
    def __init__(self, pool_size, pad=(0, 0), ignore_border=True, mode='max', stride=None):
        super(Pooling, self).__init__()

        self.pool_size = pool_size  # (vertical ds, horizontal ds)
        self.pad = pad
        self.ignore_border = ignore_border
        self.mode = mode
        self.stride = stride

    def forward(self, input, **kwargs):
        output = pool.pool_2d(input=input, ws=self.pool_size, pad=self.pad,
                              ignore_border=self.ignore_border, mode=self.mode,
                              stride=self.stride)
        return output

    def to_json(self):
        config = {
            'pool_size': self.pool_size,
            'pad': self.pad,
            'ignore_border': self.ignore_border,
            'mode': self.mode,
            'stride': self.stride,
        }
        return config


class MaxPooling(Pooling):
    def __init__(self, pool_size, **kwargs):
        super(MaxPooling, self).__init__(pool_size, **kwargs)


class MeanPooling(Pooling):
    def __init__(self, pool_size, mode='inc', **kwargs):
        if mode == 'inc':
            super(MeanPooling, self).__init__(pool_size, mode='average_inc_pad', **kwargs)

        elif mode == 'exc':
            super(MeanPooling, self).__init__(pool_size, mode='average_exc_pad', **kwargs)

        else:
            raise ValueError
