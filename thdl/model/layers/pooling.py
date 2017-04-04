# -*- coding: utf-8 -*-

from theano.tensor.signal import pool

from .base import Layer


class Pool2D(Layer):
    def __init__(self, pool_size, pad=(0, 0), ignore_border=True, mode='max'):
        super(Pool2D, self).__init__()

        self.pool_size = pool_size  # (vertical ds, horizontal ds)
        self.pad = pad
        self.ignore_border = ignore_border
        self.mode = mode

    def __call__(self, input):
        # output
        output = pool.pool_2d(input=input, ws=self.pool_size, pad=self.pad,
                              ignore_border=self.ignore_border, mode=self.mode)

        return output

    def __str__(self):
        return 'Pool2D'

    def to_json(self):
        config = {
            'pool_size': self.pool_size,
            'pad': self.pad,
            'ignore_border': self.ignore_border,
            'mode': self.mode
        }
        return config
