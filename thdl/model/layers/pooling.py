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

    def connect_to(self, pre_layer=None):
        assert pre_layer is not None
        assert 4 >= len(pre_layer.out_shape) >= 3

        old_h = pre_layer.out_shape[-2] + self.pad[0]
        old_w = pre_layer.out_shape[-1] + self.pad[1]
        pool_h, pool_w = self.pool_size

        if self.stride is None:
            new_h = old_h // pool_h
            new_w = old_w // pool_w
            assert old_h % pool_h == old_w % pool_w == 0
        else:
            new_h = (old_h - pool_h + 1) // self.stride[0] + 1
            new_w = (old_w - pool_w + 1) // self.stride[1] + 1

        self.out_shape = pre_layer.out_shape[:-2] + (new_h, new_w)

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
