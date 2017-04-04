# -*- coding: utf-8 -*-

from theano import tensor
from theano.tensor.nnet import conv
from theano.tensor.signal import pool

from .base import Layer
from ..activation import get_activation
from ..initialization import get_shared

dot = tensor.dot


class Conv2D(Layer):
    def __init__(self, rng, image_shape, filter_shape,
                 activation='relu', init='glorot_uniform',
                 W=None, b=None):
        """
        :param rng:
        :param filter_shape: (number of filters, num input feature maps, filter height, filter width)
        :param image_shape: (batch size, num input feature maps, image height, image width)
        :param W:
        :param b:
        :param activation:
        :param init:
        """
        super(Conv2D, self).__init__()

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init

        # variables
        self.W = get_shared(rng, filter_shape, init) if W is None else W
        self.b = get_shared(rng, (filter_shape[0],), 'zero') if b is None else b

        # params
        self.train_params.extend([self.W, self.b])
        self.reg_params.extend([self.W])

    def __call__(self, input):
        # output
        conv_out = conv.conv2d(input=input, filters=self.W, image_shape=self.image_shape,
                               filter_shape=self.filter_shape)
        output = self.act_func(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        return output

    def __str__(self):
        return "Conv2D"

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_shape': self.filter_shape,
            'activation': self.act_name,
            'init': self.init,
        }
        return config


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
