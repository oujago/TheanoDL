# -*- coding:utf-8 -*-

from theano import tensor
from ..nonlinearity import ReLU
from ..initialization import GlorotUniform
from .base import Layer
from .convolution import Convolution
from .pooling import Pooling


class NLPConvPooling(Layer):
    def __init__(self, input_shape, nb_filters, filter_heights,
                 conv_strides=(1, 1), conv_border_mode='valid',
                 pool_pad=(0, 0), pool_ignore_border=True, pool_mode='max', pool_strides=None,
                 activation=ReLU(), init=GlorotUniform(),
                 W_regularizer=None, b_regularizer=None, bias=True):
        super(NLPConvPooling, self).__init__()

        self.nb_filters = nb_filters
        self.filter_heights = filter_heights
        self.input_shape = input_shape
        self.conv_strides = conv_strides
        self.conv_border_mode = conv_border_mode
        self.pool_pad = pool_pad
        self.pool_ignore_border = pool_ignore_border
        self.activation = activation
        self.pool_mode = pool_mode
        self.pool_strides = pool_strides
        self.init = init
        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer
        self.bias = bias

        assert len(input_shape) == 4

        # check
        nb_batch, pre_nb_filter, pre_height, pre_width = input_shape
        if self.nb_filters.__class__.__name__ == 'int':
            nb_filters = [self.nb_filters for _ in self.filter_heights]
        else:
            nb_filters = self.nb_filters

        self.all_convs = []
        self.all_poolings = []
        for i in range(len(self.filter_heights)):
            # filter_shape = (nb_filters[i], pre_nb_filter, self.filter_heights[i], pre_width)
            conv = Convolution(input_shape, nb_filters[i], (self.filter_heights[i], pre_width),
                             strides=self.conv_strides, border_mode=self.conv_border_mode,
                             activation=self.activation, init=self.init,
                             W_regularizer=self.W_regularizer, b_regularizer=self.b_regularizer,
                             bias=self.bias)
            self.all_convs.append(conv)

            pool_size = (pre_height - self.filter_heights[i] + 1, 1)
            pool = Pooling(pool_size, self.pool_pad, self.pool_ignore_border,
                                            self.pool_mode, self.pool_strides)
            self.all_poolings.append(pool)

    def forward(self, input, **kwargs):
        outputs = []
        for i in range(len(self.filter_heights)):
            conv_out = self.all_convs[i].forward(input)
            pool_out = self.all_poolings[i].forward(conv_out)
            outputs.append(pool_out.flatten(2))
        return tensor.concatenate(outputs, axis=1)

    def to_json(self):
        config = {
            "nb_filters": self.nb_filters,
            "filter_heights": self.filter_heights,
            "input_shape": self.input_shape,
            "conv_strides": self.conv_strides,
            "conv_border_mode": self.conv_border_mode,
            "pool_pad": self.pool_pad,
            "pool_ignore_border": self.pool_ignore_border,
            "pool_mode": self.pool_mode,
            "pool_strides": self.pool_strides,
            "activation": self.activation,
            "init": self.init,
            "W_regularizer": self.W_regularizer,
            "b_regularizer": self.b_regularizer,
            "bias": self.bias,
        }
        return config

    @property
    def params(self):
        params = []
        for conv in self.all_convs:
            params.extend(conv.params)
        return params

    @property
    def regularizers(self):
        returns = []

        for conv in self.all_convs:
            returns.extend(conv.regularizers)

        return returns

    @property
    def updates(self):
        ups = super(NLPConvPooling, self).updates
        for conv in self.all_convs:
            ups.update(conv.updates)
        for pool in self.all_poolings:
            ups.update(pool.updates)
        return ups
