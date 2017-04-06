# -*- coding: utf-8 -*-

from theano.tensor.nnet import conv

from .base import Layer
from ..nonlinearity import ReLU
from ..initialization import GlorotUniform
from ..initialization import _zero


class Convolution(Layer):
    def __init__(self, nb_filter, filter_size, input_shape=None, strides=(1, 1),
                 border_mode='valid', activation=ReLU(), init=GlorotUniform(),
                 W_regularizer=None, b_regularizer=None, bias=True):
        super(Convolution, self).__init__()

        self.nb_filter = nb_filter
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.strides = strides
        self.border_mode = border_mode
        self.activation = activation
        self.init = init
        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer
        self.bias = bias

    def connect_to(self, pre_layer=None):
        if pre_layer is None:
            assert self.input_shape is not None
            input_shape = self.input_shape
        else:
            input_shape = pre_layer.out_shape

        # input_shape: (batch size, num input feature maps, image height, image width)
        assert len(input_shape) == 4

        nb_batch, pre_nb_filter, pre_height, pre_width = input_shape
        filter_height, filter_width = self.filter_size

        height = (pre_height - filter_height) // self.strides[0] + 1
        width = (pre_width - filter_width) // self.strides[1] + 1

        # output shape
        self.out_shape = (nb_batch, self.nb_filter, height, width)

        # filters
        self.W = self.init((self.nb_filter, pre_nb_filter, filter_height, filter_width))
        if self.bias:
            self.b = _zero((self.nb_filter,))

    def forward(self, input, **kwargs):
        conv_out = conv.conv2d(input=input, filters=self.W, image_shape=input.shape,
                               filter_shape=self.W.shape, subsample=self.strides)
        if self.bias:
            conv_out += self.b.dimshuffle('x', 0, 'x', 'x')
        output = self.activation(conv_out)

        return output

    def to_json(self):
        config = {
            'nb_filter': self.nb_filter,
            'filter_size': self.filter_size,
            'input_shape': self.input_shape,
            'strides': self.strides,
            'border_mode': self.border_mode,
            'activation': self.activation,
            'init': self.init,
            'W_regularizer': self.W_regularizer,
            'b_regularizer': self.b_regularizer,
            'bias': self.bias,
        }
        return config

    @property
    def params(self):
        if self.bias:
            return self.W, self.b
        else:
            return self.W

    @property
    def regularizers(self):
        returns = []

        if self.W_regularizer:
            returns.append(self.W_regularizer(self.W))

        if self.b_regularizer:
            returns.append(self.b_regularizer(self.b))

        return returns
