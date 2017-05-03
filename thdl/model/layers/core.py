# -*- coding: utf-8 -*-

from theano import tensor


from .base import Layer
from ..nonlinearity import Softmax as SoftmaxAct
from ..nonlinearity import Tanh
from ..initialization import GlorotUniform
from ..initialization import _zero


class Dense(Layer):
    def __init__(self, n_in, n_out, activation=Tanh(), init=GlorotUniform(),
                 W_regularizer=None, b_regularizer=None, bias=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :param n_in: dimensionality of input
        :param n_out: number of hidden units
        :param activation: Non linearity to be applied in the hidden layer
        :param init: parameter initialize
        """
        super(Dense, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.init = init
        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer
        self.bias = bias

        self.W = self.init((n_in, self.n_out))
        if self.bias:
            self.b = _zero((self.n_out,))

    def forward(self, input, **kwargs):
        output = tensor.dot(input, self.W)
        if self.bias:
            output += self.b
        output = self.activation(output)
        return output

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'activation': self.activation,
            'init': self.init,
            'bias': self.bias
        }
        return config

    @property
    def params(self):
        if self.bias:
            return [self.W, self.b]
        else:
            return [self.W, ]

    @property
    def regularizers(self):
        returns = []

        if self.W_regularizer:
            returns.append(self.W_regularizer(self.W))

        if self.b_regularizer:
            returns.append(self.b_regularizer(self.b))

        return returns


class Softmax(Dense):
    def __init__(self, n_in, n_out, init=GlorotUniform(), W_regularizer=None, b_regularizer=None, bias=True):
        super(Softmax, self).__init__(n_in, n_out,
                                      activation=SoftmaxAct(),
                                      init=init,
                                      W_regularizer=W_regularizer,
                                      b_regularizer=b_regularizer,
                                      bias=bias)

