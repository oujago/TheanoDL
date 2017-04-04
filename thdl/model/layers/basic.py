# -*- coding: utf-8 -*-

from theano import tensor


from .base import Layer
from ..activation import Softmax as SoftmaxAct
from ..activation import Tanh
from ..initialization import GlorotUniform


class Dense(Layer):
    def __init__(self, n_out, n_in=None, activation=Tanh(), init=GlorotUniform(),
                 W_regularizer=None, b_regularizer=None):
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

    def connect_to(self, pre_layer=None):
        if pre_layer is None:
            assert self.n_in
            n_in = self.n_in
        else:
            n_in = pre_layer.output_shape[-1]

        self.output_shape = (None, self.n_out)
        self.W = self.init((n_in, self.n_out))
        self.b = self.init((self.n_out,))

    def forward(self, input, **kwargs):
        output = tensor.dot(input, self.W) + self.b
        output = self.activation(output)
        return output

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'activation': self.activation.__name__,
            'init': self.init.__name__
        }
        return config

    @property
    def params(self):
        return self.W, self.b

    @property
    def regularizers(self):
        returns = []

        if self.W_regularizer:
            returns.append(self.W_regularizer(self.W))

        if self.b_regularizer:
            returns.append(self.b_regularizer(self.b))

        return returns


class Softmax(Dense):
    def __init__(self, n_out, n_in=None, init=GlorotUniform(), W_regularizer=None, b_regularizer=None):
        super(Softmax, self).__init__(n_out, n_in, activation=SoftmaxAct(), init=init,
                                      W_regularizer=W_regularizer, b_regularizer=b_regularizer)

