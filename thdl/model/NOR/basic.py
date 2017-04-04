# -*- coding: utf-8 -*-

from theano import tensor

from .base import SubNet
from ..activation import get_activation
from ..variables import get_shared

dot = tensor.dot


class MLP(SubNet):
    def __init__(self, rng, n_in, n_out,
                 activation='tanh', init='glorot_uniform', bias=True):
        super(MLP, self).__init__()

        # parameters
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init
        self.bias = bias

        # variables
        self.W = get_shared(rng, (n_in, n_out), init)
        self.b = get_shared(rng, (n_out,), 'zero')

        # params
        if bias:
            self.train_params.extend([self.W, self.b])
        else:
            self.train_params.extend([self.W, ])
        self.reg_params.extend([self.W])

    def __str__(self):
        return 'MLPSubNet'

    def __call__(self, input, ):
        if self.bias:
            output = self.act_func(dot(input, self.W) + self.b)
        else:
            output = self.act_func(dot(input, self.W))
        return output

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'activation': self.act_name,
            'init': self.init,
            'bias': self.bias
        }
        return config
