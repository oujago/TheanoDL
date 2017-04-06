# -*- coding: utf-8 -*-

from theano import tensor

from thdl.model.layers.base import Layer
from thdl.model.nonlinearity import get_activation
from thdl.model.initialization import get_shared

dot = tensor.dot


class Attention(Layer):
    def __init__(self, rng, n_in, activation='tanh',
                 init='orthogonal', context_init='glorot_uniform', ):
        super(Attention, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_in
        self.init = init
        self.context_init = context_init
        self.act_name = activation
        self.act_func = get_activation(activation)

        # variables
        self.att_W = get_shared(rng, (n_in, n_in), init)
        self.att_b = get_shared(None, (n_in,), 'zero')
        self.att_vec = get_shared(rng, (n_in,), context_init)

        self.train_params.extend([self.att_W, self.att_b, self.att_vec])
        self.reg_params.extend([self.att_W])

    def __call__(self, input):
        assert input.ndim == 3

        a = dot(self.act_func(dot(input, self.att_W) + self.att_b), self.att_vec)
        b = tensor.nnet.softmax(a)
        c = input.dimshuffle(2, 0, 1) * b
        d = tensor.sum(c, axis=2)
        e = d.dimshuffle(1, 0)

        return e

    def __str__(self):
        return 'Attention'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'activation': self.act_name,
            'init': self.init,
            'context_init': self.context_init
        }
        return config
