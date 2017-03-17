# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/17

@notes:
    
"""

from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .base import Layer
from ..theano_tools import get_activation
from ..theano_tools import get_shared

dot = tensor.dot


class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__()
        self.act_name = activation
        self.act_func = get_activation(activation, **self.kwargs)
        self.kwargs = kwargs

    def __call__(self, input):
        return self.act_func(input)

    def __str__(self):
        return 'Activation'

    def to_json(self):
        config = {
            'activation': self.act_name,
            'kwargs': self.kwargs
        }
        return config


class Dropout(Layer):
    def __init__(self, rng, p):
        """
        :param rng: a random number generator used to initialize weights
        :param p: the probability of dropping a unit
        """
        super(Dropout, self).__init__()

        # parameters
        self.srng = RandomStreams(seed=rng.randint(100, 2147462579))
        self.p = p

    def __call__(self, input, train=True):
        """
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :param train:
        :return:
        """
        # outputs
        if 0. < self.p < 1.:
            if train:
                output = input * self.srng.binomial(
                    n=1, p=1 - self.p, size=input.shape,
                    dtype=input.dtype) / (1 - self.p)
            else:
                output = input * (1 - self.p)
        else:
            output = input
        return output

    def __str__(self):
        return 'Dropout'

    def to_json(self):
        config = {
            'p': self.p
        }
        return config


class Dense(Layer):
    def __init__(self, rng, n_in, n_out,
                 activation='tanh', init='glorot_uniform',
                 W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :param rng: a random number generator used to initialize weights
        :param n_in: dimensionality of input
        :param n_out: number of hidden units
        :param activation: Non linearity to be applied in the hidden layer
        :param init: parameter initialize
        """
        super(Dense, self).__init__()

        # parameters
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init

        # variables
        self.W = get_shared(rng, (n_in, n_out), init) if W is None else W
        self.b = get_shared(rng, (n_out,), 'zero') if b is None else b

        # params
        self.train_params.extend([self.W, self.b])
        self.reg_params.extend([self.W])

    def __call__(self, input):
        # outputs
        output = dot(input, self.W) + self.b
        output = self.act_func(output)

        return output

    def __str__(self):
        return "Dense"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'activation': self.act_name,
            'init': self.init
        }
        return config



class Softmax(Layer):
    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform',
                 W=None, b=None):
        super(Softmax, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init

        # variables
        self.W = get_shared(rng, (n_in, n_out), init) if W is None else W
        self.b = get_shared(rng, (n_out,), 'zero') if b is None else b

        # params
        self.train_params.extend([self.W, self.b])
        self.reg_params.extend([self.W])

    def __call__(self, input):
        # outputs
        output = tensor.nnet.softmax(dot(input, self.W) + self.b)
        return output

    def __str__(self):
        return 'Softmax'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init
        }
        return config


