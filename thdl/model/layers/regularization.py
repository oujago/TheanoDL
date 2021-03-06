# -*- coding: utf-8 -*-


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from thdl.utils.random import get_rng
from .base import Layer


class Dropout(Layer):
    def __init__(self, p, seed=None, input_shape=None):
        """
        :param p: the probability of dropping a unit
        """
        super(Dropout, self).__init__()

        self.p = p
        self.seed = seed
        self.input_shape = input_shape

        if seed is None:
            seed = get_rng().randint(1, 2147462579)
        self.srng = RandomStreams(seed)

    def forward(self, input, train=True):
        """
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :param train:
        :return:
        """
        # outputs
        if 0. < self.p < 1.:
            if train:
                mask = self.srng.binomial(n=1, p=1 - self.p, size=input.shape, dtype=input.dtype)
                output = input * mask / (1 - self.p)
            else:
                output = input * (1 - self.p)
        else:
            output = input
        return output

    def to_json(self):
        config = {
            'p': self.p,
            'seed': self.seed
        }
        return config
