# -*- coding: utf-8 -*-


from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .base import Layer
from ..utils.random import get_rng


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
            seed = get_rng().randint(10000, 100000000000)
        self.srng = RandomStreams(seed)

    def connect_to(self, pre_layer=None):
        if pre_layer is None:
            self.output_shape = pre_layer.output_shape
        else:
            assert self.input_shape is not None
            self.output_shape = self.input_shape

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
