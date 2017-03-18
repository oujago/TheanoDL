# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""
from theano import tensor

from .base import SubNet
from ..variables import get_shared


class BN(SubNet):
    def __init__(self, rng, n_in, epsilon=1e-6, axis=0,
                 beta_init='zero', gamma_init='one', ):
        super(BN, self).__init__()

        # parameters
        self.n_in = n_in
        self.epsilon = epsilon
        self.axis = axis
        self.beta_init = beta_init
        self.gamma_init = gamma_init

        # variables
        self.beta = get_shared(None, (n_in,), beta_init)
        self.gamma = get_shared(rng, (n_in,), gamma_init)

        # params
        self.train_params.extend([self.beta, self.gamma])

    def __call__(self, input):
        m = tensor.mean(input, axis=self.axis)
        std = tensor.sqrt(tensor.mean(tensor.sqr(input - m), axis=self.axis))
        x = (input - m) / (std + 1e-8)
        return self.gamma * x + self.beta

    def __str__(self):
        return "BNSubNet"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'epsilon': self.epsilon,
            'axis': self.axis,
            'beta_init': self.beta_init,
            'gamma_init': self.gamma_init
        }
        return config
