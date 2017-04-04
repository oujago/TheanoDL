# -*- coding: utf-8 -*-

import numpy as np
from theano import tensor

from .base import Layer
from ..variables import dtype
from ..initializations import get_shared


class BatchNormal(Layer):
    """
    Copy from 'https://github.com/GabrielPereyra/norm-rnn'
    """

    def __init__(self, rng, n_in, epsilon=1e-6, momentum=0.9, axis=0,
                 beta_init='zero', gamma_init='one', params=None):
        super(BatchNormal, self).__init__()

        # parameters
        self.n_in = n_in
        self.epsilon = epsilon
        self.momentum = momentum
        self.axis = axis
        self.beta_init = beta_init
        self.gamma_init = gamma_init

        # variables
        self.running_mean = None
        self.running_std = None

        if params:
            self.beta, self.gamma = params
        else:
            self.beta = get_shared(rng, (n_in,), beta_init)
            self.gamma = get_shared(rng, (n_in,), gamma_init)

        # params
        self.train_params.extend([self.beta, self.gamma])

    def __call__(self, input, train=True):
        if input.ndim == 3:
            # init variables
            self.running_mean = tensor.alloc(np.cast[dtype](0.), input.shape[0], input.shape[1], self.n_in)
            self.running_std = tensor.alloc(np.cast[dtype](0.), input.shape[0], input.shape[1], self.n_in)
        elif input.ndim == 2:
            # init variables
            self.running_mean = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_in)
            self.running_std = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_in)
        else:
            raise ValueError("")

        # batch statistics
        m = tensor.mean(input, axis=self.axis)
        std = tensor.sqrt(tensor.mean(tensor.sqr(input - m) + self.epsilon, axis=self.axis))

        # update shared running averages
        mean_update = (1 - self.momentum) * self.running_mean + self.momentum * m
        std_update = (1 - self.momentum) * self.running_std + self.momentum * std
        self.updates[self.running_mean] = mean_update
        self.updates[self.running_std] = std_update

        # normalize using running averages
        # (is this better than batch statistics?)
        # (this version seems like it is using the running average
        #  of the previous batch since updates happens after)
        if train:
            input = (input - m) / (std + self.epsilon)
        else:
            input = (input - mean_update) / (std_update + self.epsilon)

        # scale and shift
        return self.gamma * input + self.beta

    def __str__(self):
        return "BatchNormal"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'axis': self.axis,
            'beta_init': self.beta_init,
            'gamma_init': self.gamma_init
        }
        return config
