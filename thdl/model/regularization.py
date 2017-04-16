# -*- coding: utf-8 -*-


from theano import tensor

from thdl.base import ThdlObj


class Regularizer(ThdlObj):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, param):
        return self.call(param)

    def call(self, param):
        assert param

        if self.l1 > 0. and self.l2 > 0.:
            return tensor.sum(tensor.abs_(param) * self.l1) + tensor.sum(tensor.square(param) * self.l2)

        if self.l1 > 0.:
            return tensor.sum(tensor.abs_(param) * self.l1)

        if self.l2 > 0.:
            return tensor.sum(tensor.square(param) * self.l2)

        raise ValueError

    def to_json(self):
        config = {
            'l1': self.l1,
            "l2": self.l2
        }
        return config


