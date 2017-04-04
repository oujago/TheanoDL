# -*- coding: utf-8 -*-


from theano import tensor


class Regularizer(object):
    def __init__(self, l1=0., l2=0.):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, param):
        return self.call(param)

    def call(self, param):
        regularization = 0.

        if self.l1 > 0.:
            regularization += tensor.sum(tensor.abs_(param) * self.l1)

        if self.l2 > 0.:
            regularization += tensor.sum(tensor.square(param) * self.l2)

        return regularization
