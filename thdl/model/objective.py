# -*- coding: utf-8 -*-

from theano import tensor


class Objective(object):
    def __call__(self, outputs, targets):
        return self.call(outputs, targets)

    def call(self, outputs, targets):
        raise NotImplementedError()


class MeanSquaredError(Objective):
    def call(self, outputs, targets):
        return tensor.mean((outputs - targets) ** 2)


class Hinge(Objective):
    def call(self, outputs, targets):
        return tensor.mean(tensor.sqr(tensor.maximum(1. - outputs * targets, 0.)), axis=-1)


class BinaryCrossEntropy(Objective):
    def call(self, outputs, targets):
        return tensor.mean(tensor.nnet.binary_crossentropy(outputs, targets))


class CategoricalCrossEntropy(Objective):
    def call(self, outputs, targets):
        return tensor.mean(tensor.nnet.categorical_crossentropy(outputs, targets))


class MeanAbsoluteError(Objective):
    def call(self, outputs, targets):
        return tensor.mean(tensor.abs_(outputs - targets), axis=-1)


class NegativeLogLikelihood(Objective):
    def call(self, outputs, targets):
        if outputs.ndim == targets.ndim:
            return -tensor.mean(targets * tensor.log(outputs) + (1 - targets) * tensor.log(1 - outputs))
        else:
            return -tensor.mean(tensor.log(outputs)[tensor.arange(targets.shape[0]), targets])


MSE = MeanSquaredError
NLL = NegativeLogLikelihood
MAE = MeanAbsoluteError
CCE = CategoricalCrossEntropy
BCE = BinaryCrossEntropy
