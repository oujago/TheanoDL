# -*- coding: utf-8 -*-


from collections import OrderedDict

from theano import function
from theano import tensor

from .layers import Dropout
from .utils import random
from .objective import CategoricalCrossEntropy
from .optimizer import SGD


TRAIN_TEST_SPLIT_LAYERS = [Dropout,]


class AbstractModel(object):
    def set_input(self, input_tensor=None, in_dim=None, in_tensor_type=None):
        raise NotImplementedError()

    def set_output(self, output_tensor=None, out_dim=None, out_tensor_type=None):
        raise NotImplementedError()

    def add(self, layer):
        raise NotImplementedError()

    def compile(self, loss, optimizer, **kwargs):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError

    @classmethod
    def from_json(cls, config):
        raise NotImplementedError()


class Model(AbstractModel):
    def __init__(self, seed=None):
        # seed
        self.seed = seed

        # other parameters
        self.train_test_split = False

        # function
        self.train = None
        self.predict = None

        # in and out
        self.input = None
        self.output = None

        # components
        self.layer_comp = []
        self.loss_comp = None
        self.optimizer_comp = None

    def set_input(self, input_tensor=None, in_dim=None, in_tensor_type=None):
        if input_tensor:
            self.input = input_tensor
        else:
            assert in_dim and in_tensor_type
            self.input = tensor.TensorType(in_tensor_type, [False] * in_dim)()

    def set_output(self, output_tensor=None, out_dim=None, out_tensor_type=None):
        if output_tensor:
            self.output = output_tensor
        else:
            assert out_dim and out_tensor_type
            self.output = tensor.TensorType(out_tensor_type, [False] * out_dim)()

    def add(self, layer):
        self.layer_comp.append(layer)
        self._check_train_test_split(layer)

    def compile(self, loss=CategoricalCrossEntropy(), optimizer=SGD(), **kwargs):

        self.loss_comp = loss
        self.optimizer_comp = optimizer

        # random seed
        if self.seed:
            random.set_seed(self.seed)

        # connect to
        pre_layer = None
        for layer in self.layer_comp:
            layer.connect_to(pre_layer)

        # forward
        train_prob_ys, train_ys, train_loss = self._forward(True)
        if self.train_test_split:
            predict_prob_ys, predict_ys, predict_loss = self._forward(False)
        else:
            predict_prob_ys, predict_ys, predict_loss = train_prob_ys, train_ys, train_loss

        # regularizers
        regularizers = []
        for layer in self.layer_comp:
            regularizers.extend(layer.regularizers)
        regularizer_loss = tensor.sum(regularizers)

        # loss
        losses = regularizer_loss + train_loss

        # params
        params = ()
        for layer in self.layer_comp:
            params += layer.params

        # layer updates
        layer_updates = OrderedDict()
        for layer in self.layer_comp:
            layer_updates.update(layer.updates)

        # model updates
        updates = self.optimizer_comp(params, sum(losses))
        updates.update(layer_updates)

        # train functions
        inputs = [self.input, self.output]
        if isinstance(self.optimizer_comp.learning_rate, tensor.TensorVariable):
            inputs.append(self.optimizer_comp.learning_rate)
        self.train = function(inputs=inputs,
                              outputs=[train_loss, regularizer_loss, train_ys],
                              updates=updates)

        # test functions
        self.predict = function(inputs=[self.input, self.output],
                                outputs=[predict_loss, predict_ys])

    def _forward(self, train=True):
        pre_layer_output = self.input
        for layer in self.layer_comp:
            if layer.__class__ in TRAIN_TEST_SPLIT_LAYERS:
                pre_layer_output = layer.forward(pre_layer_output, train=train)
            else:
                pre_layer_output = layer.forward(pre_layer_output)

        prob_ys = pre_layer_output
        ys = tensor.argmax(prob_ys, axis=1)
        loss = self.loss_comp(prob_ys, self.output)
        return prob_ys, ys, loss

    def to_json(self):

        # layer component
        layer_json = OrderedDict()
        for layer in self.layer_comp:
            layer_json[layer.__class__.__name__] = layer.to_json()

        # loss component
        loss_json = self.loss_comp.__class__.__name__

        # optimizer component
        optimizer_json = {
            self.optimizer_comp.__class__.__name__: self.optimizer_comp.to_json()
        }

        # configuration
        config = {
            'seed': self.seed,
            'layers': layer_json,
            'loss': loss_json,
            'optimizer': optimizer_json,
        }

        return config

    @classmethod
    def from_json(cls, config):
        raise NotImplementedError()

    def _check_train_test_split(self, layer):
        if layer.__class__ in TRAIN_TEST_SPLIT_LAYERS:
            self.train_test_split = True

