# -*- coding: utf-8 -*-


from collections import OrderedDict

from theano import function
from theano import tensor

from thdl.model import metrics
from thdl.model.layers import Dropout
from thdl.model.objective import CategoricalCrossEntropy
from thdl.model.optimizer import SGD
from thdl.model.tensors import get_tensor
from thdl.utils import is_iterable
from thdl.utils.random import get_dtype
from thdl.utils.random import set_seed
from .base import BaseNetwork

_TRAIN_TEST_SPLIT_LAYERS = [Dropout, ]


class Network(BaseNetwork):
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)

        # in and out
        self.input_tensor = None

    def set_input_tensor(self, input_tensor=None, in_dim=None, in_tensor_type=None):
        if input_tensor:
            if isinstance(input_tensor, tensor.TensorVariable):
                self.input_tensor = input_tensor
            else:
                self.input_tensor = get_tensor(input_tensor)
        else:
            assert in_dim and in_tensor_type
            self.input_tensor = tensor.TensorType(in_tensor_type, [False] * in_dim)()

    def add_layer(self, layer):
        self.comp_layers.append(layer)
        self._check_train_test_split(layer)

    def build(self, loss=CategoricalCrossEntropy(), optimizer=SGD(), **kwargs):

        self.comp_objective = loss
        self.comp_optimizer = optimizer

        # random seed
        if self.seed:
            set_seed(self.seed)

        # connect to
        pre_layer = None
        for layer in self.comp_layers:
            layer.connect_to(pre_layer)
            pre_layer = layer

        # forward
        train_prob_ys, train_ys, train_loss = self._forward(True)
        if self.train_test_split:
            predict_prob_ys, predict_ys, predict_loss = self._forward(False)
        else:
            predict_prob_ys, predict_ys, predict_loss = train_prob_ys, train_ys, train_loss

        # regularizers
        regularizers = []
        for layer in self.comp_layers:
            regularizers.extend(layer.regularizers)
        regularizer_loss = tensor.cast(tensor.sum(regularizers), get_dtype())
        # loss
        losses = regularizer_loss + train_loss

        # params
        params = []
        for layer in self.comp_layers:
            params += layer.params

        # layer updates
        layer_updates = OrderedDict()
        for layer in self.comp_layers:
            layer_updates.update(layer.updates)

        # model updates
        updates = self.comp_optimizer(params, losses)
        updates.update(layer_updates)

        # train functions
        if is_iterable(self.input_tensor):
            inputs = self.input_tensor + [self.output_tensor]
        else:
            inputs = [self.input_tensor, self.output_tensor]
        train_outputs = [train_ys, ]
        for metric in self.train_metrics:
            if isinstance(metric, metrics.Regularizer):
                train_outputs.append(regularizer_loss)
            elif isinstance(metric, metrics.Loss):
                train_outputs.append(train_loss)
            elif isinstance(metric, metrics.TotalLoss):
                train_outputs.append(losses)
            else:
                train_outputs.append(metric(train_prob_ys, self.output_tensor))
        self.train_func_for_eval = function(inputs=inputs,
                                            outputs=train_outputs,
                                            updates=updates)

        # test functions
        if is_iterable(self.input_tensor):
            inputs = self.input_tensor + [self.output_tensor]
        else:
            inputs = [self.input_tensor, self.output_tensor]
        test_outputs = [predict_ys, ]
        for metric in self.predict_metrics:
            if isinstance(metric, metrics.Loss):
                test_outputs.append(predict_loss)
            else:
                test_outputs.append(metric(predict_prob_ys, self.output_tensor))
        self.predict_func_for_eval = function(inputs=inputs,
                                              outputs=test_outputs)

    def _forward(self, train=True):
        pre_layer_output = self.input_tensor
        for layer in self.comp_layers:
            if layer.__class__ in _TRAIN_TEST_SPLIT_LAYERS:
                pre_layer_output = layer.forward(pre_layer_output, train=train)
            else:
                pre_layer_output = layer.forward(pre_layer_output)

        prob_ys = pre_layer_output
        ys = tensor.argmax(prob_ys, axis=1)
        loss = self.comp_objective(prob_ys, self.output_tensor)
        return prob_ys, ys, loss

    def to_json(self):

        # layer component
        layer_json = OrderedDict()
        for layer in self.comp_layers:
            layer_json[layer.__class__.__name__] = layer.to_json()

        # loss component
        loss_json = self.comp_objective.__class__.__name__

        # optimizer component
        optimizer_json = {
            self.comp_optimizer.__class__.__name__: self.comp_optimizer.to_json()
        }

        # configuration
        config = {
            'seed': self.seed,
            'layers': layer_json,
            'loss': loss_json,
            'optimizer': optimizer_json,
        }

        return config


class MultiInNetwork(Network):
    def __init__(self, **kwargs):
        super(MultiInNetwork, self).__init__(**kwargs)

        # in and out
        self.input_tensors = None

    def set_input_tensors(self, *input_tensors):
        if is_iterable(input_tensors[0]):
            self.input_tensors = input_tensors[0]
        else:
            self.input_tensors = input_tensors



