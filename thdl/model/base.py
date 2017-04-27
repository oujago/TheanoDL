# -*- coding: utf-8 -*-


from collections import OrderedDict

from theano import function
from theano import tensor

from thdl.model.tensors import get_tensor
from thdl.utils.random import set_seed
from thdl.utils.random import get_dtype
from .abstract import AbstractModel
from .layers import Dropout
from .objective import CategoricalCrossEntropy
from .optimizer import SGD
from . import metrics

TRAIN_TEST_SPLIT_LAYERS = [Dropout, ]


class Network(AbstractModel):
    def __init__(self, seed=None):
        # seed
        self.seed = seed

        # other parameters
        self.train_test_split = False

        # function
        self.train_func_for_eval = None
        self.predict_func_for_eval = None
        self.train_func_for_res = None
        self.predict_func_for_eval = None

        # in and out
        self.input_tensor = None
        self.output_tensor = None

        # components
        self.comp_layers = []
        self.comp_objective = None
        self.comp_optimizer = None

        # metric
        self.train_metrics = None
        self.predict_metrics = None

    def set_input_tensor(self, input_tensor=None, in_dim=None, in_tensor_type=None):
        if input_tensor:
            if isinstance(input_tensor, tensor.TensorVariable):
                self.input_tensor = input_tensor
            else:
                self.input_tensor = get_tensor(input_tensor)
        else:
            assert in_dim and in_tensor_type
            self.input_tensor = tensor.TensorType(in_tensor_type, [False] * in_dim)()

    def set_output_tensor(self, output_tensor=None, out_dim=None, out_tensor_type=None):
        if output_tensor:
            if isinstance(output_tensor, tensor.TensorVariable):
                self.output_tensor = output_tensor
            else:
                self.output_tensor = get_tensor(output_tensor)
        else:
            assert out_dim and out_tensor_type
            self.output_tensor = tensor.TensorType(out_tensor_type, [False] * out_dim)()

    def set_objective(self, loss_func):
        self.comp_objective = loss_func

    def set_optimizer(self, optimizer):
        self.comp_optimizer = optimizer

    def add_layer(self, layer):
        self.comp_layers.append(layer)
        self._check_train_test_split(layer)

    def set_metrics(self, metrics=None, train_metrics=None, predict_metrics=None):
        self.train_metrics = []
        self.predict_metrics = []

        if metrics:
            if type(metrics) in [tuple, list]:
                self.train_metrics.extend(metrics)
                self.predict_metrics.extend(metrics)
            else:
                self.train_metrics.append(metrics)
                self.predict_metrics.append(metrics)

        if train_metrics:
            if type(train_metrics) in [list, tuple]:
                self.train_metrics.extend(train_metrics)
            else:
                self.train_metrics.append(train_metrics)

        if predict_metrics:
            if type(predict_metrics) in [tuple, list]:
                self.predict_metrics.extend(predict_metrics)
            else:
                self.predict_metrics.append(predict_metrics)

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
            if layer.__class__ in TRAIN_TEST_SPLIT_LAYERS:
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

    def _check_train_test_split(self, layer):
        if layer.__class__ in TRAIN_TEST_SPLIT_LAYERS:
            self.train_test_split = True
