# -*- coding:utf-8 -*-

from theano import tensor

from thdl.model.network.abstract import AbstractNetwork
from thdl.model.tensors import get_tensor
from thdl.utils import is_iterable
from thdl.utils.random import get_dtype
from thdl.utils.random import set_seed
from thdl.model import metrics
from thdl.model.layers import Dropout
from thdl.model.objective import CategoricalCrossEntropy
from thdl.model.optimizer import SGD


_TRAIN_TEST_SPLIT_LAYERS = [Dropout, ]


class BaseNetwork(AbstractNetwork):
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

        # components
        self.comp_layers = []
        self.comp_objective = None
        self.comp_optimizer = None

        # metric
        self.train_metrics = None
        self.predict_metrics = None

        # in and out
        self.output_tensor = None

    def set_objective(self, loss_func):
        self.comp_objective = loss_func

    def set_optimizer(self, optimizer):
        self.comp_optimizer = optimizer

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

    def _check_train_test_split(self, layer):
        if layer.__class__ in _TRAIN_TEST_SPLIT_LAYERS:
            self.train_test_split = True

    def set_output_tensor(self, output_tensor=None, out_dim=None, out_tensor_type=None):
        if output_tensor:
            if isinstance(output_tensor, tensor.TensorVariable):
                self.output_tensor = output_tensor
            else:
                self.output_tensor = get_tensor(output_tensor)
        else:
            assert out_dim and out_tensor_type
            self.output_tensor = tensor.TensorType(out_tensor_type, [False] * out_dim)()
