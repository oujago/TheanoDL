# -*- coding: utf-8 -*-


from thdl.base import ThdlObj


class AbstractNetwork(ThdlObj):
    def set_input_tensor(self, input_tensor=None, in_dim=None, in_tensor_type=None):
        raise NotImplementedError

    def set_output_tensor(self, output_tensor=None, out_dim=None, out_tensor_type=None):
        raise NotImplementedError

    def set_input_tensors(self, input_tensors):
        raise NotImplementedError

    def add_layer(self, layer):
        raise NotImplementedError

    def set_objective(self, loss_func):
        raise NotImplementedError

    def set_optimizer(self, optimizer):
        raise NotImplementedError

    def set_metrics(self, metrics):
        raise NotImplementedError

    def build(self, **kwargs):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

