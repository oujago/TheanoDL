# -*- coding: utf-8 -*-

from theano import tensor

from thdl.model.layers.base import Layer


class InOutLayer(Layer):
    """
    Define Input and Output Symbolic variables
    """
    def __init__(self, in_tensor=None, out_tensor=None,
                 in_dim=None, in_tensor_type=None,
                 out_dim=None, out_tensor_type=None,):
        super(InOutLayer, self).__init__()

        self.in_dim = in_dim
        self.in_tensor_type = in_tensor_type
        self.out_dim = out_dim
        self.out_tensor_type = out_tensor_type
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor

    def connect_to(self, pre_layer=None):
        assert pre_layer is None
        assert self.in_dim or self.input
        assert self.out_dim or self.output

        if self.in_tensor:
            self.input = self.in_tensor
        else:
            self.input = tensor.TensorType(self.in_tensor_type, [False] * self.in_dim)()

        if self.out_tensor:
            self.output = self.out_tensor
        else:
            self.output = tensor.TensorType(self.out_tensor_type, [False] * self.out_dim)()


    def to_json(self):
        config = {
            'x_ndim': self.x_ndim,
            'x_tensor_type': self.x_tensor_type,
            'y_ndim': self.y_ndim,
            'y_tensor_type': self.y_tensor_type,
            'X': self.X_,
            'Y': self.Y_
        }

        return config


class TCInOutLayer(InOutLayer):
    def __init__(self):
        pass

