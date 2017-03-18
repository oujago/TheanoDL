# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/17

@notes:
    
"""
from theano import tensor

from .base import Layer


class XY(Layer):
    def __init__(self, X=None, Y=None,
                 x_ndim=2, x_tensor_type='int32',
                 y_ndim=1, y_tensor_type='int32'):
        """
        :param X:
        :param Y:
        :param x_ndim:
        :param x_tensor_type:
        :param y_ndim:
        :param y_tensor_type:
        """

        super(XY, self).__init__()

        self.x_ndim = x_ndim
        self.x_tensor_type = x_tensor_type
        self.y_ndim = y_ndim
        self.y_tensor_type = y_tensor_type
        self.X_ = X
        self.Y_ = Y

        self.X = tensor.TensorType(self.x_tensor_type, [False] * x_ndim)() if X is None else X
        self.Y = tensor.TensorType(self.y_tensor_type, [False] * y_ndim)() if Y is None else Y

    def __str__(self):
        return 'XY'

    def __call__(self, *args, **kwargs):
        return

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
