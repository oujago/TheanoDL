# -*- coding: utf-8 -*-


from .abstract import AbstractLayer
from collections import OrderedDict


class Layer(AbstractLayer):

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        return self.__class__.__name__

    def forward(self, input, **kwargs):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    @property
    def params(self):
        return []

    @property
    def regularizers(self):
        return []

    @property
    def updates(self):
        return OrderedDict()
