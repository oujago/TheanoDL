# -*- coding: utf-8 -*-


from .abstract import AbstractLayer


class Layer(AbstractLayer):
    output_shape = None

    def connect_to(self, pre_layer=None):
        raise NotImplementedError

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
        return []
