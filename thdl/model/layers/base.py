# -*- coding: utf-8 -*-


import json
from collections import OrderedDict


class Layer(object):
    def connet_to(self, pre_layer=None):
        raise NotImplementedError()

    def forward(self, pre_layer, **kwargs):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()

    @classmethod
    def from_json(cls, config):
        if type(config).__name__ == 'str':
            config = json.loads(config)

        if type(config).__name__ == 'dict':
            return cls(**config)

        else:
            raise ValueError("config must be dict object.")

    @property
    def params(self):
        return []

    @property
    def regulars(self):
        return []

    # def __str__(self):
    #     raise NotImplementedError("Every Layer class should implement the '__str__' method.")
    #
    # def __call__(self, *args, **kwargs):
    #     raise NotImplementedError("Every Layer class should implement the '__call__' method.")
