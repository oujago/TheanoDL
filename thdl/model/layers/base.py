# -*- coding: utf-8 -*-


import json
from collections import OrderedDict


class Layer(object):
    def connect_to(self, pre_layer=None):
        raise NotImplementedError()

    def forward(self, input, **kwargs):
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
    def regularizers(self):
        return []

    @property
    def updates(self):
        return []
