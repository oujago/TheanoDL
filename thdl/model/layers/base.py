# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 2017/3/17

@notes:
    
"""

import json
from collections import OrderedDict


class Layer(object):
    @classmethod
    def from_json(cls, config):
        if type(config).__name__ == 'str':
            config = json.loads(config)

        if type(config).__name__ == 'dict':
            return cls(**config)

        else:
            raise ValueError("config must be dict object.")

    def to_json(self):
        raise NotImplementedError("Every Layer class should implement the 'to_json' method.")

    def __init__(self, ):
        self.train_params = []
        self.reg_params = []
        self.updates = OrderedDict()
        self.masks = []

    def __str__(self):
        raise NotImplementedError("Every Layer class should implement the '__str__' method.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Every Layer class should implement the '__call__' method.")
