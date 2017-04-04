# -*- coding: utf-8 -*-

from ..layers import Layer


class SubNet(Layer):
    def __init__(self):
        super(SubNet, self).__init__()

        self.outputs_info_len = 0

    def __str__(self):
        return 'SubNet'

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("")

    def to_json(self):
        raise NotImplementedError("")
