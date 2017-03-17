# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""

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
