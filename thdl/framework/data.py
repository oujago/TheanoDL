# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/17

@notes:
    
"""


class DataCls:
    def get_xs(self):
        raise NotImplementedError("Please implement 'get_xs' method.")

    def get_ys(self):
        raise NotImplementedError("Please implement 'get_ys' method.")

    def to_json(self):
        raise NotImplementedError("Please implement 'to_json' method.")

