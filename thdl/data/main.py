# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 2017/3/18

@notes:
    Define the abstract data class
"""


class Data:
    def get_xs(self):
        raise NotImplementedError("Please implement 'get_xs' method.")

    def get_ys(self):
        raise NotImplementedError("Please implement 'get_ys' method.")

    def to_json(self):
        raise NotImplementedError("Please implement 'to_json' method.")


