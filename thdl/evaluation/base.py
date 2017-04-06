# -*- coding: utf-8 -*-

from thdl.base import ThdlObj


class AbstractEvaluation(ThdlObj):
    def to_json(self):
        raise NotImplementedError
