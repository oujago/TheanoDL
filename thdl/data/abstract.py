# -*- coding: utf-8 -*-

from thdl.base import ThdlObj


class AbstractData(ThdlObj):
    index_to_tag = None

    def to_json(self):
        raise NotImplementedError

    def get_train_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_valid_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_test_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_index_to_tag(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError
