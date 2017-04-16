# -*- coding: utf-8 -*-


from thdl.base import ThdlObj


class AbstractTask(ThdlObj):
    def output_config(self):
        raise NotImplementedError

    def hold_out_validation(self, *args, **kwargs):
        raise NotImplementedError

    def cross_validation(self, *args, **kwargs):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    def set_model(self, model):
        raise NotImplementedError

    def set_data(self, data):
        raise NotImplementedError

    def set_exeval(self, exeval):
        raise NotImplementedError

    def set_logfile(self, logfile):
        raise NotImplementedError





