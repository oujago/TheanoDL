# -*- coding: utf-8 -*-


from thdl.base import ThdlObj


class AbstractTask(ThdlObj):
    def output_config(self):
        raise NotImplementedError

    def hold_out_validation(self):
        raise NotImplementedError

    def cross_validation(self):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    def set_model(self, model):
        raise NotImplementedError

    def set_data(self, data):
        raise NotImplementedError

    def set_execution(self, execution):
        raise NotImplementedError

    def set_evaluation(self, evaluation):
        raise NotImplementedError

    def set_logfile(self, logfile):
        raise NotImplementedError





