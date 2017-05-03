# -*- coding: utf-8 -*-

import sys

from thdl.data import AbstractData
from thdl.exeval import AbstractExeEval
from thdl.model import AbstractNetwork
from .abstract import AbstractTask


class BaseTask(AbstractTask):
    def __init__(self):
        # components
        self.model = None
        self.data = None
        self.exeval = None

        # log file
        self.logfile = sys.stdout

    def set_model(self, model):
        if model is None:
            return
        assert isinstance(model, AbstractNetwork)
        if self.model is None:
            self.model = model
        else:
            raise ValueError("Model already exists.")

    def set_data(self, data):
        if data is None:
            return
        assert isinstance(data, AbstractData)
        if self.data is None:
            self.data = data
        else:
            raise ValueError("Data already exists.")

    def set_exeval(self, exeval):
        if exeval is None:
            return
        assert isinstance(exeval, AbstractExeEval)
        if self.exeval is None:
            self.exeval = exeval
        else:
            raise ValueError("Execution already exists.")

    def set_logfile(self, logfile=sys.stdout):
        self.logfile = logfile

    def output_config(self):
        print("", file=self.logfile)
        for name, cls in (
                ('Model Class', self.model),
                ("Data Class", self.data),
                ("ExeEval Class", self.exeval),
        ):
            print('%s Parameter:' % name, file=self.logfile)
            for key, value in sorted(cls.to_json().items()):
                if isinstance(value, dict):
                    print("\t%s: " % key, file=self.logfile)
                    for key2, value2 in value.items():
                        print("\t\t%s = %s" % (key2, value2), file=self.logfile)
                else:
                    print("\t%s = %s" % (key, value), file=self.logfile)
            print("", file=self.logfile)
        self.logfile.flush()
