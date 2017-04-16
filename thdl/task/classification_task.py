# -*- coding: utf-8 -*-

import sys
import time
from collections import Counter

import numpy as np

from thdl.exeval import AbstractExeEval
from thdl.data import AbstractData
from thdl.model import AbstractModel
from thdl.task.base import AbstractTask
from thdl.utils.data_nlp_processing import yield_item
from thdl.utils.usual import time_format


class ClassificationTask(AbstractTask):
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
        assert isinstance(model, AbstractModel)
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

    def hold_out_validation(self, name=None, plot=False):
        t0 = time.time()
        history_name = name or 'default'

        ##############################
        # Step 1: get data
        ##############################
        if self.data.index_to_tag:
            print("DataProvider has been build.\n", file=self.logfile)
        else:
            t1 = time.time()
            print("building data ...", file=self.logfile)
            self.data.build()
            print("building done, used %s.\n" % time_format(time.time() - t1), file=self.logfile)
            assert self.data.index_to_tag

        train_xs, train_ys = self.data.get_train_data()
        self.output_data_statistics_info('train', train_ys)

        valid_xs, valid_ys = self.data.get_valid_data()
        self.output_data_statistics_info('validation', valid_ys)

        test_xs, test_ys = self.data.get_test_data()
        self.output_data_statistics_info('test', test_ys)

        ##############################
        # Step 2: build model
        ##############################
        if self.model.train_func_for_eval and self.model.train_func_for_eval:
            print("Model has been build.\n", file=self.logfile)
        else:
            t1 = time.time()
            print("building model ...", file=self.logfile)
            self.model.build()
            print("building done, used %s.\n" % time_format(time.time() - t1), file=self.logfile)
            assert self.model.train_func_for_eval and self.model.train_func_for_eval

        ##############################
        # Step 3: execute model
        ##############################

        self.exeval.dock_gpu_train_metrics(self.model.train_metrics)
        self.exeval.dock_gpu_predict_metrics(self.model.predict_metrics)
        self.exeval.dock_index_to_tag(self.data.get_index_to_tag())

        for epoch in range(self.exeval.epochs):
            t1 = time.time()

            # training
            self.exeval.epoch_train_execution(history_name, self.model, train_xs, train_ys)

            # trained
            self.exeval.epoch_predict_execution(history_name, self.model, train_xs, train_ys, 'trained')

            # validation
            self.exeval.epoch_predict_execution(history_name, self.model, valid_xs, valid_ys, 'valid')

            # test
            self.exeval.epoch_predict_execution(history_name, self.model, test_xs, test_ys, 'test')

            # output
            self.exeval.output_epoch_evaluation(history_name, epoch, file=self.logfile)
            print("Used time %s" % time_format(time.time() - t1))
            self.logfile.flush()

        ##############################
        # Step 4: do evaluation
        ##############################
        self.exeval.output_bests(history_name, file=self.logfile)
        print("Used time %s" % time_format(time.time() - t0), file=self.logfile)
        self.logfile.flush()

    def cross_validation(self):
        pass

    def to_json(self):
        config = {

        }
        return config

    def output_data_statistics_info(self, aspect, ys):
        if ys is None:
            return

        print("In %s data:" % aspect, file=self.logfile)
        print("\tTotal data number - %d" % ys.shape[0], file=self.logfile)

        if np.ndim(ys) == 1:
            counter = Counter(yield_item(ys))
        elif np.ndim(ys) == 2:
            counter = Counter(yield_item(np.argmax(ys, axis=1)))
        else:
            raise ValueError

        for key, value in sorted(counter.items()):
            print("\t%s data number - %d" % (self.data.index_to_tag[key], value), file=self.logfile)
        print("", file=self.logfile)
