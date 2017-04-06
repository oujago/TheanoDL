# -*- coding: utf-8 -*-

import sys
import time
from collections import Counter

import numpy as np

from thdl.task.base import AbstractTask
from thdl.data import AbstractData
from thdl.evaluation import AbstractEvaluation
from thdl.execution import AbstractExecution
from thdl.model import AbstractModel
from thdl.utils.data_nlp_processing import yield_item
from thdl.utils.usual import time_format


class ClassificationTask(AbstractTask):
    # def __init__(self, model=None, data=None, evaluation=None, execution=None,
    #              logfile=sys.stdout):
    #     # components
    #     self.model = None
    #     self.data = None
    #     self.evaluation = None
    #     self.execution = None
    #
    #     self.add_model(model)
    #     self.add_data(data)
    #     self.add_execution(execution)
    #     self.add_evaluation(evaluation)
    #
    #     # log file
    #     self.logfile = logfile
    def __init__(self):
        # components
        self.model = None
        self.data = None
        self.evaluation = None
        self.execution = None

        # log file
        self.logfile = sys.stdout

    def add_model(self, model):
        if model is None:
            return
        assert isinstance(model, AbstractModel)
        if self.model is None:
            self.model = model
        else:
            raise ValueError("Model already exists.")

    def add_data(self, data):
        if data is None:
            return
        assert isinstance(data, AbstractData)
        if self.data is None:
            self.data = data
        else:
            raise ValueError("Data already exists.")

    def add_execution(self, execution):
        if execution is None:
            return
        assert isinstance(execution, AbstractExecution)
        if self.execution is None:
            self.execution = execution
        else:
            raise ValueError("Execution already exists.")

    def add_evaluation(self, evaluation):
        if evaluation is None:
            return
        assert isinstance(evaluation, AbstractEvaluation)
        if self.evaluation is None:
            self.evaluation = evaluation
        else:
            raise ValueError("Evaluation already exists.")

    def set_logfile(self, logfile):
        self.logfile = logfile

    def output_config(self):
        print("", file=self.logfile)
        for name, cls in (
                ('Model Class', self.model),
                ("Data Class", self.data),
                ("Execute Class", self.execution),
                ("Evaluation Class", self.evaluation)
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

    def hold_out_validation(self, name=None):
        t0 = time.time()
        history_name = name or 'default'

        ##############################
        # Step 1: get data
        ##############################
        if self.data.index2tag:
            print("DataProvider has been build.\n", file=self.logfile)
        else:
            t1 = time.time()
            print("building data ...", file=self.logfile)
            self.model.build()
            print("building done, used %s.\n" % time_format(time.time() - t1), file=self.logfile)

        train_xs, train_ys = self.data.get_train_data()
        self.output_data_statistics_info('train', train_ys)

        valid_xs, valid_ys = self.data.get_valid_data()
        self.output_data_statistics_info('validation', valid_ys)

        test_xs, test_ys = self.data.get_test_data()
        self.output_data_statistics_info('test', test_ys)

        ##############################
        # Step 2: build model
        ##############################
        if self.model.func_train and self.model.func_predict:
            print("Model has been build.\n", file=self.logfile)
        else:
            t1 = time.time()
            print("building model ...", file=self.logfile)
            self.model.build()
            print("building done, used %s.\n" % time_format(time.time() - t1), file=self.logfile)

        ##############################
        # Step 3: execute model
        ##############################

        epoch_train_execution = self.execution.train_execution
        epoch_predict_execution = self.execution.predict_execution

        self.evaluation.dock_train_metrics(self.model.train_metrics)
        self.evaluation.dock_predict_metrics(self.model.predict_metrics)

        for epoch in range(self.execution.epochs):
            t1 = time.time()

            # training
            outputs = epoch_train_execution(self.model, train_xs, train_ys)
            if 'training' in self.evaluation.aspects:
                self.evaluation.add_training_history(history_name, outputs)

            # trained
            if 'trained' in self.evaluation.aspects:
                outputs = epoch_predict_execution(self.model, train_xs, train_ys)
                self.evaluation.add_trained_history(history_name, outputs)

            # validation
            if 'valid' in self.evaluation.aspects:
                outputs = epoch_predict_execution(self.model, valid_xs, valid_ys)
                self.evaluation.add_validation_history(history_name, outputs)
            else:
                raise OSError("Model must valid.")

            # test
            if 'test' in self.evaluation.aspects:
                outputs = epoch_predict_execution(self.model, test_xs, test_ys)
                self.evaluation.add_test_history(history_name, outputs)

            self.execution.output_epoch(history_name, epoch, file=self.logfile)
            print("Used time %s" % time_format(time.time() - t1))
            self.logfile.flush()

        ##############################
        # Step 4: do evaluation
        ##############################
        self.execution.output_bests(history_name, file=self.logfile)
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
            counter = Counter(yield_item(np.argmax(ys, axis=0)))
        else:
            raise ValueError

        for key, value in sorted(counter.items()):
            print("\t%s data number - %d" % (self.data.index2tag[key], value), file=self.logfile)
        print("", file=self.logfile)
