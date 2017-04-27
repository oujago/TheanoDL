# -*- coding: utf-8 -*-

import time
from collections import Counter

import numpy as np

from thdl.utils.data_nlp_processing import yield_item
from thdl.utils.common import time_format
from .base import BaseTask


class ClassificationTask(BaseTask):
    def __init__(self):
        super(ClassificationTask, self).__init__()

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
        self.output_data_statistics_info('tests', test_ys)

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

        ####################################
        # Step 3: dock data and model
        #         with exeval
        ####################################

        self.exeval.dock_gpu_train_metrics(self.model.train_metrics)
        self.exeval.dock_gpu_predict_metrics(self.model.predict_metrics)
        self.exeval.dock_index_to_tag(self.data.get_index_to_tag())

        ####################################
        # Step 4: output the configurations
        #         of all models
        ####################################
        self.output_config()

        ####################################
        # Step 5: execution and evaluation
        ####################################
        for epoch in range(self.exeval.epochs):
            t1 = time.time()

            # training
            self.exeval.epoch_train_execution(history_name, self.model, train_xs, train_ys)

            # trained
            self.exeval.epoch_predict_execution(history_name, self.model, train_xs, train_ys, 'trained')

            # validation
            self.exeval.epoch_predict_execution(history_name, self.model, valid_xs, valid_ys, 'valid')

            # tests
            self.exeval.epoch_predict_execution(history_name, self.model, test_xs, test_ys, 'tests')

            # epoch evaluation
            self.exeval.output_epoch_evaluation(history_name, epoch, file=self.logfile)
            print("Used time %s" % time_format(time.time() - t1))
            self.logfile.flush()

        ##############################
        # Step 6: do total evaluations
        ##############################
        self.exeval.output_ho_bests(history_name, file=self.logfile)
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
