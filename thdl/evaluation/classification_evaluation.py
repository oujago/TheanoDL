# -*- coding: utf-8 -*-


import os
import sys

import matplotlib
import numpy as np

from .base import AbstractEvaluation


class ClassificationEvaluation(AbstractEvaluation):
    def __init__(self,
                 cpu_metrics=('macro_acc', 'macro_recall', 'macro_f1', 'micro_acc'),
                 aspects=('training', 'trained', 'valid', 'test'),
                 metrics_to_choose_model=None,
                 split_line_len=20):

        # parameters
        self.index_to_tag = None
        self.gpu_metrics = None
        self.gpu_train_metrics = None
        self.gpu_predict_metrics = None

    def dock_gpu_train_metrics(self, train_metrics):
        self.gpu_train_metrics = train_metrics

    def dock_gpu_predict_metrics(self, predict_metrics):
        self.gpu_predict_metrics = predict_metrics

    def dock_gpu_metrics(self, metrics=None, train_metrics=None, predict_metrics=None):
        if metrics:
            self.dock_gpu_train_metrics(metrics)
            self.dock_gpu_predict_metrics(metrics)
        else:
            assert train_metrics and predict_metrics
            self.dock_gpu_train_metrics(train_metrics)
            self.dock_gpu_predict_metrics(predict_metrics)

    def dock_index_to_tag(self, index_to_tags):
        self.index_to_tag = index_to_tags
