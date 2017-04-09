# -*- coding: utf-8 -*-

from thdl.base import ThdlObj


class AbstractEvaluation(ThdlObj):
    def dock_gpu_metrics(self, metrics=None, train_metrics=None, predict_metrics=None):
        raise NotImplementedError

    def dock_gpu_train_metrics(self, train_metrics):
        raise NotImplementedError

    def dock_gpu_predict_metrics(self, predict_metrics):
        raise NotImplementedError

    def dock_index_to_tag(self, index_to_tags):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    def add_history(self, history_name, aspect, eval_outputs):
        raise NotImplementedError

    def add_training_history(self, history_name, outputs):
        raise NotImplementedError

    def add_trained_history(self, history_name, outputs):
        raise NotImplementedError

    def add_validation_history(self, history_name, outputs):
        raise NotImplementedError

    def add_test_history(self, history_name, outputs):
        raise NotImplementedError

    def output_epoch_evaluation(self, history_name, file):
        raise NotImplementedError

    def set_aspects(self, *args):
        raise NotImplementedError

