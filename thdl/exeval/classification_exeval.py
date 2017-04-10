# -*- coding: utf-8 -*-


import numpy as np
from .base import AbstractExeEval


class ClassifyExeEval(AbstractExeEval):
    def __init__(self, batch_size, epochs=50):
        """
        >>>
        >>> self.histories = {
        >>>     "folder-0": {
        >>>         'training': [
        >>>             [], # epoch 0
        >>>             [], # epoch 1
        >>>          ],
        >>>         'trained': [
        >>>             [], # epoch 0
        >>>             [], # epoch 1
        >>>          ],
        >>>         'valid': [
        >>>             [], # epoch 0
        >>>             [], # epoch 1
        >>>          ],
        >>>         'test': [
        >>>             [], # epoch 0
        >>>             [], # epoch 1
        >>>          ],
        >>>     }
        >>> }
        >>> 

        """
        # parameters to dock
        self.index_to_tag = None
        self.gpu_metrics = None
        self.gpu_train_metrics = None
        self.gpu_predict_metrics = None

        # parameters to set
        self.aspects = None
        self.cpu_metrics = None
        self.model_chosen_metrics = None

        #
        self.batch_size = batch_size
        self.epochs = epochs

        #
        self.histories = {}

    def dock_gpu_train_metrics(self, train_metrics):
        self.gpu_train_metrics = []
        for metric in train_metrics:
            self.gpu_train_metrics.append(metric.__class__.__name__)

    def dock_gpu_predict_metrics(self, predict_metrics):
        self.gpu_predict_metrics = []
        for metric in predict_metrics:
            self.gpu_predict_metrics.append(metric.__class__.__name__)

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

    def set_aspects(self, *args):
        for aspect in args:
            if aspect not in ['training', 'trained', 'valid', 'test']:
                raise ValueError
        self.aspects = tuple(args)

    def set_cpu_metrics(self, *args):
        cpu_metrics = set([])
        for metric in args:
            if metric in ['micro_acc', 'micro_recall', 'micro_f1', 'micro']:
                cpu_metrics.add('micro')
            elif metric in ['macro_acc', 'macro_recall', 'macro_f1']:
                cpu_metrics.add(metric)
            else:
                raise ValueError
        self.cpu_metrics = sorted(cpu_metrics)

    def set_model_chosen_metrics(self, *args):
        self.model_chosen_metrics = args

    def to_json(self):
        config = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }
        return config

    def add_history(self, history_name, aspect, outputs):
        assert aspect in self.aspects

        if history_name not in self.histories:
            self.histories[history_name] = {}

        if aspect not in self.histories[history_name]:
            self.histories[history_name][aspect] = []

        self.histories[history_name][aspect].append(outputs)

    def add_training_history(self, history_name, outputs):
        self.add_history(history_name, 'training', outputs)

    def add_trained_history(self, history_name, outputs):
        self.add_history(history_name, 'trained', outputs)

    def add_validation_history(self, history_name, outputs):
        self.add_history(history_name, 'valid', outputs)

    def add_test_history(self, history_name, outputs):
        self.add_history(history_name, 'test', outputs)

    def output_epoch_evaluation(self, history_name, epoch, file, end="; "):
        # epoch
        runout = 'epoch %d%s' % (epoch, end)

        # aspects
        for aspect in self.aspects:
            outputs = self.histories[history_name][aspect][-1]
            if aspect == 'training':
                temps = ["%s:%.4f" % (self.gpu_train_metrics[i], outputs[i]) for i in
                         range(len(self.gpu_train_metrics))]
            else:
                temps = ["%s:%.4f" % (self.gpu_predict_metrics[i], outputs[i]) for i in
                         range(len(self.gpu_predict_metrics))]
            aspect_runout = '%s-[%s]%s' % (aspect, ' '.join(temps), end)
            runout += aspect_runout

        # times

        # output
        print(runout, file=file)

    def output_bests(self, history_name, file):
        pass

    def _execution(self, x_data, y_data, func_to_exe):
        total_len = x_data.shape[0]
        nb_batches = len(y_data) // self.batch_size
        outputs = []

        i = 0
        for i in range(nb_batches):
            xs = x_data[self.batch_size * i: self.batch_size * (i + 1)]
            ys = y_data[self.batch_size * i: self.batch_size * (i + 1)]
            res = func_to_exe(xs, ys)

            outputs.append(list(res[1:]))

        else:
            if self.batch_size * (i + 1) < total_len:
                actual_len = total_len - self.batch_size * (i + 1)

                xs = x_data[-actual_len:]
                ys = y_data[-actual_len:]

                res = func_to_exe(xs, ys)

                outputs.append(list(res[1:]))

        outputs = np.mean(np.asarray(outputs), axis=1)

        return outputs

    def train_execution(self, model, all_xs, all_ys):
        return self._execution(all_xs, all_ys, model.train_func_for_eval)

    def predict_execution(self, model, all_xs, all_ys):
        return self._execution(all_xs, all_ys, model.predict_func_for_eval)
