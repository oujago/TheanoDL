# -*- coding: utf-8 -*-


import numpy as np
from .base import AbstractExeEval
from . import matrix_eval


class ClassifyExeEval(AbstractExeEval):
    def __init__(self, batch_size, epochs=50):
        """
        >>> # the format of `self.histories` attribute 
        >>> epoch_out0 = None
        >>> epoch_out1 = None
        >>> self.histories = {
        >>>     "folder-0": {
        >>>         # aspect        epoch 0 (list type)     epoch 1 (list type)
        >>>         'training':     [epoch_out0, epoch_out1],
        >>>         'trained':      [epoch_out0, epoch_out1],
        >>>         'valid':        [epoch_out0, epoch_out1],
        >>>         'test':         [epoch_out0, epoch_out1],
        >>>     }
        >>> }
        >>> 
        >>> eval_mat0= None
        >>> eval_mat1 = None
        >>> history_evaluation_matrices = {
        >>>     'folder0': {
        >>>         # apect         epoch0 (numpy.array)      epoch1 (numpy.array)
        >>>         "training":     [eval_mat0, eval_mat1],
        >>>         'trained':      [eval_mat0, eval_mat1],
        >>>         'valid':        [eval_mat0, eval_mat1],
        >>>         'test':         [eval_mat0, eval_mat1]
        >>>     }
        >>>     'folder1': {
        >>>         # apect           epoch0      epoch1
        >>>         "training":     [eval_mat0, eval_mat1],
        >>>         'trained':      [eval_mat0, eval_mat1],
        >>>         'valid':        [eval_mat0, eval_mat1],
        >>>         'test':         [eval_mat0, eval_mat1]
        >>>     }
        >>> }
        >>>
        >>> conf_mat0 = None
        >>> conf_mat1 = None
        >>> history_confusion_matrices  = {
        >>>     'folder0': {
        >>>         # apect           epoch0      epoch1
        >>>         "training":     [conf_mat0, conf_mat1],
        >>>         'trained':      [conf_mat0, conf_mat1],
        >>>         'valid':        [conf_mat0, conf_mat1],
        >>>         'test':         [conf_mat0, conf_mat1]
        >>>     }
        >>>     'folder1': {
        >>>         # apect           epoch0      epoch1
        >>>         "training":     [conf_mat0, conf_mat1],
        >>>         'trained':      [conf_mat0, conf_mat1],
        >>>         'valid':        [conf_mat0, conf_mat1],
        >>>         'test':         [conf_mat0, conf_mat1]
        >>>     }
        >>> }
        >>>
        >>>  history_losses = {
        >>>         'folder0': {
        >>>             "training": [],
        >>>             'trained': [],
        >>>             'valid': [],
        >>>             'test': [],
        >>>             'L1': [],
        >>>             'L2': []
        >>>         }
        >>>         'folder1': {
        >>>             "training": [],
        >>>             'trained': [],
        >>>             'valid': [],
        >>>             'test': [],
        >>>             'L1': [],
        >>>             'L2': []
        >>>         }
        >>>     }
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
        self.history_evaluation_matrices = {}
        self.history_confusion_matrices = {}

        # hidden params
        self._metric_index = None

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
        self._metric_index = {'micro': (-1, 0)}
        for i, tag in enumerate(index_to_tags + ['macro', 'micro']):
            self._metric_index['%s_acc' % tag] = (i, 0)
            self._metric_index['%s_recall' % tag] = (i, 1)
            self._metric_index['%s_f1' % tag] = (i, 2)

    def set_aspects(self, *args):
        for aspect in args:
            if aspect not in ['training', 'trained', 'valid', 'test']:
                raise ValueError
        self.aspects = tuple(args)

    def set_cpu_metrics(self, *args):
        cpu_metrics = set([])
        for metric in args:
            if metric in ['macro_acc', 'macro_recall', 'macro_f1',
                          'micro_acc', 'micro_recall', 'micro_f1',
                          'micro']:
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

    def add_history(self, history_name, aspect, gpu_outputs, conf_mat=None, eval_mat=None):
        assert aspect in self.aspects

        if history_name not in self.histories:
            self.histories[history_name] = {}
        if history_name not in self.history_confusion_matrices:
            self.history_confusion_matrices[history_name] = {}
        if history_name not in self.history_evaluation_matrices:
            self.history_evaluation_matrices[history_name] = {}

        if aspect not in self.histories[history_name]:
            self.histories[history_name][aspect] = []
        if aspect not in self.history_confusion_matrices[history_name]:
            self.history_confusion_matrices[history_name][aspect] = []
        if aspect not in self.history_evaluation_matrices[history_name]:
            self.history_evaluation_matrices[history_name][aspect] = []

        self.histories[history_name][aspect].append(gpu_outputs)
        if conf_mat is None or eval_mat is None:
            pass
        else:
            self.history_confusion_matrices[history_name][aspect].append(conf_mat)
            self.history_evaluation_matrices[history_name][aspect].append(eval_mat)

    def add_training_history(self, history_name, gpu_outputs, conf_mat=None, eval_mat=None):
        self.add_history(history_name, 'training', gpu_outputs, conf_mat, eval_mat)

    def add_trained_history(self, history_name, outputs, conf_mat=None, eval_mat=None):
        self.add_history(history_name, 'trained', outputs, conf_mat, eval_mat)

    def add_validation_history(self, history_name, outputs, conf_mat=None, eval_mat=None):
        self.add_history(history_name, 'valid', outputs, conf_mat, eval_mat)

    def add_test_history(self, history_name, outputs, conf_mat=None, eval_mat=None):
        self.add_history(history_name, 'test', outputs, conf_mat, eval_mat)

    def output_epoch_evaluation(self, history_name, epoch, file, end="; "):
        # epoch
        runout = 'epoch %d%s' % (epoch, end)

        # aspects
        for aspect in self.aspects:
            gpu_outputs = self.histories[history_name][aspect]
            if aspect == 'training':
                temps = ["%s:%.4f" % (self.gpu_train_metrics[i], gpu_outputs[-1][i]) for i in
                         range(len(self.gpu_train_metrics))]
            else:
                temps = ["%s:%.4f" % (self.gpu_predict_metrics[i], gpu_outputs[-1][i]) for i in
                         range(len(self.gpu_predict_metrics))]
            if self.cpu_metrics:
                cpu_eval_outputs = self.history_evaluation_matrices[history_name][aspect][-1]
                temps += ["%s:%.4f" % (metric, cpu_eval_outputs[self._metric_index[metric]])
                          for metric in self.cpu_metrics]
            aspect_runout = '%s-[%s]%s' % (aspect, ' '.join(temps), end)
            runout += aspect_runout

        # times

        # output
        print(runout, file=file)

    def output_bests(self, history_name, file):
        pass

    def _execution(self, x_data, y_data, func_to_exe):
        # variable
        total_len = x_data.shape[0]
        nb_batches = len(y_data) // self.batch_size
        gpu_metric_outputs = []
        predictions = []

        # execution
        i = 0
        for i in range(nb_batches):
            xs = x_data[self.batch_size * i: self.batch_size * (i + 1)]
            ys = y_data[self.batch_size * i: self.batch_size * (i + 1)]
            res = func_to_exe(xs, ys)

            predictions.append(res[0])
            gpu_metric_outputs.append(list(res[1:]))

        else:
            if self.batch_size * (i + 1) < total_len:
                actual_len = total_len - self.batch_size * (i + 1)

                xs = x_data[-actual_len:]
                ys = y_data[-actual_len:]

                res = func_to_exe(xs, ys)

                predictions.append(res[0])
                gpu_metric_outputs.append(list(res[1:]))

        # cpu metrics evaluation
        if self.cpu_metrics:
            predictions = np.concatenate(predictions, axis=0)
            confusion_mat = matrix_eval.get_confusion_matrix(predictions, y_data, len(self.index_to_tag))
            evaluation_mat = matrix_eval.get_evaluation_matrix(confusion_mat)
        else:
            confusion_mat = evaluation_mat = None

        # gpu metrics evaluation
        gpu_metric_outputs = np.asarray(gpu_metric_outputs, dtype='float32')
        gpu_metric_outputs = np.mean(gpu_metric_outputs, axis=0)

        return confusion_mat, evaluation_mat, gpu_metric_outputs

    def epoch_train_execution(self, history_name, model, all_xs, all_ys):
        confusion_mat, evaluation_mat, gpu_metric_outputs = self._execution(
            all_xs, all_ys, model.train_func_for_eval)

        if 'training' in self.aspects:
            self.add_training_history(history_name, gpu_metric_outputs, confusion_mat, evaluation_mat)

    def epoch_predict_execution(self, history_name, model, all_xs, all_ys, aspect):
        if aspect in self.aspects:
            confusion_mat, evaluation_mat, gpu_metric_outputs = self._execution(
                all_xs, all_ys, model.predict_func_for_eval)
            self.add_history(history_name, aspect, gpu_metric_outputs, confusion_mat, evaluation_mat)


