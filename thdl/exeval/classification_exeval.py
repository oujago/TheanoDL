# -*- coding: utf-8 -*-

import os

import matplotlib
import numpy as np

from thdl.utils.file import check_duplicate_path
from . import matrix_eval
from . import output
from .abstract import AbstractExeEval

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        self._cpu_metric_indexes = None

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
        self._cpu_metric_indexes = {'micro': (-1, 0)}
        for i, tag in enumerate(index_to_tags + ['macro', 'micro']):
            self._cpu_metric_indexes['%s_acc' % tag] = (i, 0)
            self._cpu_metric_indexes['%s_recall' % tag] = (i, 1)
            self._cpu_metric_indexes['%s_f1' % tag] = (i, 2)

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

        if self.model_chosen_metrics is None:
            self.model_chosen_metrics = self.cpu_metrics

    def set_model_chosen_metrics(self, *args):
        if type(args[0]).__name__ in ['tuple', 'list']:
            args = args[0]
        else:
            assert type(args[0]).__name__ == 'str'
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
                temps += ["%s:%.4f" % (metric, cpu_eval_outputs[self._cpu_metric_indexes[metric]])
                          for metric in self.cpu_metrics]
            aspect_runout = '%s-[%s]%s' % (aspect, ' '.join(temps), end)
            runout += aspect_runout

        # times

        # output
        print(runout, file=file)

    def format_runout_history_metric_aspect(self, confusion_mat, evaluation_mat=None, matrix_desc=None):
        """
        Output a value matrix.
        The functions for this functions are:
            1, output the value matrix;
            2, output the evaluation matrix;
            3, return the final total accuracy and F1.

        :param confusion_mat:
        :param evaluation_mat:
        :param matrix_desc: matrix description
        :param file:
        """

        output_lines = []
        split_line_indexes = []

        # blank line
        i = 0
        output_lines.append("")
        # split line
        i += 1
        output_lines.append("")
        split_line_indexes.append(i)
        # matrix description
        i += 1
        output_lines.append(matrix_desc)
        # split line
        i += 1
        output_lines.append('')
        split_line_indexes.append(i)
        # confusion matrix
        mat = matrix_eval.format_runout_matrix(confusion_mat, self.index_to_tag, self.index_to_tag)
        output_lines.extend(mat)
        i += len(mat)
        # split line
        i += 1
        output_lines.append("")
        split_line_indexes.append(i)
        # evaluation matrix
        if evaluation_mat is None:
            evaluation_mat = matrix_eval.get_evaluation_matrix(confusion_mat)
        output_lines.extend(matrix_eval.format_runout_matrix(
            evaluation_mat, self.index_to_tag + ['macro', 'micro'], ('Precision', 'Recall', 'F1')))

        # get max length
        max_len = max([len(line) for line in output_lines])

        # split_line
        split_line = '-' * (max_len + 3)
        for index in split_line_indexes:
            output_lines[index] = split_line

        # return
        return output_lines

    def output_ho_bests(self, history_name, file):
        for metric in self.model_chosen_metrics:
            best_epoch = self._get_best_epoch(history_name, metric)

            aspect_metric_all_output_lines = []
            for aspect in self.aspects:
                confusion_mat = self.history_confusion_matrices[history_name][aspect][best_epoch]
                evaluation_mat = self.history_evaluation_matrices[history_name][aspect][best_epoch]
                matrix_desc = "name: %s, metric: %s, aspect: %s" % (history_name, metric, aspect)
                output_lines = self.format_runout_history_metric_aspect(confusion_mat, evaluation_mat, matrix_desc)
                aspect_metric_all_output_lines.append(output_lines)
            output.print_runout_history_metric(aspect_metric_all_output_lines, file=file)

    def output_cv_bests(self, file):
        history_names = sorted(self.history_evaluation_matrices.keys())
        if len(history_names) == 1:
            return

        for metric in self.model_chosen_metrics:
            total_confusion_mats = {aspect: np.zeros((len(self.index_to_tag), len(self.index_to_tag)), dtype='int32')
                                    for aspect in self.aspects}
            for history_name in history_names:
                best_epoch = self._get_best_epoch(history_name, metric)
                for aspect in self.aspects:
                    total_confusion_mats[aspect] += self.history_confusion_matrices[history_name][aspect][best_epoch]

            aspect_metric_all_output_lines = []
            for aspect in self.aspects:
                confusion_mat = total_confusion_mats[aspect]
                evaluation_mat = matrix_eval.get_evaluation_matrix(confusion_mat)
                matrix_desc = "name: total, metric: %s, aspect: %s" % (metric, aspect)
                output_lines = self.format_runout_history_metric_aspect(confusion_mat, evaluation_mat, matrix_desc)
                aspect_metric_all_output_lines.append(output_lines)
            output.print_runout_history_metric(aspect_metric_all_output_lines, file=file)

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

    def _get_best_epoch(self, history_name, metric, return_value=False):
        # get aspect
        if 'valid' in self.aspects:
            aspect = 'valid'
        elif 'trained' in self.aspects:
            aspect = 'trained'
        elif 'training' in self.aspects:
            aspect = 'training'
        else:
            raise ValueError

        better_metric = ">"

        # get histories
        if metric in self.cpu_metrics:
            idx = self._cpu_metric_indexes[metric]
            histories = [matrix[idx] for matrix in self.history_evaluation_matrices[history_name][aspect]]
        else:
            if 'train' in aspect and metric in self.gpu_train_metrics:
                idx = self.gpu_train_metrics.index(metric)
                histories = [outs[idx] for outs in self.histories[history_name][aspect]]
            elif aspect == 'valid' and metric in self.gpu_predict_metrics:
                idx = self.gpu_predict_metrics.index(metric)
                histories = [outs[idx] for outs in self.histories[history_name][aspect]]
            else:
                raise ValueError

            if 'Loss' in metric or 'Regularizer' == metric:
                better_metric = '<'

        best_value, best_epoch = histories[0], 0

        # compare
        if better_metric == '>':
            for i, value in enumerate(histories):
                if value > best_value:
                    best_epoch = i
                    best_value = value
                i += 1

        elif better_metric == '<':
            for i, value in enumerate(histories):
                if value < best_value:
                    best_epoch = i
                    best_value = value
                i += 1

        else:
            raise ValueError

        # return
        if return_value:
            return best_epoch, best_value
        else:
            return best_epoch

    def plot_tag_history_evaluations(self, filename, tag_name):
        metrics = ["%s_%s" % (tag_name, a) for a in ['acc', 'recall', 'f1']]
        self.plot_cpu_metric_histories(filename, metrics, False)

    def plot_history_losses(self, filename):
        # variables
        filename = os.path.join(os.getcwd(), filename)
        history_num = len(self.histories)
        grid_row, grid_col = history_num, len(self.aspects)
        history_names = sorted(list(self.histories.keys()))

        # plots
        plt.figure(figsize=(10 * grid_col, 10 * grid_row))  # width, height
        for i, history_name in enumerate(history_names):
            for j, aspect in enumerate(self.aspects):
                aspect_value = np.array(self.histories[history_name][aspect])
                if aspect in self.aspects:
                    plt.subplot(grid_row, grid_col, grid_col * i + j + 1)
                    if aspect == 'training':
                        metrics = self.gpu_train_metrics
                    else:
                        metrics = self.gpu_predict_metrics

                    for k, metric in enumerate(metrics):
                        if 'Loss' in metric or 'Regularizer' == metric:
                            value = aspect_value[:, k]
                            plt.plot(value, label=metric)
                    plt.title("%s: %s" % (history_name, aspect))
                    plt.legend(loc='best', fontsize='medium')
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss Value")

        # save figure
        plt.savefig(check_duplicate_path(filename))
        plt.close()

    def plot_cpu_metric_histories(self, filename, metrics=None, mark_best=True):
        # if there are no cpu metrics
        if not self.cpu_metrics:
            return

        # else
        if metrics is None:
            metrics = self.cpu_metrics
        elif type(metrics).__name__ in ['tuple', 'list']:
            metrics = metrics
        else:
            raise ValueError

        # variables
        filename = os.path.join(os.getcwd(), filename)
        history_num = len(self.history_evaluation_matrices)
        grid_row, grid_col = history_num, len(self.aspects)
        history_names = sorted(self.history_evaluation_matrices.keys())

        # plots
        plt.figure(figsize=(10 * grid_col, 10 * grid_row))  # width, height
        for i, history_name in enumerate(history_names):
            aspect_values = self.history_evaluation_matrices[history_name]
            metric_best_epoch = {metric: self._get_best_epoch(history_name, metric) for metric in metrics}
            for j, aspect in enumerate(self.aspects):
                plt.subplot(grid_row, grid_col, grid_col * i + j + 1)

                for metric in metrics:
                    metric_idx = self._cpu_metric_indexes[metric]
                    metric_history = [matrix[metric_idx] for matrix in aspect_values[aspect]]
                    plt.plot(np.array(metric_history), label=metric)
                    # if mark_best and metric in self.model_chosen_metrics:
                    if mark_best:
                        best_epoch = metric_best_epoch[metric]
                        best_value = float(metric_history[best_epoch])
                        plt.annotate(s="%.2f" % best_value, xy=(best_epoch, best_value), xycoords='data',
                                     xytext=(10, 10), textcoords='offset points',
                                     arrowprops={"arrowstyle": "->",
                                                 "connectionstyle": "arc,angleA=0,armA=20,angleB=90,armB=15,rad=7"})

                plt.title("%s: %s" % (history_name, aspect))
                plt.legend(loc='best', fontsize='small')
                plt.xlabel("Epoch")
                plt.ylabel("Score")

        # save figure
        plt.savefig(check_duplicate_path(filename))
        plt.close()

    def plot_gpu_metric_histories(self, filename, metrics, mark_best=True):
        # get metrics
        assert metrics is not None
        temp_metrics = []
        for metric in metrics:
            if metric.__class__.__name__ == 'str':
                temp_metrics.append(metric)
            else:
                temp_metrics.append(metric.__class__.__name__)
        metrics = temp_metrics

        # variables
        filename = os.path.join(os.getcwd(), filename)
        history_num = len(self.histories)
        grid_row, grid_col = history_num, len(self.aspects)
        history_names = sorted(self.histories.keys())

        # plots
        plt.figure(figsize=(10 * grid_col, 10 * grid_row))  # width, height
        for i, history_name in enumerate(history_names):
            aspect_values = self.histories[history_name]
            metric_best_epoch = {metric: self._get_best_epoch(history_name, metric) for metric in metrics}
            for j, aspect in enumerate(self.aspects):
                plt.subplot(grid_row, grid_col, grid_col * i + j + 1)

                if aspect == 'training':
                    temp_metrics = self.gpu_train_metrics
                else:
                    temp_metrics = self.gpu_predict_metrics

                for k, metric in enumerate(temp_metrics):
                    metric_history = [outs[k] for outs in aspect_values[aspect]]
                    plt.plot(np.array(metric_history), label=metric)
                    # if mark_best and metric in self.model_chosen_metrics:
                    if mark_best:
                        if metric in metric_best_epoch:
                            best_epoch = metric_best_epoch[metric]
                            best_value = float(metric_history[best_epoch])
                            plt.annotate(s="%.2f" % best_value, xy=(best_epoch, best_value), xycoords='data',
                                         xytext=(10, 10), textcoords='offset points',
                                         arrowprops={"arrowstyle": "->",
                                                     "connectionstyle": "arc,angleA=0,armA=20,angleB=90,armB=15,rad=7"})

                plt.title("%s: %s" % (history_name, aspect))
                plt.legend(loc='best', fontsize='small')
                plt.xlabel("Epoch")
                plt.ylabel("Score")

        # save figure
        plt.savefig(check_duplicate_path(filename))
        plt.close()

    def plot_bests(self, filename, aspect='test'):
        # variables
        filename = os.path.join(os.getcwd(), filename)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        history_names = sorted(list(self.history_evaluation_matrices.keys()))
        grid_col, grid_row = 1, len(self.metrics_to_choose_model)

        if len(history_names) > 1:
            bar_width, opacity = .55, 0.4
            gap = int(bar_width * len(self.metrics)) + 1
            fontsize = 7
            figsize_x = 20 * grid_col
            loc = 'best'
        else:
            bar_width, opacity = .55, 0.4
            gap = round(bar_width * len(self.metrics))
            fontsize = 15
            figsize_x = 5 * grid_col
            loc = 'lower right'

        # values
        metric2bests = {}
        for metric in self.metrics_to_choose_model:
            metric2bests[metric] = []
            total_conf_mat = np.zeros((len(self.index2tag), len(self.index2tag)), dtype='int32')

            for history_name in history_names:
                best_epoch = self._get_best_epoch(history_name, metric)
                total_conf_mat += self.history_confusion_matrices[history_name][aspect][best_epoch]
                eval_mat = self.history_evaluation_matrices[history_name][aspect][best_epoch]
                metric2bests[metric].append([eval_mat[self.metric_index[metric]] for metric in self.metrics])

            if len(history_names) > 1:
                total_eval_mat = self.get_evaluation_matrix(total_conf_mat)
                metric2bests[metric].append([total_eval_mat[self.metric_index[metric]] for metric in self.metrics])

        # plots
        if len(history_names) > 1:
            history_names.append('total')
        index = np.arange(start=0, stop=gap * len(history_names), step=gap)

        plt.figure(figsize=(figsize_x, 5 * grid_row))
        for i, metric in enumerate(self.metrics_to_choose_model):
            values = np.asarray(metric2bests[metric]) * 100
            plt.subplot(grid_row, grid_col, i + 1)
            for j, best_metric in enumerate(self.metrics):
                rects = plt.bar(index + bar_width * j, values[:, j], width=bar_width, alpha=opacity, label=best_metric,
                                color=colors[j])
                for rect in rects:
                    width, height = rect.get_x() + rect.get_width() / 2, rect.get_height()
                    plt.text(width, height, '%.2f' % float(height), fontsize=fontsize, horizontalalignment='center')
            plt.xlabel('History Names')
            plt.ylabel('Scores')
            plt.title('Metric: %s, Aspect: %s' % (metric, aspect))
            plt.xticks(index + bar_width, history_names)
            plt.legend(loc=loc, fontsize='small')
            plt.tight_layout()

        # save figure
        plt.savefig(check_duplicate_path(filename))
        plt.close()
