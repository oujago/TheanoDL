# -*- coding: utf-8 -*-


import os
import sys

import matplotlib
import numpy as np

from thdl.utils.other import check_duplicate_path

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Evaluation:
    """
    In this class, there are some complicated attributions, like:
        
        >>> eval_mat0= None
        >>> eval_mat1 = None
        >>> history_evaluation_matrices = {
        >>>     'folder0': {
        >>>         # apect           epoch0      epoch1
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

    In this class, several useful functions are provided. Like:

        - plot
            * plot every validation's best evaluation and total bets evaluation
            * plot the trend of every metric's evaluation in every validation with the epoch goes
            * plot the trend of every tag's evaluation in every validation with the epoch goes
            * plot the loss, including L1, L2, train, test, and valid loss

        - output
            * output the confusion matrix
            * output the evaluation matrix
            * output the epoch's evaluation

        - calculate
            * calculate the confusion matrix
            * calculate the evaluation matrix
            * calculate the best epoch


    """

    def __init__(self, index2tag,
                 metrics=('macro_acc', 'macro_recall', 'macro_f1', 'micro_acc'),
                 aspects=('training', 'trained', 'valid', 'test'),
                 metrics_to_choose_model=None,
                 split_line_len=20):
        """
        The most important class to evaluate the classification task.

        :param index2tag: Please tell me what classes do we want to classify into

        :param metrics: what metrics do we use. possible values are:
                        ('macro_acc', 'macro_recall', 'macro_f1',
                         'micro_acc', 'micro_recall', 'micro_f1',
                         'micro')

        :param aspects: which datasets do we want to evaluate. possible values are:
                        ('training', 'trained', 'valid', 'test')

        :param metrics_to_choose_model: what metrics do we use to choose model
                        if it is None, then use all metrics to choose model.

        :param split_line_len:

        """
        # get all tags
        self.index2tag = index2tag

        # get total metric index
        self.metric_index = {'micro': (-1, 0)}
        for i, tag in enumerate(index2tag + ['macro', 'micro']):
            self.metric_index['%s_acc' % tag] = (i, 0)
            self.metric_index['%s_recall' % tag] = (i, 1)
            self.metric_index['%s_f1' % tag] = (i, 2)

        # get evaluation metric index
        if type(metrics).__name__ == 'str':
            metrics = (metrics,)

        i, micro, metrics = 0, False, list(metrics)
        while i < len(metrics):
            if metrics[i] not in ['macro_acc', 'macro_recall', 'macro_f1']:
                if metrics[i] in ['micro_acc', 'micro_recall', 'micro_f1', 'micro']:
                    micro = True
                    metrics.remove(metrics[i])
                else:
                    raise ValueError("Unknown metric: %s" % metrics[i])
            else:
                i += 1
        if micro:
            metrics.append('micro_acc')
        self.metrics = metrics

        # get evaluation aspects
        self.aspects = aspects

        # get model choosing metrics
        self.metrics_to_choose_model = metrics_to_choose_model or self.metrics

        # history attributions
        self.history_evaluation_matrices = {}
        self.history_confusion_matrices = {}
        self.history_losses = {}

        # split line
        self.split_line_len = split_line_len
        self.split_line = "---" * split_line_len

    def to_json(self):
        config = {
            'index2tag': self.index2tag,
            'metrics': self.metrics,
            'aspects': self.aspects,
            "metrics_to_choose_model": self.metrics_to_choose_model,
            'split_line_len': self.split_line_len
        }
        return config

    def add_history_confusion_matrix(self, history_name, aspect, confusion_mat):
        """ Add one epoch confusion matrix """
        assert aspect in self.aspects

        if history_name not in self.history_confusion_matrices:
            self.history_confusion_matrices[history_name] = {}
        if aspect not in self.history_confusion_matrices[history_name]:
            self.history_confusion_matrices[history_name][aspect] = []
        self.history_confusion_matrices[history_name][aspect].append(confusion_mat)

    def add_history_evaluation_matrix(self, history_name, aspect, evaluation_mat):
        """ Add one epoch evaluation matrix """
        assert aspect in self.aspects

        if history_name not in self.history_evaluation_matrices:
            self.history_evaluation_matrices[history_name] = {}
        if aspect not in self.history_evaluation_matrices[history_name]:
            self.history_evaluation_matrices[history_name][aspect] = []
        self.history_evaluation_matrices[history_name][aspect].append(evaluation_mat)

    def add_history_loss(self, history_name, aspect, loss):
        """ Add one epoch evaluation loss """
        assert aspect in self.aspects

        if history_name not in self.history_losses:
            self.history_losses[history_name] = {}
        if aspect == 'training':
            asp_loss_pairs = [('training', loss[0]), ('L1', loss[1]), ('L2', loss[2])]
        else:
            asp_loss_pairs = [(aspect, loss)]
        for asp, loss in asp_loss_pairs:
            if asp not in self.history_losses[history_name]:
                self.history_losses[history_name][asp] = []
            self.history_losses[history_name][asp].append(loss)

    def add_history(self, history_name, aspect, confusion_mat, evaluation_mat, loss):
        """ Add one epoch history """
        self.add_history_confusion_matrix(history_name, aspect, confusion_mat)
        self.add_history_evaluation_matrix(history_name, aspect, evaluation_mat)
        self.add_history_loss(history_name, aspect, loss)

    def output_epoch(self, history_name, epoch, end='; ', file=sys.stdout):
        """ Output the loss and the evaluations on one epoch """
        # epoch
        epoch_runout = 'epoch %d' % epoch
        runout = epoch_runout + end
        # loss
        loss_runout = ["%s:%.4f" % (key, value[epoch]) for key, value in
                       sorted(self.history_losses[history_name].items())]
        runout += "loss-[%s]; " % (" ".join(loss_runout))
        # aspects
        for aspect in self.aspects:
            matrix = self.history_evaluation_matrices[history_name][aspect][-1]
            aspect_runout = ["%s:%.4f" % (metric, matrix[self.metric_index[metric]]) for metric in self.metrics]
            runout += "%s-[%s]; " % (aspect, " ".join(aspect_runout))
        # print
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

        # blank line
        output_lines.append("")
        # split line
        output_lines.append(self.split_line)
        # matrix description
        output_lines.append(matrix_desc)
        # split line
        output_lines.append(self.split_line)
        # confusion matrix
        output_lines.extend(self.format_runout_matrix(confusion_mat, self.index2tag, self.index2tag))
        # split line
        output_lines.append(self.split_line)
        # evaluation matrix
        if evaluation_mat is None:
            evaluation_mat = self.get_evaluation_matrix(confusion_mat)
        output_lines.extend(self.format_runout_matrix(
            evaluation_mat, self.index2tag + ['macro', 'micro'], ('Precision', 'Recall', 'F1')))
        # return
        return output_lines

    def print_runout_history_metric(self, history_aspect_all_output_lines, file=sys.stdout):
        maxlen = max([len(line) for output_lines in history_aspect_all_output_lines for line in output_lines]) + 1

        for lines in zip(*history_aspect_all_output_lines):
            runout = ''
            for line in lines:
                runout += (line + " " * (maxlen - len(line)))
            print(runout, file=file)

    def output_bests(self, history_name, metrics=None, file=sys.stdout):
        metrics = metrics or self.metrics_to_choose_model
        for metric in metrics:
            best_epoch = self.get_best_epoch(history_name, metric)

            aspect_metric_all_output_lines = []
            for aspect in self.aspects:
                confusion_mat = self.history_confusion_matrices[history_name][aspect][best_epoch]
                evaluation_mat = self.history_evaluation_matrices[history_name][aspect][best_epoch]
                matrix_desc = "name: %s, metric: %s, aspect: %s" % (history_name, metric, aspect)
                output_lines = self.format_runout_history_metric_aspect(confusion_mat, evaluation_mat, matrix_desc)
                aspect_metric_all_output_lines.append(output_lines)
            self.print_runout_history_metric(aspect_metric_all_output_lines, file=file)

    def output_total_bests(self, metrics=None, file=sys.stdout):
        metrics = metrics or self.metrics_to_choose_model

        history_names = sorted(self.history_evaluation_matrices.keys())
        if len(history_names) == 1:
            return

        for metric in metrics:
            total_confusion_mats = {aspect: np.zeros((len(self.index2tag), len(self.index2tag)), dtype='int32')
                                    for aspect in self.aspects}
            for history_name in history_names:
                best_epoch = self.get_best_epoch(history_name, metric)
                for aspect in self.aspects:
                    total_confusion_mats[aspect] += self.history_confusion_matrices[history_name][aspect][best_epoch]

            aspect_metric_all_output_lines = []
            for aspect in self.aspects:
                confusion_mat = total_confusion_mats[aspect]
                evaluation_mat = self.get_evaluation_matrix(confusion_mat)
                matrix_desc = "name: total, metric: %s, aspect: %s" % (metric, aspect)
                output_lines = self.format_runout_history_metric_aspect(confusion_mat, evaluation_mat, matrix_desc)
                aspect_metric_all_output_lines.append(output_lines)
            self.print_runout_history_metric(aspect_metric_all_output_lines, file=file)

    def get_best_epoch(self, history_name, metric, return_value=False):
        best_value = 0
        best_epoch = 0
        idx = self.metric_index[metric]

        i = 0
        for matrix in self.history_evaluation_matrices[history_name]['valid']:
            if matrix[idx] > best_value:
                best_epoch = i
                best_value = matrix[idx]
            i += 1

        if return_value:
            return best_epoch, best_value
        else:
            return best_epoch

    def plot_history_losses(self, filename):
        # variables
        filename = os.path.join(os.getcwd(), filename)
        history_num = len(self.history_losses)
        grid_row, grid_col = history_num, 1
        history_names = sorted(list(self.history_losses.keys()))

        # plots
        plt.figure(figsize=(10 * grid_col, 10 * grid_row))  # width, height
        for i, history_name in enumerate(history_names):
            aspect_values = self.history_losses[history_name]
            plt.subplot(grid_row, grid_col, i + 1)
            for aspect, value in aspect_values.items():
                value = np.array(value)
                if aspect in self.aspects:
                    plt.plot(value, label=aspect)
                elif aspect in ['L1', 'L2']:
                    if np.max(value) > 0.:
                        plt.plot(value, label=aspect)
            plt.title("%s: loss" % history_name)
            plt.legend(loc='best', fontsize='medium')
            plt.xlabel("Epoch")
            plt.ylabel("Loss Value")

        # save figure
        plt.savefig(check_duplicate_path(filename))
        plt.close()

    def plot_history_evaluations(self, filename, metrics=None, mark_best=False):
        # metrics
        if metrics is None:
            metrics = self.metrics
        elif type(metrics).__name__ in ['tuple', 'list']:
            metrics = metrics
        elif type(metrics).__name__ == 'str' and metrics in self.index2tag:
            metrics = ["%s_%s" % (metrics, a) for a in ['acc', 'recall', 'f1']]
        else:
            raise ValueError("")

        # variables
        filename = os.path.join(os.getcwd(), filename)
        history_num = len(self.history_evaluation_matrices)
        grid_row, grid_col = history_num, len(self.aspects)
        history_names = sorted(self.history_evaluation_matrices.keys())

        # plots
        plt.figure(figsize=(10 * grid_col, 10 * grid_row))  # width, height
        for i, history_name in enumerate(history_names):
            aspect_values = self.history_evaluation_matrices[history_name]
            metric_best_epoch = {metric: self.get_best_epoch(history_name, metric) for metric in metrics}
            for j, aspect in enumerate(self.aspects):
                plt.subplot(grid_row, grid_col, grid_col * i + j + 1)

                for metric in metrics:
                    metric_idx = self.metric_index[metric]
                    metric_history = [matrix[metric_idx] for matrix in aspect_values[aspect]]
                    plt.plot(np.array(metric_history), label=metric)
                    if mark_best and metric in self.metrics_to_choose_model:
                        best_epoch = metric_best_epoch[metric]
                        best_value = metric_history[best_epoch]
                        plt.annotate(s="%.2f" % float(best_value), xy=(best_epoch, float(best_value)), xycoords='data',
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
                best_epoch = self.get_best_epoch(history_name, metric)
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

