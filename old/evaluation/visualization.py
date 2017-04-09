# -*- coding: utf-8 -*-

import os

import matplotlib
import numpy as np

from thdl.utils.file import check_duplicate_path

matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
