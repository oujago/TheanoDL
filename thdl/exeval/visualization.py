# -*- coding: utf-8 -*-

import os

import matplotlib
import numpy as np

from thdl.utils.file import check_duplicate_path

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_histories(self, histories, filename):
    assert isinstance(histories, dict)

    # variables
    filename = os.path.join(os.getcwd(), filename)
    history_names = sorted(histories.keys())
    history_num = len(histories)
    assert history_num > 0
    grid_row, grid_col = history_num, len(histories[history_names[0]])

    # plots
    plt.figure(figsize=(10 * grid_col, 10 * grid_row))  # width, height
    for i, history_name in enumerate(history_names):
        aspect_values = histories[history_name]
        for j, aspect in enumerate(self.aspects):
            plt.subplot(grid_row, grid_col, grid_col * i + j + 1)

            if aspect == 'training':
                metrics = self.gpu_train_metrics
            else:
                metrics = self.gpu_predict_metrics

            all_values = np.asarray(aspect_values[aspect])

            for k, metric in enumerate(metrics):
                metric_history = all_values[:, k]
                plt.plot(np.array(metric_history), label=metric)

            plt.title("%s: %s" % (history_name, aspect))
            plt.legend(loc='best', fontsize='small')
            plt.xlabel("Epoch")
            plt.ylabel("Score")

    # save figure
    plt.savefig(check_duplicate_path(filename))
    plt.close()

