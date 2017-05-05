# -*- coding:utf-8 -*-


from thdl.exeval import matrix_eval
import numpy as np
from thdl.exeval import ClassifyExeEval


class AnswerSelectionExe(ClassifyExeEval):
    def __init__(self, batch_size, convert_func=None, **kwargs):
        super(AnswerSelectionExe, self).__init__(batch_size, **kwargs)
        self.convert_func = convert_func

    def dock_convert_func(self, convert_func):
        self.convert_func = convert_func

    def _execution(self, x_data, y_data, func_to_exe):
        # variable
        nb_batches = len(y_data) // self.batch_size
        gpu_metric_outputs = []
        predictions = []

        # execution
        for i in range(nb_batches):
            xs = x_data[self.batch_size * i: self.batch_size * (i + 1)]
            xs = self.convert_func(xs)

            ys = y_data[self.batch_size * i: self.batch_size * (i + 1)]
            res = func_to_exe(xs[:, 0, :], xs[:, 1, :], ys)

            predictions.append(res[0])
            gpu_metric_outputs.append(list(res[1:]))

        return gpu_metric_outputs, predictions

    def to_json(self):
        config = {
            "batch_size": self.batch_size,
            "convert_func": self.convert_func,
        }
        config.update(super(AnswerSelectionExe, self).to_json())
        return config


