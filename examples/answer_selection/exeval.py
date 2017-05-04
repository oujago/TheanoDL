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
        total_len = x_data.shape[0]
        nb_batches = len(y_data) // self.batch_size
        gpu_metric_outputs = []
        predictions = []

        # execution
        for i in range(nb_batches):
            xs = x_data[self.batch_size * i: self.batch_size * (i + 1)]
            xs = self.convert_func(xs)

            ys = y_data[self.batch_size * i: self.batch_size * (i + 1)]
            res = func_to_exe(xs[0], xs[1], ys)

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

    def _exe(self, exe_func, all_xs, all_ys, **kwargs):
        nb_samples = all_xs.shape[0] // self.batch_size
        predictions, origins, losses = [], [], []

        for i in range(nb_samples):
            xs = all_xs[self.batch_size * i: self.batch_size * (i + 1)]
            ys = all_ys[self.batch_size * i: self.batch_size * (i + 1)]

            res =exe_func(xs[0], xs[1], ys, **kwargs)

            losses.append(res[: -1])
            predictions.extend(list(res[-1]))
            origins.extend(list(ys))

        return np.array(predictions), np.array(origins), losses

    def exe_train(self, model, all_xs, all_ys, lr, **kwargs):
        return self._exe(model.train, all_xs, all_ys, lr=lr)

    def exe_predict(self, model, all_xs, all_ys):
        return self._exe(model.predict, all_xs, all_ys)

    def to_json(self):
        config = {
            "batch_size": self.batch_size,
            "convert_func": self.convert_func,
        }
        config.update(super(AnswerSelectionExe, self).to_json())
        return config


