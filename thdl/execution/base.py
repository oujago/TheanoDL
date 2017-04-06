# -*- coding: utf-8 -*-


import numpy as np
from thdl.base import ThdlObj


class AbstractExecution(ThdlObj):
    batch_size = None
    epochs = None

    def train_execution(self, *args, **kwargs):
        raise NotImplementedError

    def predict_execution(self, *args, **kwargs):
        raise NotImplementedError

    def to_json(self, *args, **kwargs):
        raise NotImplementedError


class Execution(AbstractExecution):
    def __init__(self, batch_size, epochs=50):
        self.batch_size = batch_size
        self.epochs = epochs

    def _execution(self, x_data, y_data, func_to_exe):
        total_len = x_data.shape[0]

        nb_samples = len(y_data) // self.batch_size
        predictions, other_outputs = [], []

        i = 0
        for i in range(nb_samples):
            xs = x_data[self.batch_size * i: self.batch_size * (i + 1)]
            ys = y_data[self.batch_size * i: self.batch_size * (i + 1)]
            res = func_to_exe(xs, ys)

            predictions.append(res[0])
            other_outputs.extend(list(res[1:]))

        else:
            if self.batch_size * (i + 1) < total_len:
                actual_len = total_len - self.batch_size * (i + 1)

                xs = x_data[-actual_len:]
                ys = y_data[-actual_len:]

                res = func_to_exe(xs, ys)

                predictions.append(res[0])
                other_outputs.extend(list(res[1:]))

        predictions = np.concatenate(predictions, axis=0)
        assert predictions.shape == total_len
        return predictions, other_outputs

    def train_execution(self, model, all_xs, all_ys):
        return self._execution(all_xs, all_ys, model.func_train)

    def predict_execution(self, model, all_xs, all_ys):
        return self._execution(all_xs, all_ys, model.func_predict)

    def to_json(self):
        config = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }
        return config

