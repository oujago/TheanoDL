# -*- coding: utf-8 -*-


import numpy as np
from thdl.base import ThObject


class AbstractExecution(ThObject):
    def train_execution(self, *args, **kwargs):
        raise NotImplementedError

    def predict_execution(self, *args, **kwargs):
        raise NotImplementedError

    def to_json(self, *args, **kwargs):
        raise NotImplementedError


class Execution:
    def __init__(self, batch_size, epochs=50, shuffle=True, shuffle_seed=12345, **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.kwargs = kwargs

    def exe_train(self, model, all_xs, all_ys):
        total_len = len(all_xs)

        nb_samples = total_len // self.batch_size
        predictions, origins, losses = [], [], []
        for i in range(nb_samples):
            xs = all_xs[self.batch_size * i: self.batch_size * (i + 1)]
            ys = all_ys[self.batch_size * i: self.batch_size * (i + 1)]

            res = model.train(np.asarray(xs, dtype='int32'), np.asarray(ys, dtype='int32'), lr)

            losses.append(res[: -1])
            predictions.extend(list(res[-1]))
            origins.extend(list(ys))

        else:
            # this version is important, because it is training on the tails of the datasets
            if self.batch_size * (i + 1) < total_len:
                actual_len = total_len - self.batch_size * (i + 1)
                xs = all_xs[-self.batch_size:]
                ys = all_ys[-self.batch_size:]

                res = model.train(np.asarray(xs, dtype='int32'), np.asarray(ys, dtype='int32'), lr)

                losses.append(res[:-1])
                predictions.extend(list(res[-1][-actual_len:]))
                origins.extend(list(ys[-actual_len:]))

        return np.array(predictions), np.array(origins), losses

    def exe_predict(self, model, all_xs, all_ys):
        total_len = len(all_xs)

        nb_samples = len(all_xs) // self.batch_size
        predictions, origins, losses = [], [], []
        for i in range(nb_samples):
            xs = all_xs[self.batch_size * i: self.batch_size * (i + 1)]
            ys = all_ys[self.batch_size * i: self.batch_size * (i + 1)]
            res = model.predict(np.asarray(xs, dtype='int32'), np.asarray(ys, dtype='int32'))

            losses.append(res[0])
            predictions.extend(list(res[-1]))
            origins.extend(list(ys))

        else:
            if self.batch_size * (i + 1) < total_len:
                actual_len = total_len - self.batch_size * (i + 1)

                xs = all_xs[-self.batch_size:]
                ys = all_ys[-self.batch_size:]

                res = model.predict(np.asarray(xs, dtype='int32'), np.asarray(ys, dtype='int32'))

                losses.append(res[0])
                predictions.extend(list(res[-1][-actual_len:]))
                origins.extend(list(ys[-actual_len:]))

        assert len(origins) == len(all_ys)
        return np.array(predictions), np.array(origins), losses

    def to_json(self):
        config = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lr': self.lr,
            'decay': self.decay,
            'shuffle': self.shuffle,
            'shuffle_seed': self.shuffle_seed,
            'kwargs': self.kwargs
        }
        return config

