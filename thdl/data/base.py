# -*- coding: utf-8 -*-


import numpy as np

from thdl.base import ThdlObj
from thdl.utils import random


class AbstractData(ThdlObj):
    index_to_tag = None

    def to_json(self):
        raise NotImplementedError

    def get_train_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_valid_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_test_data(self, *args, **kwargs):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError


class Data(AbstractData):
    def __init__(self, shuffle=True, shuffle_seed=None, index_to_tag=None):
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.index_to_tag = index_to_tag

        if shuffle_seed:
            self.shuffle_rng = np.random.RandomState(seed=shuffle_seed)
        else:
            self.shuffle_rng = random.get_rng().randint(1000, 10000000000)

    def to_json(self):
        config = {
            'shuffle': self.shuffle,
            "shuffle_seed": self.shuffle_seed,
        }
        return config

    def get_valid_data(self):
        return None, None

    def get_test_data(self):
        return None, None

    def shuffle_data(self, xs, ys):
        if self.shuffle:
            s = self.shuffle_rng.randint(1000, 9999999999)
            np.random.seed(s)  # definitely important
            np.random.shuffle(xs)
            np.random.seed(s)  # definitely important
            np.random.shuffle(ys)
