# -*- coding: utf-8 -*-


import numpy as np
from thdl.utils import random


class Data:
    def __init__(self, shuffle=True, shuffle_seed=None):
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        if shuffle_seed:
            self.shuffle_rng = np.random.RandomState(seed=shuffle_seed)
        else:
            self.shuffle_rng = random.get_rng().randint(1000, 10000000000)

    def get_xs(self):
        raise NotImplementedError("Please implement 'get_xs' method.")

    def get_ys(self):
        raise NotImplementedError("Please implement 'get_ys' method.")

    def to_json(self):
        config = {
            'shuffle': self.shuffle,
            "shuffle_seed": self.shuffle_seed,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def shuffle_data(self, xs, ys):
        if self.shuffle:
            s = self.shuffle_rng.randint(0, 99999)
            np.random.seed(s)  # definitely important
            np.random.shuffle(xs)
            np.random.seed(s)  # definitely important
            np.random.shuffle(ys)
