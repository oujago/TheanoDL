# -*- coding: utf-8 -*-

import numpy as np

from thdl.utils.random import get_rng
from thdl.utils import type
from .abstract import AbstractData
import random


class Data(AbstractData):
    def __init__(self, shuffle=True, shuffle_seed=None, index_to_tag=None):
        """
        
        :param shuffle: When provide the training data, 
                        if shuffle == True, the shuffle the total training data. 
        :param shuffle_seed: If provide, then use shuffle_seed to get numpy.random.RandomState instance.
                            Else, randomly get the shuffle_seed
        :param index_to_tag: instance of dict object, to provide the relationship between indexes and tags.
        """
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.index_to_tag = index_to_tag

        if shuffle_seed is None:
            shuffle_seed = get_rng().randint(1000, 1000000)
        self.shuffle_rng = np.random.RandomState(seed=shuffle_seed)

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
            s = self.shuffle_rng.randint(1000, 100000000)
            if type(xs) == 'list':
                random.seed(s)
                random.shuffle(xs)
                random.seed(s)
                random.shuffle(ys)
            elif type(xs) == 'ndarray':
                np.random.seed(s)  # definitely important
                np.random.shuffle(xs)
                np.random.seed(s)  # definitely important
                np.random.shuffle(ys)
            else:
                raise ValueError
