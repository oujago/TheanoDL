# -*- coding: utf-8 -*-


class Data:

    def get_xs(self):
        raise NotImplementedError("Please implement 'get_xs' method.")

    def get_ys(self):
        raise NotImplementedError("Please implement 'get_ys' method.")

    def to_json(self):
        raise NotImplementedError("Please implement 'to_json' method.")

    @classmethod
    def from_config(cls, config):
        return cls(**config)

