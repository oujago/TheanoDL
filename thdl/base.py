# -*- coding: utf-8 -*-


class ThObject(object):
    def to_json(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_json(cls, config=None):
        if config:
            return cls(**config)
        else:
            return cls()
