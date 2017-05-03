# -*- coding: utf-8 -*-


class ThdlObj(object):
    def to_json(self):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    def from_json(cls, config=None):
        if config:
            return cls(**config)
        else:
            return cls()
