# -*- coding:utf-8 -*-


from thdl.base import ThdlObj


class AbstractLayer(ThdlObj):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, **kwargs):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError

    @property
    def regularizers(self):
        raise NotImplementedError

    @property
    def updates(self):
        raise NotImplementedError
