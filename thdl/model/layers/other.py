# -*- coding: utf-8 -*-


from theano import tensor

from .base import Layer


class ToolBox(Layer):
    def __init__(self, tool, *args, **kwargs):
        super(ToolBox, self).__init__()

        self.tool = tool
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input):
        if self.tool == 'flatten':
            return input.flatten(self.kwargs['ndim'])

        if self.tool == 'mean':
            return tensor.mean(input, axis=self.kwargs['axis'])

        if self.tool == 'reshape':
            return tensor.reshape(input, newshape=self.kwargs['newshape'])

        if self.tool == 'dimshuffle':
            return input.dimshuffle(self.kwargs['pattern'])

        raise ValueError("Unknown utils method: %s" % self.tool)

    def __str__(self):
        return "ToolBox"

    def to_json(self):
        config = {
            'utils': self.tool,
            'args': self.args,
            'kwargs': self.kwargs
        }
        return config