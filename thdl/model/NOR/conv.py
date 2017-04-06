# -*- coding: utf-8 -*-

from thdl.utils.variables import get_shared
from .base import SubNet


class Conv(SubNet):
    def __init__(self, rng, image_shape, filter_shape,
                 activation='relu', init='glorot_uniform'):
        super(Conv, self).__init__()

        raise NotImplementedError('')

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init

        # variables
        self.W = get_shared(rng, filter_shape, init)
        self.b = get_shared(rng, (filter_shape[0],), 'zero')

        # params
        self.train_params.extend([self.W, self.b])
        self.reg_params.extend([self.W])

    def __call__(self, input, *pre_h):
        # output
        conv_out = conv.conv2d(input=input, filters=self.W,
                               image_shape=self.image_shape,
                               filter_shape=self.filter_shape)
        conv_out = conv_out.flatten(2)
        output = self.act_func(conv_out + self.b)
        return output

    def __str__(self):
        return 'ConvSubNet'

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_shape': self.filter_shape,
            'activation': self.act_name,
            'init': self.init,
        }
        return config
