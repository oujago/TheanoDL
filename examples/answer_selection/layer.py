# -*- coding:utf-8 -*-

from theano import tensor

import thdl.model
from thdl.model.layers import Layer


class ASLayer(Layer):
    def __init__(self, embedding_layer=None, q1_conv_layer=None, q2_conv_layer=None):
        self.embedding_layer = embedding_layer
        self.q1_conv_layer = q1_conv_layer
        self.q2_conv_layer = q2_conv_layer
        self.q1_dimshufle = thdl.model.layers.Dimshuffle((0, 'x', 1, 2))
        self.q2_dimshufle = thdl.model.layers.Dimshuffle((0, 'x', 1, 2))

    def forward(self, inputs, **kwargs):
        q1_embed = self.embedding_layer.forward(inputs[0])
        q1_embed_shuffle = self.q1_dimshufle.forward(q1_embed)
        q1_conv = self.q1_conv_layer.forward(q1_embed_shuffle)

        q2_embed = self.embedding_layer.forward(inputs[1])
        q2_embed_shuffle = self.q2_dimshufle.forward(q2_embed)
        q2_conv = self.q1_conv_layer.forward(q2_embed_shuffle)

        return tensor.concatenate([q1_conv, q2_conv], axis=1)

    def to_json(self):
        config = {
            'embedding_layer': self.embedding_layer.to_json(),
            "q1_conv_layer": self.q1_conv_layer.to_json(),
            "q2_conv_layer": self.q2_conv_layer.to_json(),
        }
        return config

    @property
    def params(self):
        return self.embedding_layer.params + self.q1_conv_layer.params + self.q2_conv_layer.params

    @property
    def regularizers(self):
        return self.embedding_layer.regularizers + self.q1_conv_layer.regularizers + self.q2_conv_layer.regularizers

    @property
    def updates(self):
        ups = super(ASLayer, self).updates
        ups.update(self.q1_conv_layer.updates)
        ups.update(self.q2_conv_layer.updates)
        return ups