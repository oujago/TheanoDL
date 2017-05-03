# -*- coding:utf-8 -*-

from theano import tensor
from thdl.model.layers import Layer


class ASLayer(Layer):
    def __init__(self, embedding_layer=None, q1_conv_layer=None, q2_conv_layer=None):
        self.embedding_layer = embedding_layer
        self.q1_conv_layer = q1_conv_layer
        self.q2_conv_layer = q2_conv_layer

    def add_embedding_layer(self, embedding_layer):
        self.embedding_layer = embedding_layer

    def add_conv_layers(self, q1_conv_layer, q2_conv_layer):
        self.q1_conv_layer = q1_conv_layer
        self.q2_conv_layer = q2_conv_layer

    def connect_to(self, pre_layer=None):
        # connect to
        self.embedding_layer.connect_to(pre_layer)
        self.q1_conv_layer.connect_to(self.embedding_layer)
        self.q2_conv_layer.connect_to(self.embedding_layer)

        # output shape
        assert self.q1_conv_layer.output_shape[0] == self.q2_conv_layer.output_shape[0]
        nb_batch =  self.q1_conv_layer.output_shape[0]
        length =  self.q1_conv_layer.output_shape[1] +  self.q2_conv_layer.output_shape[1]
        self.output_shape = (nb_batch, length)

    def forward(self, inputs, **kwargs):
        q1_embed = self.embedding_layer.forward(inputs[0])
        q1_conv = self.q1_conv_layer.forward(q1_embed)

        q2_embed = self.embedding_layer.forward(inputs[1])
        q2_conv = self.q1_conv_layer.forward(q2_embed)

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

