# -*- coding: utf-8 -*-

from theano import tensor

from .base import Layer


class Bidirectional(Layer):
    def __init__(self, rnn, merge_mode='concat', **rnn_params):
        super(Bidirectional, self).__init__()

        # parameters
        self.merge_mode = merge_mode
        self.rnn_params = rnn_params
        self.forward_rnn = rnn(backward=False, **rnn_params)
        self.backward_rnn = rnn(backward=True, **rnn_params)

    def connect_to(self, pre_layer=None):
        self.forward_rnn.connect_to(pre_layer)
        self.backward_rnn.connect_to(pre_layer)

        output_shape = self.forward_rnn.output_shape

        if self.merge_mode == 'concat':
            self.output_shape = output_shape[-1:] + (output_shape[-1] * 2)
        else:
            raise ValueError("Unknown merge mode: %s" % self.merge_mode)

    def forward(self, input, **kwargs):
        forward_res = self.forward_rnn(input)
        backward_res = self.backward_rnn(input)

        if self.merge_mode == 'concat':
            return tensor.concatenate([forward_res, backward_res], axis=-1)

    def to_json(self):
        rnn_params = self.forward_rnn.to_json()
        rnn_params.pop('backward')

        config = {
            'rnn_params': rnn_params,
            'merge_mode': self.merge_mode,
            'rnn': str(self.forward_rnn),
        }

        return config

    @property
    def params(self):
        return self.forward_rnn.params + self.backward_rnn.params

    @property
    def regularizers(self):
        return self.forward_rnn.regularizers + self.backward_rnn.regularizers
