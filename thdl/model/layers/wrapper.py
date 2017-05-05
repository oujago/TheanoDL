# -*- coding:utf-8 -*-


from theano import tensor

from .base import Layer
from thdl.utils import is_iterable


class MultiInput(Layer):
    def __init__(self, *layers):
        if len(layers) == 0:
            self.layers = []
        else:
            self.layers = list(layers)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs, **kwargs):
        outs = []
        assert len(inputs) > 1
        if len(inputs) == len(self.layers):
            for i in range(len(self.layers)):
                outs.append(self.layers[i].forward(inputs[0]))
        elif len(self.layers) == 1:
            for input in inputs:
                outs.append(self.layers[0].forward(input))
        else:
            raise ValueError("Length do not match.")
        return outs

    def to_json(self):
        config = {}
        for i in range(len(self.layers)):
            config["layer-{}".format(i)] = self.layers[i].to_json()
        return config

    @property
    def params(self):
        all_params = []

        for layer in self.layers:
            layer_prams = layer.params
            if is_iterable(layer_prams):
                all_params.extend(layer_prams)
            else:
                all_params.append(layer_prams)

        return all_params

    @property
    def regularizers(self):
        all_regularizers = []
        for layer in self.layers:
            all_regularizers.extend(layer.regularizers)

        return all_regularizers



class Bidirectional(Layer):
    def __init__(self, rnn, merge_mode='concat', **rnn_params):
        super(Bidirectional, self).__init__()

        # parameters
        self.merge_mode = merge_mode
        self.rnn_params = rnn_params
        self.forward_rnn = rnn(backward=False, **rnn_params)
        self.backward_rnn = rnn(backward=True, **rnn_params)

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
