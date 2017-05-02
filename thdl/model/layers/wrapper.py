# -*- coding:utf-8 -*-


from .base import Layer
from thdl.utils import is_iterable


class MultiInput(Layer):
    def __init__(self, *layers):
        self.layers = layers if layers else []

    def add(self, layer):
        self.layers.append(layer)

    def connect_to(self, pre_layer=None):
        for layer in self.layers:
            layer.connect_to()

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
