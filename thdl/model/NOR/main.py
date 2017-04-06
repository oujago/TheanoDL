# -*- coding: utf-8 -*-

import numpy as np
from theano import scan
from theano import tensor
from theano.tensor.signal.pool import pool_2d

from thdl.utils.variables import dtype
from ..layers import Layer


class NOR(Layer):
    def __init__(self, inner_merge='max', attention_cls=None, backward=False, return_sequence=True, **kwargs):
        """
        :param inner_merge:
        :param attention_cls:
        :param backward:
        """
        super(NOR, self).__init__()

        # parameters
        self.inner_merge = inner_merge
        self.attention_cls = attention_cls
        self.backward = backward
        self.return_sequence = return_sequence
        if self.inner_merge == 'attention':
            assert attention_cls is not None

        # variables
        self.paths = []
        self.outputs_info_len = 0
        self._ensure_check = False

    def add_layer(self, path_idx, layer):
        # check path number
        if path_idx == len(self.paths):
            self.paths.append([])

        # add path layer
        self.paths[path_idx].append(layer)

    def add_path(self, path):
        self.paths.append(path)

    def step(self, *args):
        """
        :param args:
            args = batch_xs + pre_hs

            pre_hs:
            [
                pre_layer0_hs0, pre_layer0_hs1, pre_layer0_hs2,
                pre_layer1_hs0, pre_layer1_hs1, pre_layer1_hs2,
                pre_layer2_hs0, pre_layer2_hs1, pre_layer2_hs2,
            ]

        :return:
            [
                layer0_hs0, layer0_hs1, layer0_hs2,
                layer1_hs0, layer1_hs1, layer1_hs2,
                layer2_hs0, layer2_hs1, layer2_hs2,
                res
            ]
        """
        # variables
        outputs_info = []
        xs = args[0]
        pre_outputs_info = args[1:]

        # layer
        i = 0
        paths_output = []
        for path in self.paths:
            input = xs
            for j, layer in enumerate(path):
                # work start
                pre_hs = pre_outputs_info[i: i + layer.outputs_info_len]
                i += layer.outputs_info_len

                # work
                outputs = layer(input, *pre_hs)

                # work end
                if str(layer) in ['BNSubNet', 'MLPSubNet']:
                    input = outputs
                else:
                    input = outputs[0]
                    outputs_info.extend(outputs)

            paths_output.append(input)

        if self.inner_merge == 'concat':
            e = tensor.concatenate(paths_output, axis=1)
            return outputs_info + [e]

        hs = tensor.concatenate([h.dimshuffle(0, 'x', 1) for h in paths_output], axis=1)
        if self.inner_merge == 'max':
            e = pool_2d(input=hs, ws=(len(self.paths), 1), ignore_border=True, mode='max').flatten(2)

        elif self.inner_merge == 'mean':
            e = pool_2d(input=hs, ws=(len(self.paths), 1), ignore_border=True, mode='average_exc_pad').flatten(2)

        # elif self.inner_merge == 'sum':
        #     e = pool_2d(input=hs, ws=(len(self.paths), 1), ignore_border=True, mode='sum').flatten(2)

        elif self.inner_merge == 'attention':
            # get attention
            e = self.attention_cls(hs)

        else:
            raise ValueError("")

        if len(outputs_info) != 0:
            return outputs_info + [e]
        else:
            return e

    def check(self):
        for path in self.paths:
            for layer in path:
                self.outputs_info_len += layer.outputs_info_len
                self.train_params.extend(layer.train_params)
                self.reg_params.extend(layer.reg_params)
                self.updates.update(layer.updates)

        if self.inner_merge == 'attention':
            self.train_params.extend(self.attention_cls.train_params)
            self.reg_params.extend(self.attention_cls.reg_params)
            self.updates.update(self.attention_cls.updates)

        self._ensure_check = True

    def __str__(self):
        return "EnNet_V1"

    def __call__(self, input):
        if self._ensure_check is False:
            raise ValueError("Please check before call.")

        # input
        if input.ndim != 3:
            raise ValueError("Unknown input dimension: %d" % input.ndim)
        input = input.dimshuffle(1, 0, 2)
        if self.backward:
            input = input[::-1]

        # outputs_info
        outputs_info = []
        for path in self.paths:
            for layer in path:
                for i in range(layer.outputs_info_len):
                    outputs_info.append(tensor.alloc(np.cast[dtype](0.), input.shape[1], layer.n_out))
        outputs_info += [None]

        # scan
        res, _ = scan(fn=self.step, sequences=input, outputs_info=outputs_info)

        # return
        if self.outputs_info_len == 0:
            return_res = res
        else:
            return_res = res[-1]
        assert return_res.ndim == 3
        if self.return_sequence:
            if self.backward:
                return_res = return_res[::-1]
            return return_res.dimshuffle(1, 0, 2)
        else:
            return return_res[-1]

    def to_json(self):
        config = {
            'inner_merge': self.inner_merge,
            'backward': self.backward,
            'return_sequence': self.return_sequence,
        }
        for i, path in enumerate(self.paths):
            path_name = 'path-%d' % i
            config[path_name] = []
            for j, layer in enumerate(path):
                layer_json = {str(layer): layer.to_json()}
                config[path_name].append(layer_json)

        if self.inner_merge == 'attention':
            config['attention'] = self.attention_cls.to_json()

        return config
