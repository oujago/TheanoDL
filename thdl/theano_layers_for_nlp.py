# -*- coding: utf-8 -*-

"""
@author: ChaoMing (oujago.github.io)

@date: Created on 2016/12/18

@notes:
    
"""

import numpy as np
from thdl.theano_layers import Layer
from theano import scan
from theano import tensor
from theano.tensor.nnet import conv
from theano.tensor.signal.pool import pool_2d

from .theano_tools import dtype
from .theano_tools import get_activation
from .theano_tools import get_gru_variables
from .theano_tools import get_lstm_variables
from .theano_tools import get_mgu_variables
from .theano_tools import get_rnn_variables
from .theano_tools import get_shared
from .theano_layers import Conv2D
from .theano_layers import Pool2D

dot = tensor.dot


class NLPConvPooling(Layer):
    """
    Related Paper:
        Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    """

    def __init__(self, rng,
                 image_shape, filter_heights, filter_width, filter_nums,
                 activation='relu', init='glorot_uniform',
                 padding=(0, 0), ignore_border=True, mode='max'):
        super(NLPConvPooling, self).__init__()

        self.image_shape = image_shape
        self.filter_heights = filter_heights
        self.filter_width = filter_width
        self.filter_nums = filter_nums if isinstance(filter_nums, (list, tuple)) \
            else [filter_nums] * len(filter_heights)
        self.act_name = activation
        self.init = init
        self.padding = padding
        self.ignore_border = ignore_border
        self.mode = mode

        self.all_conv = []
        self.all_pooling = []
        for i in range(len(filter_heights)):
            filter_shape = (self.filter_nums[i], image_shape[1], filter_heights[i], filter_width)
            self.all_conv.append(Conv2D(rng, image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_conv[i].train_params)
            self.reg_params.extend(self.all_conv[i].reg_params)
            self.updates.update(self.all_conv[i].updates)

            pool_size = (image_shape[2] - filter_heights[i] + 1, 1)
            self.all_pooling.append(Pool2D(pool_size, padding, ignore_border, mode))
            self.train_params.extend(self.all_pooling[i].train_params)
            self.reg_params.extend(self.all_pooling[i].reg_params)
            self.updates.update(self.all_pooling[i].updates)

    def __str__(self):
        return 'NLPConvPooling'

    def __call__(self, input):
        outputs = []
        for i in range(len(self.filter_heights)):
            conv_out = self.all_conv[i](input)
            pool_out = self.all_pooling[i](conv_out)
            outputs.append(pool_out.flatten(2))

        return tensor.concatenate(outputs, axis=1)

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_heights': self.filter_heights,
            'filter_width': self.filter_width,
            'filter_nums': self.filter_nums,
            'activation': self.act_name,
            'init': self.init,
            'padding': self.padding,
            'ignore_border': self.ignore_border,
            'mode': self.mode,
        }
        return config


class NLPConv(Layer):
    def __init__(self, rng,
                 image_shape, filter_heights, filter_width, filter_nums,
                 activation='relu', init='glorot_uniform'):
        super(NLPConv, self).__init__()

        self.image_shape = image_shape
        self.filter_heights = filter_heights
        self.filter_width = filter_width
        self.filter_nums = filter_nums if isinstance(filter_nums, (list, tuple)) \
            else [filter_nums] * len(filter_heights)
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.init = init

        self.truncat_len = image_shape[2] - max(filter_heights) + 1

        self.all_conv = []
        for i in range(len(filter_heights)):
            filter_shape = (self.filter_nums[i], image_shape[1], filter_heights[i], filter_width)
            self.all_conv.append(Conv2D(rng, image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_conv[i].train_params)
            self.reg_params.extend(self.all_conv[i].reg_params)
            self.updates.update(self.all_conv[i].updates)

    def __str__(self):
        return "NLPConv"

    def __call__(self, input):
        # if input.ndim == 3:
        #     input = input.dimshuffle(0, 'x', 1, 2)

        outputs = []
        for i in range(len(self.filter_heights)):
            # if self.filter_heights[i] > 1:
            #     zero = tensor.zeros(input.shape[:2] + (self.filter_heights[i]-1,) + input.shape[3:])
            #     conv_out = self.all_conv[i](tensor.concatenate([zero, input], axis=2))
            # else:
            #     conv_out = self.all_conv[i](input)
            conv_out = self.all_conv[i](input)
            output = conv_out.flatten(3)[:, :, -self.truncat_len:]
            output = output.dimshuffle(0, 2, 1)
            outputs.append(output)

        return tensor.concatenate(outputs, axis=-1)

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_heights': self.filter_heights,
            'filter_width': self.filter_width,
            'filter_nums': self.filter_nums,
            'activation': self.act_name,
            'init': self.init,
        }
        return config


class NLPAsymConv(Layer):
    def __init__(self, rng,
                 image_shape, filter_heights, filter_width, filter_nums,
                 activation='relu', init='glorot_uniform',
                 concat='truncate'):
        super(NLPAsymConv, self).__init__()

        self.image_shape = image_shape
        self.filter_heights = filter_heights
        self.filter_width = filter_width
        self.filter_nums = filter_nums if isinstance(filter_nums, (list, tuple)) \
            else [filter_nums] * len(filter_heights)
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.init = init
        self.concat = concat

        self.all_row_conv = []
        self.all_col_conv = []
        for i in range(len(filter_heights)):
            filter_shape = (self.filter_nums[i], image_shape[1], 1, filter_width)
            self.all_row_conv.append(Conv2D(rng, image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_row_conv[i].train_params)
            self.reg_params.extend(self.all_row_conv[i].reg_params)
            self.updates.update(self.all_row_conv[i].updates)

            col_image_shape = (image_shape[0], self.filter_nums[i], image_shape[2], 1)
            filter_shape = (self.filter_nums[i], self.filter_nums[i], filter_heights[i], 1)
            self.all_col_conv.append(Conv2D(rng, col_image_shape, filter_shape, activation, init))
            self.train_params.extend(self.all_col_conv[i].train_params)
            self.reg_params.extend(self.all_col_conv[i].reg_params)
            self.updates.update(self.all_col_conv[i].updates)

        if concat == 'truncate':
            self.truncat_len = image_shape[2] - max(filter_heights) + 1
        elif concat == '':
            pass
        else:
            raise ValueError("Unknown concatenate method: %s" % concat)

    def __str__(self):
        return "NLPAsymConv"

    def __call__(self, input):
        outputs = []
        for i in range(len(self.filter_heights)):
            row_conv_out = self.all_row_conv[i](input)
            col_conv_out = self.all_col_conv[i](row_conv_out)

            output = col_conv_out.flatten(3)[:, :, :self.truncat_len]
            output = output.dimshuffle(0, 2, 1)
            outputs.append(output)

        return tensor.concatenate(outputs, axis=-1)

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_heights': self.filter_heights,
            'filter_width': self.filter_width,
            'filter_nums': self.filter_nums,
            'activation': self.act_name,
            'init': self.init,
            'concat': self.concat,
        }
        return config


class SubNet(Layer):
    def __init__(self):
        super(SubNet, self).__init__()

        self.outputs_info_len = 0

    def __str__(self):
        return 'SubNet'

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("")

    def to_json(self):
        raise NotImplementedError("")


class RNN(SubNet):
    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', bias=True):
        super(RNN, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init
        self.inner_init = inner_init
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.bias = bias

        # variables
        self.x2o, self.o2o, self.b = get_rnn_variables(rng, n_in, n_out, init, inner_init)
        self.outputs_info_len = 1

        # params
        self.train_params.extend([self.x2o, self.o2o, ])
        if self.bias: self.train_params.append(self.b)
        self.reg_params.extend([self.x2o, self.o2o])

    def __call__(self, input, *pre_hs):
        out = dot(input, self.x2o) + dot(pre_hs[0], self.o2o)
        if self.bias:
            out = self.act_func(out + self.b)
        else:
            out = self.act_func(out)
        return [out,]

    def __str__(self):
        return "RNNSubNet"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'bias': self.bias,
        }
        return config


class GRU(SubNet):
    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', gate_activation='sigmoid',
                 bias=True):
        super(GRU, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init
        self.inner_init = inner_init
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.gae_act_name = gate_activation
        self.gate_act_func = get_activation(gate_activation)
        self.bias = bias
        self.outputs_info_len = 1

        # variables
        self.r_x2h_W, self.r_h2h_W, self.r_h_b, \
        self.z_x2h_W, self.z_h2h_W, self.z_h_b, \
        self.f_x2h_W, self.f_h2h_W, self.f_h_b = get_gru_variables(rng, n_in, n_out, init, inner_init)

        # params
        self.train_params.extend([self.r_x2h_W, self.r_h2h_W,
                                  self.z_x2h_W, self.z_h2h_W,
                                  self.f_x2h_W, self.f_h2h_W])
        if bias: self.train_params.extend([self.r_h_b, self.z_h_b, self.f_h_b])
        self.reg_params.extend([self.r_x2h_W, self.r_h2h_W,
                                self.z_x2h_W, self.z_h2h_W,
                                self.f_x2h_W, self.f_h2h_W])

    def __str__(self):
        return 'GRUSubNet'

    def __call__(self, input, *pre_hs):
        pre_h = pre_hs[0]
        r_mul = dot(input, self.r_x2h_W) + dot(pre_h, self.r_h2h_W)
        z_mul = dot(input, self.z_x2h_W) + dot(pre_h, self.z_h2h_W)
        if self.bias:
            r = self.gate_act_func(r_mul + self.r_h_b)
            z = self.gate_act_func(z_mul + self.z_h_b)
        else:
            r = self.gate_act_func(r_mul)
            z = self.gate_act_func(z_mul)

        c_mul = dot(input, self.f_x2h_W) + dot(r * pre_h, self.f_h2h_W)
        if self.bias:
            c = self.act_func(c_mul + self.f_h_b)
        else:
            c = self.act_func(c_mul)

        h = z * pre_h + (tensor.ones_like(z) - z) * c
        return [h, ]

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'gate_activation': self.gae_act_name,
            'bias': self.bias,
        }
        return config


class LSTM(SubNet):
    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', gate_activation='sigmoid',
                 bias=True):
        super(LSTM, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init
        self.inner_init = inner_init
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.gae_act_name = gate_activation
        self.gate_act_func = get_activation(gate_activation)
        self.bias = bias
        self.outputs_info_len = 2

        # variables
        self.f_x2h_W, self.f_h2h_W, self.f_h_b, \
        self.i_x2h_W, self.i_h2h_W, self.i_h_b, \
        self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
        self.o_x2h_W, self.o_h2h_W, self.o_h_b = get_lstm_variables(rng, n_in, n_out, init, inner_init)

        # params
        self.train_params.extend([self.f_x2h_W, self.f_h2h_W,
                                  self.i_x2h_W, self.i_h2h_W,
                                  self.g_x2h_W, self.g_h2h_W,
                                  self.o_x2h_W, self.o_h2h_W])
        if bias:
            self.train_params.extend([self.f_h_b, self.i_h_b, self.g_h_b, self.o_h_b])
        self.reg_params.extend([self.f_x2h_W, self.f_h2h_W,
                                self.i_x2h_W, self.i_h2h_W,
                                self.g_x2h_W, self.g_h2h_W,
                                self.o_x2h_W, self.o_h2h_W])

    def __str__(self):
        return "LSTMSubNet"

    def __call__(self, input, *pre_hs):
        pre_h, pre_c = pre_hs
        f_mul = dot(input, self.f_x2h_W) + dot(pre_h, self.f_h2h_W)
        i_mul = dot(input, self.i_x2h_W) + dot(pre_h, self.i_h2h_W)
        o_mul = dot(input, self.o_x2h_W) + dot(pre_h, self.o_h2h_W)
        g_mul = dot(input, self.g_x2h_W) + dot(pre_h, self.g_h2h_W)

        if self.bias:
            f = self.gate_act_func(f_mul + self.f_h_b)
            i = self.gate_act_func(i_mul + self.i_h_b)
            o = self.gate_act_func(o_mul + self.o_h_b)
            g = self.act_func(g_mul + self.g_h_b)
        else:
            f = self.gate_act_func(f_mul)
            i = self.gate_act_func(i_mul)
            o = self.gate_act_func(o_mul)
            g = self.act_func(g_mul)
        c = f * pre_c + i * g
        h = o * self.act_func(c)
        return [h, c]

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'gate_activation': self.gae_act_name,
            'bias': self.bias,
        }
        return config


class MGU(SubNet):
    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', gate_activation='sigmoid',
                 bias=True):
        super(MGU, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init
        self.inner_init = inner_init
        self.act_name = activation
        self.act_func = get_activation(activation)
        self.gae_act_name = gate_activation
        self.gate_act_func = get_activation(gate_activation)
        self.bias = bias
        self.outputs_info_len = 1

        # variables
        self.f_x2h_W, self.f_h2h_W, self.f_h_b, \
        self.i_x2h_W, self.i_h2h_W, self.i_h_b = get_mgu_variables(rng, n_in, n_out, init, inner_init)

        # params
        self.train_params.extend([self.f_x2h_W, self.f_h2h_W,
                                  self.i_x2h_W, self.i_h2h_W])
        if bias: self.train_params.extend([self.f_h_b, self.i_h_b])
        self.reg_params.extend([self.f_x2h_W, self.f_h2h_W,
                                self.i_x2h_W, self.i_h2h_W])

    def __str__(self):
        return 'MGUSubNet'

    def __call__(self, input, *pre_hs):
        pre_h = pre_hs[0]
        f_mul = dot(input, self.f_x2h_W) + dot(pre_h, self.f_h2h_W)
        if self.bias:
            f = self.gate_act_func(f_mul + self.f_h_b)
        else:
            f = self.gate_act_func(f_mul)

        c_mul = dot(input, self.i_x2h_W) + dot(f * pre_h, self.i_h2h_W)
        if self.bias:
            c = self.act_func(c_mul + self.i_h_b)
        else:
            c = self.act_func(c_mul)

        h = f * pre_h + (tensor.ones_like(f) - f) * c
        return [h, ]

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'gate_activation': self.gae_act_name,
            'bias': self.bias,
        }
        return config


class BN(SubNet):
    def __init__(self, rng, n_in, epsilon=1e-6, axis=0,
                 beta_init='zero', gamma_init='one', ):
        super(BN, self).__init__()

        # parameters
        self.n_in = n_in
        self.epsilon = epsilon
        self.axis = axis
        self.beta_init = beta_init
        self.gamma_init = gamma_init

        # variables
        self.beta = get_shared(None, (n_in,), beta_init)
        self.gamma = get_shared(rng, (n_in,), gamma_init)

        # params
        self.train_params.extend([self.beta, self.gamma])

    def __call__(self, input):
        m = tensor.mean(input, axis=self.axis)
        std = tensor.sqrt(tensor.mean(tensor.sqr(input - m), axis=self.axis))
        x = (input - m) / (std + 1e-8)
        return self.gamma * x + self.beta

    def __str__(self):
        return "BNSubNet"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'epsilon': self.epsilon,
            'axis': self.axis,
            'beta_init': self.beta_init,
            'gamma_init': self.gamma_init
        }
        return config


class MLP(SubNet):
    def __init__(self, rng, n_in, n_out,
                 activation='tanh', init='glorot_uniform', bias=True):
        super(MLP, self).__init__()

        # parameters
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init
        self.bias = bias

        # variables
        self.W = get_shared(rng, (n_in, n_out), init)
        self.b = get_shared(rng, (n_out,), 'zero')

        # params
        if bias:
            self.train_params.extend([self.W, self.b])
        else:
            self.train_params.extend([self.W, ])
        self.reg_params.extend([self.W])

    def __str__(self):
        return 'MLPSubNet'

    def __call__(self, input,):
        if self.bias:
            output = self.act_func(dot(input, self.W) + self.b)
        else:
            output = self.act_func(dot(input, self.W))
        return output

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'activation': self.act_name,
            'init': self.init,
            'bias': self.bias
        }
        return config


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


class EnNet_V1(Layer):
    def __init__(self, inner_merge='max', attention_cls=None, backward=False, return_sequence=True, **kwargs):
        """
        :param inner_merge:
        :param attention_cls:
        :param backward:
        """
        super(EnNet_V1, self).__init__()

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


class EnNet_V2(Layer):
    """
    Each path's first layer is MLP, next layer is recurrent layer.
    """


    def __init__(self, inner_merge='max', attention_cls=None, backward=False, return_sequence=True, **kwargs):
        """
        :param inner_merge:
        :param attention_cls:
        :param backward:
        """
        super(EnNet_V2, self).__init__()

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
        # i = 0
        # paths_output = []
        # for path in self.paths:
        #     input = xs
        #     for j, layer in enumerate(path):
        #         # work start
        #         pre_hs = pre_outputs_info[i: i + layer.outputs_info_len]
        #         i += layer.outputs_info_len
        #
        #         # work
        #         outputs = layer(input, *pre_hs)
        #
        #         # work end
        #         if str(layer) in ['BNSubNet', 'MLPSubNet']:
        #             input = outputs
        #         else:
        #             input = outputs[0]
        #             outputs_info.extend(outputs)
        #
        #     paths_output.append(input)

        first_layer_outputs = []
        for path in self.paths:
            first_layer_outputs.append(path[0](xs))
        # input =


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

        return outputs_info + [e]

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

    def __str__(self):
        return "EnNet_V1"

    def __call__(self, input):
        if self.outputs_info_len == 0:
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
        return_res = res[-1]
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
            config[path_name] = {}
            for j, layer in enumerate(path):
                config[path_name][str(layer)] = layer.to_json()

        if self.inner_merge == 'attention':
            config['attention'] = self.attention_cls.to_json()

        return config
