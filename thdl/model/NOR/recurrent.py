# -*- coding: utf-8 -*-

from theano import tensor

from .base import SubNet
from ..activations import get_activation
from ..variables import get_gru_variables
from ..variables import get_lstm_variables
from ..variables import get_mgu_variables
from ..variables import get_rnn_variables

dot = tensor.dot


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
        return [out, ]

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
