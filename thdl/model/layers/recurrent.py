# -*- coding: utf-8 -*-


import numpy as np
from theano import scan
from theano import tensor

from .base import Layer
from ..activation import get_activation
from ..variables import dtype
from ..variables import get_clstm_variables
from ..variables import get_gru_variables
from ..variables import get_lstm_variables
from ..variables import get_plstm_variables
from ..variables import get_rnn_variables

dot = tensor.dot


class Recurrent(object):
    def __init__(self, rng, n_in, n_out,
                 bptt=-1, fixed=False, backward=False, return_sequence=True,
                 ):
        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.bptt = bptt
        self.fixed = fixed
        self.backward = backward
        self.return_sequence = return_sequence


class SimpleRNN(Layer):
    def __init__(self, rng, n_in, n_out, mask=None,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 bptt=-1, fixed=False, backward=False, return_sequence=True,
                 params=None):
        super(SimpleRNN, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init
        self.inner_init = inner_init
        self.bptt = bptt
        self.fixed = fixed
        self.backward = backward
        self.return_sequence = return_sequence

        # variables
        self.h0 = None
        if params is None:
            self.W, self.U, self.b = get_rnn_variables(rng, n_in, n_out, init, inner_init)
        else:
            self.W, self.U, self.b = params

        # params
        self.train_params.extend([self.W, self.U, self.b])
        self.reg_params.extend([self.W, self.U])

    def fixed_step(self, *args):
        h = self.h0
        for x_in in args:
            h = self.act_func(dot(x_in, self.W) + dot(h, self.U) + self.b)
        return h

    def unfixed_step(self, x_in, pre_h):
        return self.act_func(dot(x_in, self.W) + dot(pre_h, self.U) + self.b)

    def __call__(self, input):
        # input
        if input.ndim == 3:
            self.h0 = tensor.alloc(np.cast[tensor.config.floatX](0.), input.shape[0], self.n_out)
            input = input.dimshuffle((1, 0, 2))

        elif input.ndim == 2:
            self.h0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            input = input

        else:
            raise ValueError("Unknown input dimension: %d" % input.ndim)

        if self.backward:
            input = input[::-1]

        # outputs
        if self.fixed:
            assert self.bptt > 0
            taps = list(range(-self.bptt + 1, 1))
            sequences = [dict(input=input, taps=taps)]
            hs, _ = scan(fn=self.fixed_step, sequences=sequences)
        else:
            hs, _ = scan(fn=self.unfixed_step, sequences=input, outputs_info=self.h0,
                         truncate_gradient=self.bptt)

        # return
        if self.backward:
            hs = hs[::-1]

        if input.ndim == 3:
            if self.return_sequence:
                return hs.dimshuffle(1, 0, 2)
            else:
                return hs[-1]

        if input.ndim == 2:
            if self.return_sequence:
                return hs
            else:
                return hs[-1]

    def __str__(self):
        return "SimpleRNN"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'bptt': self.bptt,
            'fixed': self.fixed,
            'return_sequence': self.return_sequence,
            'backward': self.backward,

        }
        return config


class LSTM(Layer):
    def __init__(self, rng, n_in, n_out, mask=None,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', gate_activation='sigmoid',
                 bptt=-1, fixed=False, backward=False, return_sequence=True, bias=True,
                 params=None):
        super(LSTM, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init
        self.inner_init = inner_init
        self.gate_act_func = get_activation(gate_activation)
        self.act_func = get_activation(activation)
        self.gate_act_name = gate_activation
        self.act_name = activation
        self.bptt = bptt
        self.fixed = fixed
        self.backward = backward
        self.return_sequence = return_sequence
        self.mask = mask
        self.bias = bias
        if mask is not None:
            self.masks.append(mask)

        # variables
        self.h0 = None
        self.c0 = None

        if params:
            self.f_x2h_W, self.f_h2h_W, self.f_h_b, \
            self.i_x2h_W, self.i_h2h_W, self.i_h_b, \
            self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
            self.o_x2h_W, self.o_h2h_W, self.o_h_b = params
        else:
            self.f_x2h_W, self.f_h2h_W, self.f_h_b, \
            self.i_x2h_W, self.i_h2h_W, self.i_h_b, \
            self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
            self.o_x2h_W, self.o_h2h_W, self.o_h_b = get_lstm_variables(rng, n_in, n_out, init, inner_init)

        # params
        self.train_params.extend([self.f_x2h_W, self.f_h2h_W,
                                  self.i_x2h_W, self.i_h2h_W,
                                  self.g_x2h_W, self.g_h2h_W,
                                  self.o_x2h_W, self.o_h2h_W, ])
        if bias:
            self.train_params.extend([self.f_h_b, self.i_h_b, self.g_h_b, self.o_h_b])
        self.reg_params.extend([self.f_x2h_W, self.f_h2h_W,
                                self.i_x2h_W, self.i_h2h_W,
                                self.g_x2h_W, self.g_h2h_W,
                                self.o_x2h_W, self.o_h2h_W])

    def fixed_step(self, *args):
        h, c = self.h0, self.c0
        for x_in in args:
            f = self.gate_act_func(dot(x_in, self.f_x2h_W) + dot(h, self.f_h2h_W) + self.f_h_b)
            i = self.gate_act_func(dot(x_in, self.i_x2h_W) + dot(h, self.i_h2h_W) + self.i_h_b)
            o = self.gate_act_func(dot(x_in, self.o_x2h_W) + dot(h, self.o_h2h_W) + self.o_h_b)
            g = self.act_func(dot(x_in, self.g_x2h_W) + dot(h, self.g_h2h_W) + self.g_h_b)
            c = f * c + i * g
            h = o * self.act_func(c)
        return c, h

    def unfixed_step(self, x_in, pre_c, pre_h):
        f = self.gate_act_func(dot(x_in, self.f_x2h_W) + dot(pre_h, self.f_h2h_W) + self.f_h_b)
        i = self.gate_act_func(dot(x_in, self.i_x2h_W) + dot(pre_h, self.i_h2h_W) + self.i_h_b)
        o = self.gate_act_func(dot(x_in, self.o_x2h_W) + dot(pre_h, self.o_h2h_W) + self.o_h_b)
        g = self.act_func(dot(x_in, self.g_x2h_W) + dot(pre_h, self.g_h2h_W) + self.g_h_b)
        c = f * pre_c + i * g
        h = o * self.act_func(c)
        return c, h

    def mask_unfixed_step(self, x_in, mask, pre_c, pre_h):
        c, h = self.unfixed_step(x_in, pre_c, pre_h)

        c = tensor.switch(mask, c, pre_c)
        h = tensor.switch(mask, h, pre_h)

        return c, h

    def __call__(self, input, mask=None):
        # input
        if input.ndim == 3:
            self.h0 = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_out)
            self.c0 = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_out)
            input = input.dimshuffle(1, 0, 2)
            if self.mask is not None:
                mask = self.mask.dimshuffle(1, 0, 'x')

        elif input.ndim == 2:
            self.h0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            self.c0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            input = input
            if self.mask is not None:
                mask = self.mask.dimshuffle(0, 'x')

        else:
            raise ValueError("Unknown input dimension: %d" % input.ndim)

        if self.backward:
            input = input[::-1]

        # outputs
        if self.fixed:
            assert self.bptt > 0
            taps = list(range(-self.bptt + 1, 1))
            sequences = [dict(input=input, taps=taps)]
            [cs, hs], _ = scan(fn=self.fixed_step, sequences=sequences)
        else:
            if self.mask is not None:
                seqs = [input, mask]
                step = self.mask_unfixed_step
            else:
                seqs = input
                step = self.unfixed_step
            [cs, hs], _ = scan(fn=step, sequences=seqs, outputs_info=[self.c0, self.h0],
                               truncate_gradient=self.bptt)

        # return
        if input.ndim == 3:
            if self.return_sequence:
                if self.backward:
                    hs = hs[::-1]
                return hs.dimshuffle(1, 0, 2)
            else:
                return hs[-1]

        if input.ndim == 2:
            if self.return_sequence:
                if self.backward:
                    hs = hs[::-1]
                return hs
            else:
                return hs[-1]

    def __str__(self):
        return 'LSTM'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'mask': self.mask,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'gate_activation': self.gate_act_name,
            'bptt': self.bptt,
            'fixed': self.fixed,
            'return_sequence': self.return_sequence,
            'backward': self.backward
        }
        return config


class PLSTM(Layer):
    """
    Peephole LSTM

    ------------------
    From Paper
    ------------------
        Gers, Felix A., and Jürgen Schmidhuber. "Recurrent nets that time and count."
        Neural Networks, 2000. IJCNN 2000, Proceedings of the IEEE-INNS-ENNS
        International Joint Conference on. Vol. 3. IEEE, 2000.
    """

    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform', inner_init='orthogonal', peephole_init='uniform',
                 activation='tanh', gate_activation='sigmoid',
                 bptt=-1, fixed=False, backward=False, return_sequence=True,
                 params=None):
        super(PLSTM, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init
        self.inner_init = inner_init
        self.peephole_init = peephole_init
        self.gate_act_func = get_activation(gate_activation)
        self.act_func = get_activation(activation)
        self.gate_act_name = gate_activation
        self.act_name = activation

        self.bptt = bptt
        self.fixed = fixed
        self.backward = backward
        self.return_sequence = return_sequence

        # variables
        self.h0 = None
        self.c0 = None

        if params:
            self.f_x2h_W, self.f_h2h_W, self.p_f, self.f_h_b, \
            self.i_x2h_W, self.i_h2h_W, self.p_i, self.i_h_b, \
            self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
            self.o_x2h_W, self.o_h2h_W, self.p_o, self.o_h_b = params
        else:
            self.f_x2h_W, self.f_h2h_W, self.p_f, self.f_h_b, \
            self.i_x2h_W, self.i_h2h_W, self.p_i, self.i_h_b, \
            self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
            self.o_x2h_W, self.o_h2h_W, self.p_o, self.o_h_b = \
                get_plstm_variables(rng, n_in, n_out, init, inner_init, peephole_init)

        # params
        self.train_params.extend([self.f_x2h_W, self.f_h2h_W, self.p_f, self.f_h_b,
                                  self.i_x2h_W, self.i_h2h_W, self.p_i, self.i_h_b,
                                  self.g_x2h_W, self.g_h2h_W, self.g_h_b,
                                  self.o_x2h_W, self.o_h2h_W, self.p_o, self.o_h_b])
        self.reg_params.extend([self.f_x2h_W, self.f_h2h_W, self.p_f,
                                self.i_x2h_W, self.i_h2h_W, self.p_i,
                                self.g_x2h_W, self.g_h2h_W,
                                self.o_x2h_W, self.o_h2h_W, self.p_o])

    def fixed_step(self, *args):
        h, c = self.h0, self.c0
        for x_in in args:
            f = self.gate_act_func(dot(x_in, self.f_x2h_W) + dot(h, self.f_h2h_W) + c * self.p_f + self.f_h_b)
            i = self.gate_act_func(dot(x_in, self.i_x2h_W) + dot(h, self.i_h2h_W) + c * self.p_i + self.i_h_b)
            o = self.gate_act_func(dot(x_in, self.o_x2h_W) + dot(h, self.o_h2h_W) + c * self.p_o + self.o_h_b)
            g = self.act_func(dot(x_in, self.g_x2h_W) + dot(h, self.g_h2h_W) + self.g_h_b)
            c = f * c + i * g
            h = o * self.act_func(c)
        return c, h

    def unfixed_step(self, x_in, pre_c, pre_h):
        f = self.gate_act_func(dot(x_in, self.f_x2h_W) + dot(pre_h, self.f_h2h_W) + pre_c * self.p_f + self.f_h_b)
        i = self.gate_act_func(dot(x_in, self.i_x2h_W) + dot(pre_h, self.i_h2h_W) + pre_c * self.p_i + self.i_h_b)
        o = self.gate_act_func(dot(x_in, self.o_x2h_W) + dot(pre_h, self.o_h2h_W) + pre_c * self.p_o + self.o_h_b)
        g = self.act_func(dot(x_in, self.g_x2h_W) + dot(pre_h, self.g_h2h_W) + self.g_h_b)
        c = f * pre_c + i * g
        h = o * self.act_func(c)
        return c, h

    def __call__(self, input):
        # input
        if input.ndim == 3:
            self.h0 = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_out)
            self.c0 = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_out)
            input = input.dimshuffle((1, 0, 2))

        elif input.ndim == 2:
            self.h0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            self.c0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            input = input

        else:
            raise ValueError("Unknown input dimension: %d" % input.ndim)

        if self.backward:
            input = input[::-1]

        # outputs
        if self.fixed:
            assert self.bptt > 0
            taps = list(range(-self.bptt + 1, 1))
            sequences = [dict(input=input, taps=taps)]
            [cs, hs], _ = scan(fn=self.fixed_step, sequences=sequences)
        else:
            [cs, hs], _ = scan(fn=self.unfixed_step, sequences=input, outputs_info=[self.c0, self.h0],
                               truncate_gradient=self.bptt)

        # return
        if self.backward:
            hs = hs[::-1]

        if input.ndim == 3:
            if self.return_sequence:
                return hs.dimshuffle(1, 0, 2)
            else:
                return hs[-1]

        if input.ndim == 2:
            if self.return_sequence:
                return hs
            else:
                return hs[-1]

    def __str__(self):
        return 'PeepholeLSTM'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'peephole_init': self.peephole_init,
            'activation': self.act_name,
            'gate_activation': self.gate_act_name,
            'bptt': self.bptt,
            'fixed': self.fixed,
            'return_sequence': self.return_sequence,
            'backward': self.backward
        }
        return config


class CLSTM(Layer):
    """
    Coupled LSTM

    ------------------
    From Paper
    ------------------
        Greff, Klaus, et al. "LSTM: A search space odyssey."
        arXiv preprint arXiv:1503.04069 (2015).

    """

    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', gate_activation='sigmoid',
                 bptt=-1, fixed=False, backward=False, return_sequence=True,
                 params=None):
        super(CLSTM, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init
        self.inner_init = inner_init
        self.gate_act_func = get_activation(gate_activation)
        self.act_func = get_activation(activation)
        self.gate_act_name = gate_activation
        self.act_name = activation
        self.bptt = bptt
        self.fixed = fixed
        self.backward = backward
        self.return_sequence = return_sequence

        # variables
        self.h0 = None
        self.c0 = None

        if params:
            self.i_x2h_W, self.i_h2h_W, self.i_h_b, \
            self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
            self.o_x2h_W, self.o_h2h_W, self.o_h_b = params
        else:
            self.i_x2h_W, self.i_h2h_W, self.i_h_b, \
            self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
            self.o_x2h_W, self.o_h2h_W, self.o_h_b = get_clstm_variables(rng, n_in, n_out, init, inner_init)

        # params
        self.train_params.extend([self.i_x2h_W, self.i_h2h_W, self.i_h_b,
                                  self.g_x2h_W, self.g_h2h_W, self.g_h_b,
                                  self.o_x2h_W, self.o_h2h_W, self.o_h_b])
        self.reg_params.extend([self.i_x2h_W, self.i_h2h_W,
                                self.g_x2h_W, self.g_h2h_W,
                                self.o_x2h_W, self.o_h2h_W])

    def fixed_step(self, *args):
        h, c = self.h0, self.c0
        for x_in in args:
            i = self.gate_act_func(dot(x_in, self.i_x2h_W) + dot(h, self.i_h2h_W) + self.i_h_b)
            o = self.gate_act_func(dot(x_in, self.o_x2h_W) + dot(h, self.o_h2h_W) + self.o_h_b)
            g = self.act_func(dot(x_in, self.g_x2h_W) + dot(h, self.g_h2h_W) + self.g_h_b)
            c = (tensor.ones_like(i) - i) * c + i * g
            h = o * self.act_func(c)
        return c, h

    def unfixed_step(self, x_in, pre_c, pre_h):
        i = self.gate_act_func(dot(x_in, self.i_x2h_W) + dot(pre_h, self.i_h2h_W) + self.i_h_b)
        o = self.gate_act_func(dot(x_in, self.o_x2h_W) + dot(pre_h, self.o_h2h_W) + self.o_h_b)
        g = self.act_func(dot(x_in, self.g_x2h_W) + dot(pre_h, self.g_h2h_W) + self.g_h_b)
        c = (tensor.ones_like(i) - i) * pre_c + i * g
        h = o * self.act_func(c)
        return c, h

    def __call__(self, input):
        # input
        if input.ndim == 3:
            self.h0 = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_out)
            self.c0 = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_out)
            input = input.dimshuffle((1, 0, 2))

        elif input.ndim == 2:
            self.h0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            self.c0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            input = input

        else:
            raise ValueError("Unknown input dimension: %d" % input.ndim)

        if self.backward:
            input = input[::-1]

        # outputs
        if self.fixed:
            assert self.bptt > 0
            taps = list(range(-self.bptt + 1, 1))
            sequences = [dict(input=input, taps=taps)]
            [cs, hs], _ = scan(fn=self.fixed_step, sequences=sequences)
        else:
            [cs, hs], _ = scan(fn=self.unfixed_step, sequences=input, outputs_info=[self.c0, self.h0],
                               truncate_gradient=self.bptt)

        # return
        if self.backward:
            hs = hs[::-1]

        if input.ndim == 3:
            if self.return_sequence:
                return hs.dimshuffle(1, 0, 2)
            else:
                return hs[-1]

        if input.ndim == 2:
            if self.return_sequence:
                return hs
            else:
                return hs[-1]

    def __str__(self):
        return 'CoupledLSTM'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'gate_activation': self.gate_act_name,
            'bptt': self.bptt,
            'fixed': self.fixed,
            'return_sequence': self.return_sequence,
            'backward': self.backward
        }
        return config


class GRU(Layer):
    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh', gate_activation='sigmoid',
                 bptt=-1, fixed=False, backward=False, return_sequence=True,
                 params=None, ):
        super(GRU, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.gate_act_func = get_activation(gate_activation)
        self.act_func = get_activation(activation)
        self.gate_act_name = gate_activation
        self.act_name = activation
        self.init = init
        self.inner_init = inner_init
        self.bptt = bptt
        self.fixed = fixed
        self.backward = backward
        self.return_sequence = return_sequence

        # variables
        self.h0 = None
        if params:
            self.r_x2h_W, self.r_h2h_W, self.r_h_b, \
            self.z_x2h_W, self.z_h2h_W, self.z_h_b, \
            self.c_x2h_W, self.c_h2h_W, self.c_h_b = params
        else:
            self.r_x2h_W, self.r_h2h_W, self.r_h_b, \
            self.z_x2h_W, self.z_h2h_W, self.z_h_b, \
            self.c_x2h_W, self.c_h2h_W, self.c_h_b = get_gru_variables(rng, n_in, n_out, init, inner_init)

        # params
        self.train_params.extend([self.r_x2h_W, self.r_h2h_W, self.r_h_b,
                                  self.z_x2h_W, self.z_h2h_W, self.z_h_b,
                                  self.c_x2h_W, self.c_h2h_W, self.c_h_b])
        self.reg_params.extend([self.r_x2h_W, self.r_h2h_W,
                                self.z_x2h_W, self.z_h2h_W,
                                self.c_x2h_W, self.c_h2h_W])

    def fixed_step(self, *args):
        h = self.h0
        for x_in in args:
            r = self.gate_act_func(dot(x_in, self.r_x2h_W) + dot(h, self.r_h2h_W) + self.r_h_b)
            z = self.gate_act_func(dot(x_in, self.z_x2h_W) + dot(h, self.z_h2h_W) + self.z_h_b)
            c = self.act_func(dot(x_in, self.c_x2h_W) + dot(r * h, self.c_h2h_W) + self.c_h_b)
            h = z * h + (tensor.ones_like(z) - z) * c
        return h

    def unfixed_step(self, x_in, pre_h):
        r = self.gate_act_func(dot(x_in, self.r_x2h_W) + dot(pre_h, self.r_h2h_W) + self.r_h_b)
        z = self.gate_act_func(dot(x_in, self.z_x2h_W) + dot(pre_h, self.z_h2h_W) + self.z_h_b)
        c = self.act_func(dot(x_in, self.c_x2h_W) + dot(r * pre_h, self.c_h2h_W) + self.c_h_b)
        h = z * pre_h + (tensor.ones_like(z) - z) * c
        return h

    def __call__(self, input):
        # input
        if input.ndim == 3:
            self.h0 = tensor.alloc(np.cast[tensor.config.floatX](0.), input.shape[0], self.n_out)
            input = input.dimshuffle((1, 0, 2))

        elif input.ndim == 2:
            self.h0 = tensor.alloc(np.cast[tensor.config.floatX](0.), self.n_out)
            input = input

        else:
            raise ValueError("Unknown input dimension: %d" % input.ndim)

        if self.backward:
            input = input[::-1]

        # outputs
        if self.fixed:
            assert self.bptt > 0
            taps = list(range(-self.bptt + 1, 1))
            sequences = [dict(input=input, taps=taps)]
            hs, _ = scan(fn=self.fixed_step, sequences=sequences)
        else:
            hs, _ = scan(fn=self.unfixed_step, sequences=input, outputs_info=self.h0,
                         truncate_gradient=self.bptt)

        # return
        if self.backward:
            hs = hs[::-1]

        if input.ndim == 3:
            if self.return_sequence:
                return hs.dimshuffle(1, 0, 2)
            else:
                return hs[-1]

        if input.ndim == 2:
            if self.return_sequence:
                return hs
            else:
                return hs[-1]

    def __str__(self):
        return 'GRU'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.act_name,
            'gate_activation': self.gate_act_name,
            'bptt': self.bptt,
            'fixed': self.fixed,
            'return_sequence': self.return_sequence,
            'backward': self.backward
        }
        return config


class Bidirectional(Layer):
    def __init__(self, rnn, merge_mode='concat', **rnn_params):
        super(Bidirectional, self).__init__()

        # parameters
        self.merge_mode = merge_mode
        self.rnn_params = rnn_params
        self.n_in = rnn_params['n_in']

        if self.merge_mode == 'concat':
            self.n_out = rnn_params['n_out'] * 2
        else:
            raise ValueError("Unknown merge mode: %s" % self.merge_mode)

        # get rnn
        if type(rnn).__name__ == 'str':
            rnn = get_rnn(rnn)
        self.forward_rnn = rnn(backward=False, **rnn_params)
        self.backward_rnn = rnn(backward=True, **rnn_params)

        # params
        self.train_params.extend(self.backward_rnn.train_params + self.forward_rnn.train_params)
        self.reg_params.extend(self.backward_rnn.reg_params + self.forward_rnn.reg_params)

    def __call__(self, input):
        forward_res = self.forward_rnn(input)
        backward_res = self.backward_rnn(input)

        if self.merge_mode == 'concat':
            res = tensor.concatenate([forward_res, backward_res], axis=-1)

        return res

    def __str__(self):
        return "Bidirectional-%s" % str(self.forward_rnn)

    def to_json(self):
        rnn_params = self.forward_rnn.to_json()
        rnn_params.pop('backward')

        config = {
            'rnn_params': rnn_params,
            'merge_mode': self.merge_mode,
            'rnn': str(self.forward_rnn),
        }

        return config


def get_rnn(rnn_type):
    rnn_type = rnn_type.lower()

    if rnn_type in ['rnn', 'simple_rnn', 'RNN']:
        return SimpleRNN

    elif rnn_type in ['lstm', 'LSTM']:
        return LSTM

    elif rnn_type in ['plstm', 'peephole_lstm', 'PLSTM']:
        return PLSTM

    elif rnn_type in ['clstm', 'coupled_lstm', 'CLSTM']:
        return CLSTM

    elif rnn_type in ['gru', 'GRU']:
        return GRU

    else:
        raise ValueError("Unknown rnn type: %s" % rnn_type)
