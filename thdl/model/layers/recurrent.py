# -*- coding: utf-8 -*-


import numpy as np
from theano import scan
from theano import tensor

from thdl.utils import model_variables
from .base import Layer
from ..initialization import GlorotUniform
from ..initialization import Orthogonal
from ..nonlinearity import HardSigmoid
from ..nonlinearity import Tanh


class Recurrent(Layer):
    def __init__(self, n_in, n_out, init=GlorotUniform(), inner_init=Orthogonal(), activation=Tanh(),
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 backward=False, return_sequence=True, bias=True):
        super(Recurrent, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.init = init
        self.inner_init = inner_init
        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.backward = backward
        self.return_sequence = return_sequence
        self.bias = bias

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init,
            'inner_init': self.inner_init,
            'activation': self.activation,
            'W_regularizer': self.W_regularizer,
            'U_regularizer': self.U_regularizer,
            'b_regularizer': self.b_regularizer,
            'return_sequence': self.return_sequence,
            'backward': self.backward,
            'bias': self.bias,
        }
        return config

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def get_forward(self, input, out_num, none_num=0, index=None):
        # outputs_info
        outputs_info = [tensor.alloc(np.cast[tensor.config.floatX](0.), input.shape[0], self.n_out)
                        for _ in range(out_num)]
        outputs_info += [None for _ in range(none_num)]

        # input
        assert input.ndim == 3
        input = input.dimshuffle((1, 0, 2))

        if self.backward:
            input = input[::-1]

        # scan
        outs, _ = scan(fn=self.step, sequences=input, outputs_info=outputs_info)

        # hs
        if index is None:
            hs = outs
        else:
            hs = outs[index]

        # return
        if self.backward:
            hs = hs[::-1]

        if self.return_sequence:
            return hs.dimshuffle(1, 0, 2)
        else:
            return hs[-1]

    def get_regularizers(self, W_weights, U_weights, b_weights):
        returns = []

        if self.W_regularizer:
            returns.extend([self.W_regularizer(W) for W in W_weights])

        if self.U_regularizer:
            returns.extend([self.U_regularizer(U) for U in U_weights])

        if self.b_regularizer and self.bias:
            returns.extend([self.b_regularizer(b) for b in b_weights])

        return returns


class GatedRecurrent(Recurrent):
    def __init__(self, gate_activation=HardSigmoid(), **kwargs):
        super(GatedRecurrent, self).__init__(**kwargs)

        self.gate_activation = gate_activation

    def to_json(self):
        config = {
            "gate_activation": self.gate_activation
        }
        config.update(super(GatedRecurrent, self).to_json())
        return config


class SimpleRNN(Recurrent):
    def __init__(self, **kwargs):
        super(SimpleRNN, self).__init__(**kwargs)

        self.h0 = None
        self.W, self.U, self.b = model_variables.rnn_variables(self.n_in, self.n_out, self.init, self.inner_init)

    def forward(self, input, **kwargs):
        return self.get_forward(input, 1, 0)

    def step(self, x_in, pre_h):
        out = tensor.dot(x_in, self.W) + tensor.dot(pre_h, self.U) + self.b
        return self.activation(out)

    @property
    def params(self):
        if self.bias:
            return [self.W, self.U, self.b]
        else:
            return [self.W, self.U]

    @property
    def regularizers(self):
        return self.get_regularizers([self.W], [self.U], [self.b])


class LSTM(GatedRecurrent):
    def __init__(self, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        self.f_x2h_W, self.f_h2h_W, self.f_h_b, \
        self.i_x2h_W, self.i_h2h_W, self.i_h_b, \
        self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
        self.o_x2h_W, self.o_h2h_W, self.o_h_b = \
            model_variables.lstm_variables(self.n_in, self.n_out, self.init, self.inner_init)

    def forward(self, input, **kwargs):
        return self.get_forward(input, 2, 0, index=1)

    def step(self, x_in, pre_c, pre_h):
        f = self.gate_activation(tensor.dot(x_in, self.f_x2h_W) + tensor.dot(pre_h, self.f_h2h_W) + self.f_h_b)
        i = self.gate_activation(tensor.dot(x_in, self.i_x2h_W) + tensor.dot(pre_h, self.i_h2h_W) + self.i_h_b)
        o = self.gate_activation(tensor.dot(x_in, self.o_x2h_W) + tensor.dot(pre_h, self.o_h2h_W) + self.o_h_b)
        g = self.activation(tensor.dot(x_in, self.g_x2h_W) + tensor.dot(pre_h, self.g_h2h_W) + self.g_h_b)
        c = f * pre_c + i * g
        h = o * self.activation(c)
        return c, h

    @property
    def params(self):
        ps = [self.f_x2h_W, self.f_h2h_W,
              self.i_x2h_W, self.i_h2h_W,
              self.g_x2h_W, self.g_h2h_W,
              self.o_x2h_W, self.o_h2h_W, ]
        if self.bias:
            return ps + [self.f_h_b, self.i_h_b, self.g_h_b, self.o_h_b]
        else:
            return ps

    @property
    def regularizers(self):
        W_weights = [self.f_x2h_W, self.i_x2h_W, self.g_x2h_W, self.o_x2h_W]
        U_weights = [self.f_h2h_W, self.i_h2h_W, self.g_h2h_W, self.o_h2h_W]
        b_weights = [self.f_h_b, self.i_h_b, self.g_h_b, self.o_h_b]

        return self.get_regularizers(W_weights, U_weights, b_weights)


class GRU(GatedRecurrent):
    def __init__(self, **kwargs):
        super(GRU, self).__init__(**kwargs)

        # variables
        self.h0 = None
        self.r_x2h_W, self.r_h2h_W, self.r_h_b, \
        self.z_x2h_W, self.z_h2h_W, self.z_h_b, \
        self.c_x2h_W, self.c_h2h_W, self.c_h_b = \
            model_variables.gru_variables(self.n_in, self.n_out, self.init, self.inner_init)

    def step(self, x_in, pre_h):
        r = self.gate_activation(tensor.dot(x_in, self.r_x2h_W) + tensor.dot(pre_h, self.r_h2h_W) + self.r_h_b)
        z = self.gate_activation(tensor.dot(x_in, self.z_x2h_W) + tensor.dot(pre_h, self.z_h2h_W) + self.z_h_b)
        c = self.activation(tensor.dot(x_in, self.c_x2h_W) + tensor.dot(r * pre_h, self.c_h2h_W) + self.c_h_b)
        h = z * pre_h + (tensor.ones_like(z) - z) * c
        return h

    def forward(self, input, **kwargs):
        return self.get_forward(input, 1)

    @property
    def params(self):
        ps = [self.r_x2h_W, self.r_h2h_W,
              self.z_x2h_W, self.z_h2h_W,
              self.c_x2h_W, self.c_h2h_W, ]
        if self.bias:
            return ps + [self.r_h_b, self.z_h_b, self.c_h_b]
        else:
            return ps

    @property
    def regularizers(self):
        W_weights = [self.r_x2h_W, self.z_x2h_W, self.c_x2h_W]
        U_weights = [self.r_h2h_W, self.z_h2h_W, self.c_h2h_W]
        b_weights = [self.r_h_b, self.z_h_b, self.c_h_b]

        return self.get_regularizers(W_weights, U_weights, b_weights)


class PLSTM(GatedRecurrent):
    """
    Peephole LSTM

    ------------------
    From Paper
    ------------------
        Gers, Felix A., and Jürgen Schmidhuber. "Recurrent nets that time and count."
        Neural Networks, 2000. IJCNN 2000, Proceedings of the IEEE-INNS-ENNS
        International Joint Conference on. Vol. 3. IEEE, 2000.
    """

    def __init__(self, peephole_init=GlorotUniform(), p_regularizer=None, **kwargs):
        super(PLSTM, self).__init__(*kwargs)

        self.peephole_init = peephole_init
        self.p_regularizer = p_regularizer

        self.f_x2h_W, self.f_h2h_W, self.p_f, self.f_h_b, \
        self.i_x2h_W, self.i_h2h_W, self.p_i, self.i_h_b, \
        self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
        self.o_x2h_W, self.o_h2h_W, self.p_o, self.o_h_b = \
            model_variables.plstm_variables(self.n_in, self.n_out, self.init, self.inner_init, self.peephole_init)

    def forward(self, input, **kwargs):
        return self.get_forward(input, 2, index=1)

    def step(self, x_in, pre_c, pre_h):
        f = self.gate_activation(
            tensor.dot(x_in, self.f_x2h_W) + tensor.dot(pre_h, self.f_h2h_W) + pre_c * self.p_f + self.f_h_b)
        i = self.gate_activation(
            tensor.dot(x_in, self.i_x2h_W) + tensor.dot(pre_h, self.i_h2h_W) + pre_c * self.p_i + self.i_h_b)
        o = self.gate_activation(
            tensor.dot(x_in, self.o_x2h_W) + tensor.dot(pre_h, self.o_h2h_W) + pre_c * self.p_o + self.o_h_b)
        g = self.activation(tensor.dot(x_in, self.g_x2h_W) + tensor.dot(pre_h, self.g_h2h_W) + self.g_h_b)
        c = f * pre_c + i * g
        h = o * self.activation(c)
        return c, h

    def to_json(self):
        config = {
            'peephole_init': self.peephole_init,
            'p_regularizer': self.p_regularizer,
        }
        config.update(super(PLSTM, self).to_json())
        return config

    @property
    def params(self):
        ps = [self.f_x2h_W, self.f_h2h_W, self.p_f,
              self.i_x2h_W, self.i_h2h_W, self.p_i,
              self.g_x2h_W, self.g_h2h_W,
              self.o_x2h_W, self.o_h2h_W, self.p_o, ]

        if self.bias:
            return ps + [self.f_h_b, self.i_h_b, self.g_h_b, self.o_h_b]
        else:
            return ps

    @property
    def regularizers(self):
        W_weights = [self.f_x2h_W, self.i_x2h_W, self.g_x2h_W, self.o_x2h_W]
        U_weights = [self.f_h2h_W, self.i_h2h_W, self.g_h2h_W, self.o_h2h_W]
        b_weights = [self.f_h_b, self.i_h_b, self.g_h_b, self.o_h_b]
        returns = self.get_regularizers(W_weights, U_weights, b_weights)

        if self.p_regularizer:
            for peephole in [self.p_f, self.p_i, self.p_o]:
                returns.append(self.p_regularizer(peephole))

        return returns


class CLSTM(GatedRecurrent):
    """
    Coupled LSTM

    ------------------
    From Paper
    ------------------
        Greff, Klaus, et al. "LSTM: A search space odyssey."
        arXiv preprint arXiv:1503.04069 (2015).

    """

    def __init__(self, **kwargs):
        super(CLSTM, self).__init__(**kwargs)

        self.i_x2h_W, self.i_h2h_W, self.i_h_b, \
        self.g_x2h_W, self.g_h2h_W, self.g_h_b, \
        self.o_x2h_W, self.o_h2h_W, self.o_h_b = \
            model_variables.clstm_variables(self.n_in, self.n_out, self.init, self.inner_init)

    def step(self, x_in, pre_c, pre_h):
        i = self.gate_activation(tensor.dot(x_in, self.i_x2h_W) + tensor.dot(pre_h, self.i_h2h_W) + self.i_h_b)
        o = self.gate_activation(tensor.dot(x_in, self.o_x2h_W) + tensor.dot(pre_h, self.o_h2h_W) + self.o_h_b)
        g = self.activation(tensor.dot(x_in, self.g_x2h_W) + tensor.dot(pre_h, self.g_h2h_W) + self.g_h_b)
        c = (tensor.ones_like(i) - i) * pre_c + i * g
        h = o * self.activation(c)
        return c, h

    def forward(self, input, **kwargs):
        return self.get_forward(input, 2, index=1)

    @property
    def params(self):
        ps = [self.i_x2h_W, self.i_h2h_W,
              self.g_x2h_W, self.g_h2h_W,
              self.o_x2h_W, self.o_h2h_W, ]
        if self.bias:
            return ps + [self.i_h_b, self.g_h_b, self.o_h_b]
        else:
            return ps

    @property
    def regularizers(self):
        W_weights = [self.i_x2h_W, self.g_x2h_W, self.o_x2h_W]
        U_weights = [self.i_h2h_W, self.g_h2h_W, self.o_h2h_W]
        b_weights = [self.i_h_b, self.g_h_b, self.o_h_b]

        return self.get_regularizers(W_weights, U_weights, b_weights)
