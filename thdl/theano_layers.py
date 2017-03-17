# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2016/11/5

@notes:
    
"""

import json
from collections import OrderedDict

import numpy as np
from theano import function
from theano import scan
from theano import shared
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv
from theano.tensor.signal import pool

from .theano_tools import dtype
from .theano_tools import get_activation
from .theano_tools import get_clstm_variables
from .theano_tools import get_gru_variables
from .theano_tools import get_loss
from .theano_tools import get_lstm_variables
from .theano_tools import get_plstm_variables
from .theano_tools import get_regularization
from .theano_tools import get_rnn_variables
from .theano_tools import get_shared
from .theano_tools import get_updates

dot = tensor.dot


class Layer(object):
    @classmethod
    def from_json(cls, config):
        if type(config).__name__ == 'str':
            config = json.loads(config)

        if type(config).__name__ == 'dict':
            return cls(**config)

        else:
            raise ValueError("config must be dict object.")

    def to_json(self):
        raise NotImplementedError("Every Layer class should implement the 'to_json' method.")

    def __init__(self, ):
        self.train_params = []
        self.reg_params = []
        self.updates = OrderedDict()
        self.masks = []

    def __str__(self):
        raise NotImplementedError("Every Layer class should implement the '__str__' method.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Every Layer class should implement the '__call__' method.")


class XY(Layer):
    def __init__(self, X=None, Y=None,
                 x_ndim=2, x_tensor_type='int32',
                 y_ndim=1, y_tensor_type='int32'):
        """
        :param X:
        :param Y:
        :param x_ndim:
        :param x_tensor_type:
        :param y_ndim:
        :param y_tensor_type:
        """

        super(XY, self).__init__()

        self.x_ndim = x_ndim
        self.x_tensor_type = x_tensor_type
        self.y_ndim = y_ndim
        self.y_tensor_type = y_tensor_type
        self.X_ = X
        self.Y_ = Y

        self.X = tensor.TensorType(self.x_tensor_type, [False] * x_ndim)() if X is None else X
        self.Y = tensor.TensorType(self.y_tensor_type, [False] * y_ndim)() if Y is None else Y

    def __str__(self):
        return 'XY'

    def __call__(self, *args, **kwargs):
        return

    def to_json(self):
        config = {
            'x_ndim': self.x_ndim,
            'x_tensor_type': self.x_tensor_type,
            'y_ndim': self.y_ndim,
            'y_tensor_type': self.y_tensor_type,
            'X': self.X_,
            'Y': self.Y_
        }

        return config


class Activation(Layer):
    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__()
        self.act_name = activation
        self.act_func = get_activation(activation, **self.kwargs)
        self.kwargs = kwargs

    def __call__(self, input):
        return self.act_func(input)

    def __str__(self):
        return 'Activation'

    def to_json(self):
        config = {
            'activation': self.act_name,
            'kwargs': self.kwargs
        }
        return config


class Conv2D(Layer):
    def __init__(self, rng, image_shape, filter_shape,
                 activation='relu', init='glorot_uniform',
                 W=None, b=None):
        """
        :param rng:
        :param filter_shape: (number of filters, num input feature maps, filter height, filter width)
        :param image_shape: (batch size, num input feature maps, image height, image width)
        :param W:
        :param b:
        :param activation:
        :param init:
        """
        super(Conv2D, self).__init__()

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init

        # variables
        self.W = get_shared(rng, filter_shape, init) if W is None else W
        self.b = get_shared(rng, (filter_shape[0],), 'zero') if b is None else b

        # params
        self.train_params.extend([self.W, self.b])
        self.reg_params.extend([self.W])

    def __call__(self, input):
        # output
        conv_out = conv.conv2d(input=input, filters=self.W, image_shape=self.image_shape,
                               filter_shape=self.filter_shape)
        output = self.act_func(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        return output

    def __str__(self):
        return "Conv2D"

    def to_json(self):
        config = {
            'image_shape': self.image_shape,
            'filter_shape': self.filter_shape,
            'activation': self.act_name,
            'init': self.init,
        }
        return config


class Pool2D(Layer):
    def __init__(self, pool_size, pad=(0, 0), ignore_border=True, mode='max'):
        super(Pool2D, self).__init__()

        self.pool_size = pool_size  # (vertical ds, horizontal ds)
        self.pad = pad
        self.ignore_border = ignore_border
        self.mode = mode

    def __call__(self, input):
        # output
        output = pool.pool_2d(input=input, ws=self.pool_size, pad=self.pad,
                              ignore_border=self.ignore_border, mode=self.mode)

        return output

    def __str__(self):
        return 'Pool2D'

    def to_json(self):
        config = {
            'pool_size': self.pool_size,
            'pad': self.pad,
            'ignore_border': self.ignore_border,
            'mode': self.mode
        }
        return config


class Dropout(Layer):
    def __init__(self, rng, p):
        """
        :param rng: a random number generator used to initialize weights
        :param p: the probability of dropping a unit
        """
        super(Dropout, self).__init__()

        # parameters
        self.srng = RandomStreams(seed=rng.randint(100, 2147462579))
        self.p = p

    def __call__(self, input, train=True):
        """
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :param train:
        :return:
        """
        # outputs
        if 0. < self.p < 1.:
            if train:
                output = input * self.srng.binomial(
                    n=1, p=1 - self.p, size=input.shape,
                    dtype=input.dtype) / (1 - self.p)
            else:
                output = input * (1 - self.p)
        else:
            output = input
        return output

    def __str__(self):
        return 'Dropout'

    def to_json(self):
        config = {
            'p': self.p
        }
        return config


class Dense(Layer):
    def __init__(self, rng, n_in, n_out,
                 activation='tanh', init='glorot_uniform',
                 W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :param rng: a random number generator used to initialize weights
        :param n_in: dimensionality of input
        :param n_out: number of hidden units
        :param activation: Non linearity to be applied in the hidden layer
        :param init: parameter initialize
        """
        super(Dense, self).__init__()

        # parameters
        self.rng = rng
        self.n_in = n_in
        self.n_out = n_out
        self.act_func = get_activation(activation)
        self.act_name = activation
        self.init = init

        # variables
        self.W = get_shared(rng, (n_in, n_out), init) if W is None else W
        self.b = get_shared(rng, (n_out,), 'zero') if b is None else b

        # params
        self.train_params.extend([self.W, self.b])
        self.reg_params.extend([self.W])

    def __call__(self, input):
        # outputs
        output = dot(input, self.W) + self.b
        output = self.act_func(output)

        return output

    def __str__(self):
        return "Dense"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'activation': self.act_name,
            'init': self.init
        }
        return config


class Softmax(Layer):
    def __init__(self, rng, n_in, n_out,
                 init='glorot_uniform',
                 W=None, b=None):
        super(Softmax, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_out
        self.init = init

        # variables
        self.W = get_shared(rng, (n_in, n_out), init) if W is None else W
        self.b = get_shared(rng, (n_out,), 'zero') if b is None else b

        # params
        self.train_params.extend([self.W, self.b])
        self.reg_params.extend([self.W])

    def __call__(self, input):
        # outputs
        output = tensor.nnet.softmax(dot(input, self.W) + self.b)
        return output

    def __str__(self):
        return 'Softmax'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'n_out': self.n_out,
            'init': self.init
        }
        return config


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


class BatchNormal(Layer):
    """
    Copy from 'https://github.com/GabrielPereyra/norm-rnn'
    """

    def __init__(self, rng, n_in, epsilon=1e-6, momentum=0.9, axis=0,
                 beta_init='zero', gamma_init='one', params=None):
        super(BatchNormal, self).__init__()

        # parameters
        self.n_in = n_in
        self.epsilon = epsilon
        self.momentum = momentum
        self.axis = axis
        self.beta_init = beta_init
        self.gamma_init = gamma_init

        # variables
        self.running_mean = None
        self.running_std = None

        if params:
            self.beta, self.gamma = params
        else:
            self.beta = get_shared(rng, (n_in,), beta_init)
            self.gamma = get_shared(rng, (n_in,), gamma_init)

        # params
        self.train_params.extend([self.beta, self.gamma])

    def __call__(self, input, train=True):
        if input.ndim == 3:
            # init variables
            self.running_mean = tensor.alloc(np.cast[dtype](0.), input.shape[0], input.shape[1], self.n_in)
            self.running_std = tensor.alloc(np.cast[dtype](0.), input.shape[0], input.shape[1], self.n_in)
        elif input.ndim == 2:
            # init variables
            self.running_mean = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_in)
            self.running_std = tensor.alloc(np.cast[dtype](0.), input.shape[0], self.n_in)
        else:
            raise ValueError("")

        # batch statistics
        m = tensor.mean(input, axis=self.axis)
        std = tensor.sqrt(tensor.mean(tensor.sqr(input - m) + self.epsilon, axis=self.axis))

        # update shared running averages
        mean_update = (1 - self.momentum) * self.running_mean + self.momentum * m
        std_update = (1 - self.momentum) * self.running_std + self.momentum * std
        self.updates[self.running_mean] = mean_update
        self.updates[self.running_std] = std_update

        # normalize using running averages
        # (is this better than batch statistics?)
        # (this version seems like it is using the running average
        #  of the previous batch since updates happens after)
        if train:
            input = (input - m) / (std + self.epsilon)
        else:
            input = (input - mean_update) / (std_update + self.epsilon)

        # scale and shift
        return self.gamma * input + self.beta

    def __str__(self):
        return "BatchNormal"

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'axis': self.axis,
            'beta_init': self.beta_init,
            'gamma_init': self.gamma_init
        }
        return config


class Embedding(Layer):
    def __init__(self, embed_words=None, static=None,
                 rng=None, input_size=None, n_out=None,
                 init='uniform', **kwargs):
        """
        :param input_size:
        :param n_out:
        :param embed_words:
        """

        super(Embedding, self).__init__()

        # parameters
        self.init = init
        self.kwargs = kwargs

        if embed_words is not None:
            if static is None:
                self.static = True
            else:
                self.static = static

            self.input_size, self.n_out = embed_words.shape
            self.embed_words = shared(embed_words, name='embedd_words')
        else:
            if static is None:
                self.static = False
            else:
                self.static = static
            self.input_size, self.n_out = input_size, n_out
            self.embed_words = get_shared(rng, (input_size, n_out), init, **kwargs)

        if self.static is True:
            pass
        elif self.static is False:
            self.train_params = [self.embed_words]
        else:
            raise ValueError("static should be True or False.")

    def __call__(self, input):
        assert input.ndim == 2

        shape = (input.shape[0], input.shape[1], self.embed_words.shape[1])
        return self.embed_words[input.flatten()].reshape(shape)

    def __str__(self):
        return 'Embedding'

    def to_json(self):
        config = {
            'static': self.static,
            'input_size': self.input_size,
            'n_out': self.n_out,
            'init': self.init,
            'kwargs': self.kwargs
        }
        return config


class ToolBox(Layer):
    def __init__(self, tool, *args, **kwargs):
        super(ToolBox, self).__init__()

        self.tool = tool
        self.args = args
        self.kwargs = kwargs

    def __call__(self, input):
        if self.tool == 'flatten':
            return input.flatten(self.kwargs['ndim'])

        if self.tool == 'mean':
            return tensor.mean(input, axis=self.kwargs['axis'])

        if self.tool == 'reshape':
            return tensor.reshape(input, newshape=self.kwargs['newshape'])

        if self.tool == 'dimshuffle':
            return input.dimshuffle(self.kwargs['pattern'])

        raise ValueError("Unknown tool method: %s" % self.tool)

    def __str__(self):
        return "ToolBox"

    def to_json(self):
        config = {
            'tool': self.tool,
            'args': self.args,
            'kwargs': self.kwargs
        }
        return config


class Attention(Layer):
    def __init__(self, rng, n_in, activation='tanh',
                 init='orthogonal', context_init='glorot_uniform', ):
        super(Attention, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_in
        self.init = init
        self.context_init = context_init
        self.act_name = activation
        self.act_func = get_activation(activation)

        # variables
        self.att_W = get_shared(rng, (n_in, n_in), init)
        self.att_b = get_shared(None, (n_in,), 'zero')
        self.att_vec = get_shared(rng, (n_in,), context_init)

        self.train_params.extend([self.att_W, self.att_b, self.att_vec])
        self.reg_params.extend([self.att_W])

    def __call__(self, input):
        assert input.ndim == 3

        a = dot(self.act_func(dot(input, self.att_W) + self.att_b), self.att_vec)
        b = tensor.nnet.softmax(a)
        c = input.dimshuffle(2, 0, 1) * b
        d = tensor.sum(c, axis=2)
        e = d.dimshuffle(1, 0)

        return e

    def __str__(self):
        return 'Attention'

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'activation': self.act_name,
            'init': self.init,
            'context_init': self.context_init
        }
        return config


class Model(object):
    def __init__(self, l1=0., l2=0., loss='nll', optimizer='sgd', seed=23455, max_norm=False):
        self.l1 = l1
        self.l2 = l2
        self.loss = loss
        self.optimizer = optimizer
        self.seed = seed
        self.max_norm = max_norm

        self.lr = tensor.scalar()

        self.layers = []
        self.reg_params = []
        self.train_params = []
        self.masks = []
        self.updates = OrderedDict()
        self.train_test_split = False

        self.train = None
        self.predict = None

    def __check_train_test_split(self, layer):
        if type(layer).__name__ in ['Dropout', 'BatchNormal']:
            self.train_test_split = True

    def add(self, layer):
        self.layers.append(layer)
        self.__check_train_test_split(layer)

        self.reg_params += layer.reg_params
        self.train_params += layer.train_params
        self.updates.update(layer.updates)
        self.masks += layer.masks

    def __calc(self, train=True):
        assert type(self.layers[0]).__name__ == 'XY'

        input = self.layers[0].X
        output = self.layers[0].Y

        for layer in self.layers[1:]:
            if type(layer).__name__ in ['Dropout', 'BatchNormal']:
                input = layer(input, train=train)
            else:
                input = layer(input)

        prob_ys = input
        ys = tensor.argmax(prob_ys, axis=1)
        loss = get_loss(self.loss, prob_ys, output)
        return prob_ys, ys, loss

    def compile(self):

        # calc
        train_prob_ys, train_ys, train_loss = self.__calc(True)
        if self.train_test_split:
            predict_prob_ys, predict_ys, predict_loss = self.__calc(False)
        else:
            predict_prob_ys, predict_ys, predict_loss = train_prob_ys, train_ys, train_loss

        # l1, l2 loss
        l1_loss = get_regularization('l1', self.reg_params, self.l1)
        l2_loss = get_regularization('l2', self.reg_params, self.l2)
        # l2_loss = get_regularization('l2', self.layers[-1].train_params, self.l2)
        # l2_loss = get_regularization('l2', self.layers[-1].reg_params, self.l2)
        loss = [train_loss]
        if self.l1 > 0.:
            loss.append(l1_loss)
        if self.l2 > 0.:
            loss.append(l2_loss)



        # get updates
        updates = get_updates(self.optimizer, sum(loss), self.train_params, self.lr, self.max_norm)
        self.updates.update(updates)

        # train functions
        self.train = function(inputs=[self.layers[0].X, self.layers[0].Y, self.lr] + self.masks,
                              outputs=[train_loss, l1_loss, l2_loss, train_ys],
                              updates=updates,)

        # test functions
        self.predict = function(inputs=[self.layers[0].X, self.layers[0].Y] + self.masks,
                                outputs=[predict_loss, predict_ys])

    def to_json(self):
        layers_json = OrderedDict()
        for layer in self.layers:
            layers_json[str(layer)] = layer.to_json()

        config = {
            'model_json': {
                'l1': self.l1,
                'l2': self.l2,
                'loss': self.loss,
                'optimizer': self.optimizer,
                'seed': self.seed,
                'max_norm': self.max_norm,
            },
            'layers_json': layers_json
        }

        return config
