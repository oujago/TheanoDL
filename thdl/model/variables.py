# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2016/11/5

@notes:
    
"""

from theano import tensor

from .initializations import get_shared

dtype = tensor.config.floatX


def get_clstm_variables(rng, n_in, n_out, init, inner_init):
    """
    Coupled LSTM
    """
    i_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    i_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    i_h_b = get_shared(rng, size=(n_out,), init='zero')

    g_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    g_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    g_h_b = get_shared(rng, size=(n_out,), init='zero')

    o_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    o_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    o_h_b = get_shared(rng, size=(n_out,), init='zero')

    return i_x2h_W, i_h2h_W, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, o_h_b


def get_lstm_variables(rng, n_in, n_out, init, inner_init):
    """
    Standard LSTM
    """
    f_x2h_W = get_shared(rng, (n_in, n_out), init=init)
    f_h2h_W = get_shared(rng, (n_out, n_out), init=inner_init)
    f_h_b = get_shared(rng, size=(n_out,), init='one')

    i_x2h_W, i_h2h_W, i_h_b, \
    g_x2h_W, g_h2h_W, g_h_b, \
    o_x2h_W, o_h2h_W, o_h_b = get_clstm_variables(rng, n_in, n_out, init, inner_init)

    return f_x2h_W, f_h2h_W, f_h_b, \
           i_x2h_W, i_h2h_W, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, o_h_b


def get_plstm_variables(rng, n_in, n_out, init, inner_init, peephole_init):
    """
    Peephole LSTM
    """
    f_x2h_W, f_h2h_W, f_h_b, \
    i_x2h_W, i_h2h_W, i_h_b, \
    g_x2h_W, g_h2h_W, g_h_b, \
    o_x2h_W, o_h2h_W, o_h_b = get_lstm_variables(rng, n_in, n_out, init, inner_init)

    p_f = get_shared(rng, (n_out,), init=peephole_init)
    p_i = get_shared(rng, (n_out,), init=peephole_init)
    p_o = get_shared(rng, (n_out,), init=peephole_init)

    return f_x2h_W, f_h2h_W, p_f, f_h_b, \
           i_x2h_W, i_h2h_W, p_i, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, p_o, o_h_b


def get_gru_variables(rng, n_in, n_out, init, inner_init):
    r_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    r_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    r_h_b = get_shared(rng, size=(n_out,), init='zero')

    z_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    z_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    z_h_b = get_shared(rng, size=(n_out,), init='zero')

    f_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    f_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    f_h_b = get_shared(rng, size=(n_out,), init='zero')

    return r_x2h_W, r_h2h_W, r_h_b, \
           z_x2h_W, z_h2h_W, z_h_b, \
           f_x2h_W, f_h2h_W, f_h_b


def get_mgu_variables(rng, n_in, n_out, init, inner_init):
    f_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    f_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    f_h_b = get_shared(rng, size=(n_out,), init='zero')

    i_x2h_W = get_shared(rng, size=(n_in, n_out), init=init)
    i_h2h_W = get_shared(rng, size=(n_out, n_out), init=inner_init)
    i_h_b = get_shared(rng, size=(n_out,), init='zero')

    return f_x2h_W, f_h2h_W, f_h_b, \
           i_x2h_W, i_h2h_W, i_h_b


def get_rnn_variables(rng, n_in, n_out, init, inner_init):
    x2o = get_shared(rng, size=(n_in, n_out), init=init)
    o2o = get_shared(rng, size=(n_out, n_out), init=inner_init)
    b = get_shared(rng, size=(n_out,), init='zero')

    return x2o, o2o, b
