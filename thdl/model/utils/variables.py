# -*- coding: utf-8 -*-

from ..initialization import One
from ..initialization import Zero


def get_clstm_variables(n_in, n_out, init, inner_init):
    """
    Coupled LSTM
    """
    zero = Zero()

    i_x2h_W = init(size=(n_in, n_out))
    i_h2h_W = inner_init(size=(n_out, n_out))
    i_h_b = zero(size=(n_out,))

    g_x2h_W = init(size=(n_in, n_out))
    g_h2h_W = inner_init(size=(n_out, n_out))
    g_h_b = zero(size=(n_out,))

    o_x2h_W = init(size=(n_in, n_out))
    o_h2h_W = inner_init(size=(n_out, n_out))
    o_h_b = zero(size=(n_out,))

    return i_x2h_W, i_h2h_W, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, o_h_b


def get_lstm_variables(n_in, n_out, init, inner_init):
    """
    Standard LSTM
    """
    f_x2h_W = init(size=(n_in, n_out))
    f_h2h_W = inner_init(size=(n_out, n_out))
    f_h_b = One()(size=(n_out,))

    i_x2h_W, i_h2h_W, i_h_b, \
    g_x2h_W, g_h2h_W, g_h_b, \
    o_x2h_W, o_h2h_W, o_h_b = get_clstm_variables(n_in, n_out, init, inner_init)

    return f_x2h_W, f_h2h_W, f_h_b, \
           i_x2h_W, i_h2h_W, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, o_h_b


def get_plstm_variables(n_in, n_out, init, inner_init, peephole_init):
    """
    Peephole LSTM
    """
    f_x2h_W, f_h2h_W, f_h_b, \
    i_x2h_W, i_h2h_W, i_h_b, \
    g_x2h_W, g_h2h_W, g_h_b, \
    o_x2h_W, o_h2h_W, o_h_b = get_lstm_variables(n_in, n_out, init, inner_init)

    p_f = peephole_init(size=(n_out,))
    p_i = peephole_init(size=(n_out,))
    p_o = peephole_init(size=(n_out,))

    return f_x2h_W, f_h2h_W, p_f, f_h_b, \
           i_x2h_W, i_h2h_W, p_i, i_h_b, \
           g_x2h_W, g_h2h_W, g_h_b, \
           o_x2h_W, o_h2h_W, p_o, o_h_b


def get_gru_variables(n_in, n_out, init, inner_init):
    zero = Zero()

    r_x2h_W = init(size=(n_in, n_out))
    r_h2h_W = inner_init(size=(n_out, n_out))
    r_h_b = zero(size=(n_out,))

    z_x2h_W = init(size=(n_in, n_out))
    z_h2h_W = inner_init(size=(n_out, n_out))
    z_h_b = zero(size=(n_out,))

    f_x2h_W = init(size=(n_in, n_out))
    f_h2h_W = inner_init(size=(n_out, n_out))
    f_h_b = zero(size=(n_out,))

    return r_x2h_W, r_h2h_W, r_h_b, \
           z_x2h_W, z_h2h_W, z_h_b, \
           f_x2h_W, f_h2h_W, f_h_b


def get_mgu_variables(n_in, n_out, init, inner_init):
    zero = Zero()

    f_x2h_W = init(size=(n_in, n_out))
    f_h2h_W = inner_init(size=(n_out, n_out))
    f_h_b = zero(size=(n_out,))

    i_x2h_W = init(size=(n_in, n_out))
    i_h2h_W = inner_init(size=(n_out, n_out))
    i_h_b = zero(size=(n_out,))

    return f_x2h_W, f_h2h_W, f_h_b, \
           i_x2h_W, i_h2h_W, i_h_b


def get_rnn_variables(n_in, n_out, init, inner_init):
    x2o = init(size=(n_in, n_out))
    o2o = inner_init(size=(n_out, n_out))
    b = Zero()(size=(n_out,))

    return x2o, o2o, b
