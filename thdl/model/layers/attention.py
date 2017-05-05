# -*- coding: utf-8 -*-

from theano import tensor

from thdl.model.layers.base import Layer
from thdl.model.nonlinearity import Tanh
from thdl.model.initialization import Orthogonal
from thdl.model.initialization import GlorotUniform
from thdl.model.initialization import _zero



class Attention(Layer):
    """Neural processes involving attention have been largely studied in 
    Neuroscience and Computational Neuroscience [1, 2]. A particularly 
    studied aspect is visual attention: many animals focus on specific 
    parts of their visual inputs to compute the adequate responses. This 
    principle has a large impact on neural computation as we need to select 
    the most pertinent piece of information, rather than using all available 
    information, a large part of it being irrelevant to compute the neural response.
    
    A similar idea -focusing on specific parts of the input- has been applied in 
    Deep Learning, for speech recognition, translation, reasoning, and visual 
    identification of objects.
    
    References
    ----------
    [1] Itti, Laurent, Christof Koch, and Ernst Niebur. « A model of 
    saliency-based visual attention for rapid scene analysis. » IEEE 
    Transactions on Pattern Analysis & Machine Intelligence 11 (1998): 1254-1259.

    [2] Desimone, Robert, and John Duncan. « Neural mechanisms of selective 
    visual attention. » Annual review of neuroscience 18.1 (1995): 193-222.
    
    """
    def __init__(self, n_in, activation=Tanh(),
                 init=Orthogonal(), context_init=GlorotUniform(),
                 att_W_regularizer=None, att_b_regularizer=None, context_regularizer=None):
        super(Attention, self).__init__()

        # parameters
        self.n_in = n_in
        self.n_out = n_in
        self.init = init
        self.context_init = context_init
        self.activation = activation
        self.att_W_regularizer = att_W_regularizer
        self.att_b_regularizer = att_b_regularizer
        self.context_regularizer = context_regularizer

        # variables
        self.att_W = self.init((n_in, n_in))
        self.att_b = _zero((n_in,))
        self.att_vec = self.context_init((n_in,))

    def forward(self, input, **kwargs):
        assert input.ndim == 3

        a = tensor.dot(self.activation(tensor.dot(input, self.att_W) + self.att_b), self.att_vec)
        b = tensor.nnet.softmax(a)
        c = input.dimshuffle(2, 0, 1) * b
        d = tensor.sum(c, axis=2)
        e = d.dimshuffle(1, 0)

        return e

    def to_json(self):
        config = {
            'n_in': self.n_in,
            'activation': self.activation,
            'init': self.init,
            'context_init': self.context_init,
            'att_W_regularizer': self.att_W_regularizer,
            'att_b_regularizer': self.att_b_regularizer,
            'context_regularizer': self.context_regularizer,
        }
        return config


    @property
    def params(self):
        return [self.att_W, self.att_b, self.att_vec]

    @property
    def regularizers(self):
        regs = []

        if self.att_W_regularizer:
            regs.append(self.att_W_regularizer(self.att_W))

        if self.att_b_regularizer:
            regs.append(self.att_b_regularizer(self.att_b))

        if self.context_regularizer:
            regs.append(self.context_regularizer(self.att_vec))

        return regs