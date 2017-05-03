# -*- coding: utf-8 -*-


from theano import shared
from theano import tensor

from thdl.model.initialization import Uniform
from thdl.model.initialization import _zero
from thdl.model.layers.base import Layer
from thdl.utils import is_iterable


class Embedding(Layer):
    def __init__(self, embed_words=None, static=None,
                 input_size=None, n_out=None,
                 init=Uniform(), zero_idxs=None):
        super(Embedding, self).__init__()

        # parameters
        self.init = init

        # embedding
        if embed_words is not None:
            if static is None:
                self.static = True
            else:
                self.static = static

            self.input_size, self.n_out = embed_words.shape
        else:
            if static is None:
                self.static = False
            else:
                self.static = static
            self.input_size, self.n_out = input_size, n_out
            embed_words = init((input_size, n_out), theano_shared=False)

        # zero indexes
        if zero_idxs is not None:
            if isinstance(zero_idxs, int):
                zero_idxs = [zero_idxs]
            assert is_iterable(zero_idxs)
            for idx in zero_idxs:
                embed_words[idx] = 0.
        self.zero_idxs = zero_idxs

        self.embed_words = shared(embed_words, name='embedding_words')


    def forward(self, input, **kwargs):
        assert input.ndim == 2

        shape = (input.shape[0], input.shape[1], self.embed_words.shape[1])
        return self.embed_words[input.flatten()].reshape(shape)

    def to_json(self):
        config = {
            'static': self.static,
            'input_size': self.input_size,
            'n_out': self.n_out,
            'init': self.init,
            'zero_idxs': self.zero_idxs,
        }
        return config

    @property
    def params(self):
        if self.static:
            return []
        else:
            return [self.embed_words, ]

    @property
    def updates(self):
        ups = super(Embedding, self).updates()
        if self.static is False:
            zero_vec = _zero((self.n_out, ))
            for idx in self.zero_idxs:
            # zero_mat = _zero((len(self.zero_idxs), self.n_out))
                ups[self.embed_words] = tensor.set_subtensor(self.embed_words[idx], zero_vec)
        return ups
