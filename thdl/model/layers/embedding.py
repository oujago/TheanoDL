# -*- coding: utf-8 -*-


from theano import shared

from thdl.model.layers.base import Layer
from thdl.model.initialization import Uniform


class Embedding(Layer):
    def __init__(self, embed_words=None, static=None,
                 input_size=None, n_out=None,
                 init=Uniform()):
        super(Embedding, self).__init__()

        # parameters
        self.init = init

        if embed_words is not None:
            if static is None:
                self.static = True
            else:
                self.static = static

            self.input_size, self.n_out = embed_words.shape
            self.embed_words = shared(embed_words, name='embedding_words')
        else:
            if static is None:
                self.static = False
            else:
                self.static = static
            self.input_size, self.n_out = input_size, n_out
            self.embed_words = init((input_size, n_out))

    def connect_to(self, pre_layer=None):
        assert pre_layer is None
        self.output_shape = (None, None, self.n_out)

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
        }
        return config

    @property
    def params(self):
        if self.static:
            return []
        else:
            return [self.embed_words, ]

