# -*- coding: utf-8 -*-


from theano import shared

from .base import Layer
from ..initialization import get_shared


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
