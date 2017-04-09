# -*- coding: utf-8 -*-

from thdl.base import ThdlObj
from thdl.utils.data_nlp_processing import pad_sequences


class SentenceProcessor(ThdlObj):
    def __init__(self, maxlen=30, dtype='int32', padding='pre', truncating='pre', value=-1):
        self.maxlen = maxlen
        self.dtype = dtype
        self.padding = padding
        self.truncating = truncating
        self.value = value

    def preprocess(self, corpus):
        return pad_sequences(
            corpus,
            maxlen=self.maxlen,
            dtype=self.dtype,
            padding=self.padding,
            truncating=self.truncating,
            value=self.value,
        )

    def to_json(self):
        config = {
            'maxlen': self.maxlen,
            'dtype': self.dtype,
            'padding': self.padding,
            'truncating': self.truncating,
            'value': self.value
        }
        return config
