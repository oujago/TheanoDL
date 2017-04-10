# -*- coding: utf-8 -*-

from thdl.base import ThdlObj
from thdl.utils.data_nlp_processing import pad_sequences


class SentenceProcessor(ThdlObj):
    def __init__(self, maxlen=30, dtype='int32', padding='pre', truncating='pre', value=-1,
                 lower_case=True, threshold=1, remove_punc=True):
        self.maxlen = maxlen
        self.dtype = dtype
        self.padding = padding
        self.truncating = truncating
        self.value = value
        self.lower_case = lower_case
        self.threshold = threshold
        self.remove_punc = remove_punc

    def __call__(self, data):
        return self.preprocess(data)

    def preprocess(self, data):
        return pad_sequences(
            data,
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
            'value': self.value,
            'lower_case': self.lower_case,
            'threshold': self.threshold,
            'remove_punc': self.remove_punc,
        }
        return config
