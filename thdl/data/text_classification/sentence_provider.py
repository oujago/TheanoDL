# -*- coding: utf-8 -*-

import numpy as np
from .sentence_getter import SentenceGetter
from .sentence_processor import SentenceProcessor
from ..base import Data


class SentenceProvider(Data):
    def __init__(self, shuffle=True, shuffle_seed=None, index_to_tag=None):
        """

        :param shuffle:
        :param shuffle_seed:
        :param index_to_tag:

        :return
            self.word_res = {
                'embeddings': embeddings,
                'index2word': index2word,
                'word2index': word2index,
                'train': train_indices,
                'valid': valid_indices,
                'test': test_indices
            }
            self.words_res['train'][i] represents the sentence i,
            self.words_res['train'][i][j] represents the sentence i's word j.

            self.tag_res = {
                "index2tag": index2item,
                "tag2index": item2index,
                'train': train_index_items,
                'test': test_index_items,
                'valid': valid_index_items
            }
            each value is corresponding to the sentence's tag
        """
        super(SentenceProvider, self).__init__(shuffle, shuffle_seed, index_to_tag)

        self.getter = None
        self.processor = None

        self.word_res = None
        self.tag_res = None

    def set_getter(self, getter):
        assert isinstance(getter, SentenceGetter)
        self.getter = getter

    def set_processor(self, processor):
        assert isinstance(processor, SentenceProcessor)
        self.processor = processor

    def to_json(self):
        base_config = super(SentenceProvider, self).to_json()
        base_config['getter'] = self.getter.to_json()
        base_config['processor'] = self.processor.to_json()
        return base_config

    def build(self):
        train_indices = None
        valid_indices = None

        # words
        words_res = self.getter.get_words()

        assert len(words_res['train']) > 0
        assert len(words_res['test']) > 0

        if len(words_res['valid']) == 0:
            train_word_res = words_res['train']
            if valid_indices is None or train_indices is None:
                train_length = len(train_word_res)
                valid_length = train_length // 10
                indices = list(range(train_length))
                np.random.seed(1234)
                indices = np.random.permutation(indices)
                train_indices = indices[valid_length:]
                valid_indices = indices[:valid_length]
            words_res['valid'] = [train_word_res[i] for i in valid_indices]
            words_res['train'] = [train_word_res[i] for i in train_indices]

        words_res['train'] = self.processor(words_res['train'])
        words_res['test'] = self.processor(words_res['test'])
        words_res['valid'] = self.processor(words_res['valid'])
        self.word_res = words_res

        # tags
        tags_res = self.getter.get_tags()

        assert len(tags_res['train']) > 0
        assert len(tags_res['test']) > 0

        if len(tags_res['valid']) == 0:
            train_tags_res = tags_res['train']
            if valid_indices is None or train_indices is None:
                train_length = len(train_tags_res)
                valid_length = train_length // 10
                indices = list(range(train_length))
                np.random.seed(1234)
                indices = np.random.permutation(indices)
                train_indices = indices[valid_length:]
                valid_indices = indices[:valid_length]
            tags_res['train'] = np.asarray([train_tags_res[i] for i in train_indices])
            tags_res['valid'] = np.asarray([train_tags_res[i] for i in valid_indices])
        self.tag_res = tags_res
        self.index_to_tag = tags_res['index2tag']

    def get_train_data(self):
        return self.word_res['train'], self.tag_res['train']

    def get_valid_data(self):
        return self.word_res['valid'], self.tag_res['valid']

    def get_test_data(self):
        return self.word_res['test'], self.tag_res['test']

    def get_embedding(self):
        return self.word_res['embeddings']

    def get_index_to_tag(self):
        return self.tag_res['index2tag']
