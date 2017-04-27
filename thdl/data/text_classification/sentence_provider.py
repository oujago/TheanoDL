# -*- coding: utf-8 -*-

import os
import numpy as np
from .sentence_getter import SentenceGetter
from .sentence_processor import SentenceProcessor
from ..base import Data
from thdl.utils.common import dict_to_str
from thdl.utils.file import pickle_dump
from thdl.utils.file import pickle_load


class SentenceProvider(Data):
    def __init__(self, shuffle=True, shuffle_seed=None, index_to_tag=None, save_path=None):
        """

        :param save_path: If None, do not save the data into the file. Else, save the data into file.

        :return
            self.word_res = {
                'embeddings': embeddings,
                'index2word': index2word,
                'word2index': word2index,
                'train': train_indices,
                'valid': valid_indices,
                'tests': test_indices
            }
            self.words_res['train'][i] represents the sentence i,
            self.words_res['train'][i][j] represents the sentence i's word j.

            self.tag_res = {
                "index2tag": index2item,
                "tag2index": item2index,
                'train': train_index_items,
                'tests': test_index_items,
                'valid': valid_index_items
            }
            each value is corresponding to the sentence's tag
        """
        super(SentenceProvider, self).__init__(shuffle, shuffle_seed, index_to_tag)

        self.getter = None
        self.processor = None

        self.word_res = None
        self.tag_res = None

        self.save_path = save_path

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
        if self.save_path is None:
            to_save_filepath = None

        else:
            save_file = "{}.pkl".format(dict_to_str(self.to_json()))
            to_save_filepath = os.path.join(os.getcwd(), self.save_path, save_file)
            if os.path.exists(to_save_filepath):
                self.word_res, self.tag_res, self.index_to_tag = pickle_load(save_file)
                return

        train_indices = None
        valid_indices = None

        # words
        words_res = self.getter.get_words()

        assert len(words_res['train']) > 0
        assert len(words_res['tests']) > 0

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
        words_res['tests'] = self.processor(words_res['tests'])
        words_res['valid'] = self.processor(words_res['valid'])
        self.word_res = words_res

        # tags
        tags_res = self.getter.get_tags()

        assert len(tags_res['train']) > 0
        assert len(tags_res['tests']) > 0

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

        if to_save_filepath is None:
            return
        else:
            res = [self.word_res, self.tag_res, self.index_to_tag]
            pickle_dump(res, to_save_filepath)
            return

    def get_train_data(self):
        return self.word_res['train'], self.tag_res['train']

    def get_valid_data(self):
        return self.word_res['valid'], self.tag_res['valid']

    def get_test_data(self):
        return self.word_res['tests'], self.tag_res['tests']

    def get_embedding(self):
        return self.word_res['embeddings']

    def get_index_to_tag(self):
        return self.tag_res['index2tag']
