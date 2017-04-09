# -*- coding: utf-8 -*-

from .sentence_getter import SentenceGetter
from ..base import Data


class SentenceProvider(Data):
    def __init__(self, shuffle=True, shuffle_seed=None, index_to_tag=None):
        super(SentenceProvider, self).__init__(shuffle, shuffle_seed, index_to_tag)

        self.getter = None
        self.processor = None

    def set_getter(self, getter):
        self.getter = getter

    def set_processor(self, processor):
        self.processor = processor

        self.valid_indices = None
        self.train_indices = None

    def get_xs(self):
        """
        :return: words_res
            {
                'embeddings': embeddings,
                'index2word': index2word,
                'word2index': word2index,
                'train': train_indices,
                'valid': valid_indices,
                'test': test_indices
            }

            words_res['train'][i] represents the sentence i,
            words_res['train'][i][j] represents the sentence i's word j.
        """
        words_res = self.data_cls.get_words(**self.getting_params)

        assert len(words_res['train']) > 0
        assert len(words_res['test']) > 0

        if len(words_res['valid']) == 0:
            train_word_res = words_res['train']
            if self.valid_indices is None or self.train_indices is None:
                train_length = len(train_word_res)
                valid_length = train_length // 10
                indices = list(range(train_length))
                np.random.seed(1234)
                indices = np.random.permutation(indices)
                self.train_indices = indices[valid_length:]
                self.valid_indices = indices[:valid_length]
            words_res['valid'] = [train_word_res[i] for i in self.valid_indices]
            words_res['train'] = [train_word_res[i] for i in self.train_indices]

        words_res['train'] = pad_sequences(words_res['train'], **self.precessing_params)
        words_res['test'] = pad_sequences(words_res['test'], **self.precessing_params)
        words_res['valid'] = pad_sequences(words_res['valid'], **self.precessing_params)
        return words_res

    def get_ys(self):
        """
        :return: tags_res example ---
            {
                index2item_name: index2item,
                item2index_name: item2index,
                'train': train_index_items,
                'test': test_index_items,
                'valid': valid_index_items
            }

            each value is corresponding to the sentence's tag
        """
        tags_res = self.data_cls.get_tags(**self.getting_params)

        assert len(tags_res['train']) > 0
        assert len(tags_res['test']) > 0

        if len(tags_res['valid']) == 0:
            train_tags_res = tags_res['train']
            if self.valid_indices is None or self.train_indices is None:
                train_length = len(train_tags_res)
                valid_length = train_length // 10
                indices = list(range(train_length))
                np.random.seed(1234)
                indices = np.random.permutation(indices)
                self.train_indices = indices[valid_length:]
                self.valid_indices = indices[:valid_length]
            tags_res['train'] = [train_tags_res[i] for i in self.train_indices]
            tags_res['valid'] = [train_tags_res[i] for i in self.valid_indices]

        return tags_res

    def get_index2tag(self):
        return self.get_ys()['index2tag']

    def to_json(self):
        config = {'corpus_name': self.corpus_name, }
        config.update(self.getting_params)
        config.update(self.precessing_params)
        return config



