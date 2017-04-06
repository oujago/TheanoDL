# -*- coding: utf-8 -*-

from .sentence_getter import SentenceGetter
from ..base import Data


class SentenceProvider(Data):
    def __init__(self, corpus_name,
                 w2v_type='google', w2v_dim=300,
                 folder_num=None, test_idx=None, valid_idx=None,
                 lower_case=True, threshold=1, remove_punc=True, test_from='train',
                 maxlen=30, dtype='int32', padding='pre', truncating='pre', value=-1, **kwargs):
        """
        :param corpus_name: like
                'mr'

        :param getting_params: like
                {
                    'corpus_name': 'mr',
                    'lower_case': True,
                    "threshold": 1,
                    "w2v_type": 'google',
                    'w2v_dim': 300,
                    'folder_num': 10,
                    'test_idx': 0,
                    'valid_idx': 1,
                    'remove_punc': True,
                    'test_from': 'train',
                }

        :param precessing_params:
                {
                    'maxlen': 50,
                    'dtype': 'int32',
                    'padding': 'post',
                    'truncating': 'pre',
                    'value': 0.
                }
        """
        super(SentenceProvider, self).__init__(**kwargs)
        self.corpus_name = corpus_name
        self.getting_params = {
            'lower_case': lower_case,
            "threshold": threshold,
            "w2v_type": w2v_type,
            'w2v_dim': w2v_dim,
            'remove_punc': remove_punc,
            'test_from': test_from,
            'folder_num': folder_num,
            'test_idx': test_idx,
            'valid_idx': valid_idx,
        }
        self.precessing_params = {
            'maxlen': maxlen,
            'dtype': dtype,
            'padding': padding,
            'truncating': truncating,
            'value': value
        }
        self.data_cls = SentenceGetter(corpus_name=corpus_name, **self.getting_params)

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
