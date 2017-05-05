# -*- coding:utf-8 -*-


import os
import pickle

import numpy as np

from thdl.utils.random import get_dtype
from thdl.data.base import Data

_UNKNOWN = "UNKNOWN_VOCAB"
_ZERO = 'ZERO_VOCAB'


class AnswerSelectionData(Data):
    def __init__(self, xs_path, ys_path, maxlen, vocab_path, threshold=None,
                 valid_split=0.1, test_split=0.1, total_len = -1, prefix=None,
                 batch_size = 40,
                 **kwargs):
        super(AnswerSelectionData, self).__init__(**kwargs)

        self.xs_path = xs_path
        self.ys_path = ys_path
        self.maxlen = maxlen
        self.vocab_path = vocab_path
        self.threshold = threshold
        self.valid_split = valid_split
        self.test_split = test_split
        self.total_len = total_len
        self.prefix = prefix
        self.batch_size= batch_size

    def to_json(self):
        config = {
            "xs_path": self.xs_path,
            'ys_path': self.ys_path,
            "maxlen": self.maxlen,
            "vocab_path": self.vocab_path,
            "threshold": self.threshold,
            "prefix": self.prefix,
            "batch_size": self.batch_size,
        }
        return config

    def build(self):
        # load vocabulary
        with open(self.vocab_path, 'rb') as fin:
            vocabs_freqs = pickle.load(fin)
        threshold = 1 if self.threshold is None else self.threshold

        sorted_vocab = [word for word, freq in sorted(vocabs_freqs.items()) if freq >= threshold]
        sorted_vocab.append(_UNKNOWN)
        sorted_vocab.append(_ZERO)
        self.idx_to_vocab = {i: vocab for i, vocab in enumerate(sorted_vocab)}
        self.vocab_to_idx = {vocab: i for i, vocab in enumerate(sorted_vocab)}

        pkl_path = "./f_data/prefix-{}-thre-{}-valid-{}-test-{}-total-{}.pkl".format(
            self.prefix, self.threshold, self.valid_split, self.test_split, self.total_len)
        print("Building data ...")

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fin:
                self.all_xs, self.all_ys, \
                self._train_start, self._train_end, \
                self._valid_start, self._valid_end, \
                self._test_start, self._test_end = pickle.load(fin)
        else:
            # load xs in index
            all_xs = []
            remove_idxs = []
            with open(os.path.join(os.getcwd(), self.xs_path), encoding='utf-8') as fin:
                i = 0
                for line in fin:
                    sentences = [sent.strip().split() for sent in line.strip().split("\t")]
                    if len(sentences) != 2:
                        print("Not A Pair: {}".format(line))
                        remove_idxs.append(i)
                    else:
                        all_xs.append(sentences)
                        i += 1
                        if i == self.total_len:
                            break
            self.all_xs = all_xs

            # load ys
            with open(os.path.join(os.getcwd(), self.ys_path), 'rb') as fin:
                idx_all_ys = pickle.load(fin)
                if self.total_len > 0:
                    idx_all_ys = idx_all_ys[:self.total_len]
                for i in remove_idxs:
                    idx_all_ys.pop(i)
                idx_all_ys = np.asarray(idx_all_ys, dtype='int32')

            self.all_ys = np.zeros((idx_all_ys.shape[0], 2), dtype=get_dtype())
            for i in range(2):
                self.all_ys[idx_all_ys == i, i] = 1

            if self.total_len == -1:
                self.total_len = len(all_xs)

            # shuffle data
            self.shuffle_data(self.all_xs, self.all_ys)

            # the start and end of the valid and test splits
            valid_len = int(self.total_len * self.valid_split)
            test_len = int(self.total_len * self.test_split)
            train_len = int(self.total_len * (1 - self.valid_split - self.test_split))
            self._train_start, self._train_end = 0, train_len
            self._valid_start, self._valid_end = train_len, train_len + valid_len
            self._test_start, self._test_end = train_len + valid_len, train_len + valid_len + test_len

            # pickle
            with open(pkl_path, 'wb') as fin:
                dump_contents = [self.all_xs, self.all_ys,
                                 self._train_start, self._train_end,
                                 self._valid_start, self._valid_end,
                                 self._test_start, self._test_end]
                pickle.dump(dump_contents, fin)

        assert len(self.all_xs) == len(self.all_ys)
        self.index_to_tag = self.get_index_to_tag()

    def get_index_to_tag(self):
        return ["Different", "Similar"]

    def _get_data(self, start, end):
        xs = self.all_xs[start: end]
        ys = self.all_ys[start: end]

        epochs = len(xs) // self.batch_size
        length = self.batch_size * epochs

        return xs[:length], ys[:length]

    def get_train_data(self):
        return self._get_data(self._train_start, self._train_end)


    def get_valid_data(self):
        return self._get_data(self._valid_start, self._valid_end)

    def get_test_data(self):
        return self._get_data(self._test_start, self._test_end)

    def batch_pairs_to_idxs(self, pairs):
        error = 0

        all_pairs_idxs = []
        for pair in pairs:
            a_pair_idxs = []
            if len(pair) != 2:
                print("Not A Pair: {}".format(str(pair)))
                pair.append([])

            for sent in pair:
                # get idxs
                sent_idxs = [self.vocab_to_idx.get(w, -2) for w in sent]
                # sentence length
                length = len(sent_idxs)
                if length < self.maxlen:
                    # padding
                    sent_idxs= [-1] * (self.maxlen - length) + sent_idxs
                else:
                    # truncate
                    sent_idxs= sent_idxs[:self.maxlen]

                assert len(sent_idxs) == self.maxlen
                a_pair_idxs.append(sent_idxs)

            assert len(a_pair_idxs) == 2
            all_pairs_idxs.append(a_pair_idxs)


        try:
            all_pairs_idxs = np.asarray(all_pairs_idxs, dtype='int32')
            return all_pairs_idxs
        except ValueError:
            print(all_pairs_idxs)
            raise ValueError

