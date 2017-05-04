# -*- coding:utf-8 -*-


import os
import pickle

import numpy as np

from thdl.data.base import Data

_UNKNOWN = "UNKNOWN_VOCAB"
_ZERO = 'ZERO_VOCAB'


class AnswerSelectionData(Data):
    def __init__(self, xs_path, ys_path, maxlen, vocab_path=None, threshold=None,
                 valid_split=0.1, test_split=0.1, total_len = -1,
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

    def to_json(self):
        config = {
            "xs_path": self.xs_path,
            'ys_path': self.ys_path,
            "maxlen": self.maxlen,
            "vocab_path": self.vocab_path,
            "threshold": self.threshold,
        }
        return config

    def build(self):
        # load vocabulary
        if self.vocab_path is None:
            print("Building vocabulary ...")
            vocab_path = './f_data/vocab_thre-{}.pkl'.format(self.threshold)

            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as fin:
                    self.idx_to_vocab, self.vocab_to_idx = pickle.load(fin)
            else:
                threshold = 1 if self.threshold is None else self.threshold

                freq_vocab = {}
                with open(os.path.join(os.getcwd(), self.xs_path), encoding='utf-8') as fin:
                    for line in fin:
                        sentences = line.strip().split()
                        for word in sentences:
                            freq_vocab[word] = freq_vocab.get(word, 0) + 1
                sorted_vocab = [word for word, freq in sorted(freq_vocab.items()) if freq >= threshold]
                sorted_vocab.append(_UNKNOWN)
                sorted_vocab.append(_ZERO)
                self.idx_to_vocab = {i: vocab for i, vocab in enumerate(sorted_vocab)}
                self.vocab_to_idx = {vocab: i for i, vocab in enumerate(sorted_vocab)}
                with open(vocab_path, 'wb') as fin:
                    pickle.dump([self.idx_to_vocab, self.vocab_to_idx], fin)
        else:
            with open(os.path.join(os.getcwd(), self.vocab_path), 'rb') as fin:
                self.idx_to_vocab, self.vocab_to_idx = pickle.load(fin)

        pkl_path = "./f_data/thre-{}-valid-{}-test-{}-total-{}.pkl".format(
            self.threshold, self.valid_split, self.test_split, self.total_len)
        print("Building data ...")

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fin:
                self.all_xs, self.all_xs, \
                self._train_start, self._train_end, \
                self._valid_start, self._valid_end, \
                self._test_start, self._test_end = pickle.load(fin)
        else:
            # load xs in index
            all_xs = []
            with open(os.path.join(os.getcwd(), self.xs_path), encoding='utf-8') as fin:
                i = 0
                for line in fin:
                    sentences = line.strip().split("\t")
                    pair_idxs = []
                    for sentence in sentences:
                        sen_idxs = []
                        for word in sentence.split():
                            sen_idxs.append(self.vocab_to_idx.get(word, -2))

                        if len(sen_idxs) == 0:
                            raise ValueError
                        if len(sen_idxs) > self.maxlen:
                            sen_idxs = sen_idxs[:self.maxlen]
                        else:
                            sen_idxs = [-1] * (self.maxlen - len(sen_idxs)) + sen_idxs
                        assert len(sen_idxs) == self.maxlen
                        pair_idxs.append(sen_idxs)
                    all_xs.append(pair_idxs)

                    i += 1
                    if i == self.total_len:
                        break

            self.all_xs = np.asarray(all_xs, dtype='int32')

            # load ys
            with open(os.path.join(os.getcwd(), self.ys_path), 'rb') as fin:
                idx_all_ys = np.asarray(pickle.load(fin), dtype='int32')
                if self.total_len > 0:
                    idx_all_ys = idx_all_ys[:self.total_len]

            self.all_ys = np.zeros((idx_all_ys.shape[0], 2))
            for i in range(2):
                self.all_ys[idx_all_ys == i, i] = 1

            # shuffle data
            self.shuffle_data(self.all_xs, self.all_ys)

            # the start and end of the valid and test splits
            total_len = idx_all_ys.shape[0]
            valid_len = total_len * self.valid_split
            test_len = total_len * self.test_split
            train_len = total_len - valid_len - test_len
            self._train_start, self._train_end = 0, train_len
            self._valid_start, self._valid_end = train_len, train_len + valid_len
            self._test_start, self._test_end = train_len + valid_len, train_len + valid_len + test_len

            # pickle
            with open(pkl_path, 'wb') as fin:
                dump_contents = [self.all_xs, self.all_xs,
                                 self._train_start, self._train_end,
                                 self._valid_start, self._valid_end,
                                 self._test_start, self._test_end]
                pickle.dump(dump_contents, fin)
        self.index_to_tag = self.get_index_to_tag()

    def get_index_to_tag(self):
        return "Different", "Similar"

    def get_train_data(self):
        return self.all_xs[self._train_start: self._train_end], \
               self.all_ys[self._train_start: self._train_end]

    def get_valid_data(self):
        return self.all_xs[self._valid_start: self._valid_end], \
               self.all_ys[self._valid_start: self._valid_end]

    def get_test_data(self):
        return self.all_xs[self._test_start: self._test_end], \
               self.all_ys[self._test_start: self._test_end]
