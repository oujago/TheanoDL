# -*- coding: utf-8 -*-


import numpy as np

from thdl.utils.config import get_config
from thdl.utils.file import pickle_load


class W2VGet:
    """
    Get Word2Vec that has build the index and reformat the f_data
    """

    def __init__(self, w2v_type, w2v_dim=300):

        self.w2v_type = w2v_type
        self.w2v_dim = w2v_dim

        if w2v_type in get_config()['data']['w2v']:
            pkl_w2v_type, pkl_w2v_dim, self.vec_len, self.w2v_idx = pickle_load(
                get_config()['data']['w2v'][w2v_type][0])
            self.w2v_data_file = open(get_config()['data']['w2v'][w2v_type][1], 'rb')

        else:
            raise ValueError('unknown w2v_type: %s' % w2v_type)

        if pkl_w2v_type.lower() != w2v_type.lower() or pkl_w2v_dim != w2v_dim:
            raise ValueError("pkl_w2v_type != w2v_type or pkl_w2v_dim != w2v_dim")
        self.w2v_out = []
        self.word_len = len(self.w2v_idx)

    def find_word(self, word):
        """
        Find the vector of this word. Maybe it is in the word2vec file,
        maybe it's in the unknown word2vec list.
        :param word:
        :return: the embedding vector corresponding to the word.
        """
        # word out of w2v
        if word not in self.w2v_idx:
            raise ValueError("word not in this word2vec: %s" % word)

        index = self.w2v_idx[word]
        if index >= self.word_len:
            vector = self.w2v_out[index - self.word_len]
        else:
            self.w2v_data_file.seek(index * self.vec_len, 0)
            line = self.w2v_data_file.read(self.vec_len).decode('utf-8')
            try:
                vector = np.asarray([float(a) for a in line.strip().split()], dtype=np.float32)
            except:
                runout = "The vec_length [%d] is not correct! Please find " \
                         "the correct vec_length. \n" % self.vec_len
                runout += "The error sentence is [ %s ]" % str(line)
                raise ValueError(runout)

        return vector

    def add_word(self, word, vector=None, low=-2., high=2.):
        """
        Add the unknown word into the total Word2Vec
        :param word:
        :param vector: if None, then produce a random vector for this word.
        :param low: if vector is None, then it is the random low boundary.
        :param high: if vector is None, then it is the random high boundary.
        """
        if word not in self.w2v_idx:
            self.w2v_idx[word] = len(self.w2v_idx)
            if vector is None:
                vector = np.random.uniform(low=low, high=high, size=(self.w2v_dim,))
            self.w2v_out.append(np.asarray(vector, dtype=np.float32))
        else:
            print("word %s already in this word2vec" % word)
