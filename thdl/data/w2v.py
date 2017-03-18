# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2016/11/8

@notes:
    
"""

import numpy as np

from ..tool import get_config
from ..tool import pickle_dump
from ..tool import pickle_load
from ..tool import set_w2v


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


def reformat_glove(origin_path, w2v_data_path, w2v_index_path, w2v_type, precision=12):
    # write
    word2idx = {}
    i = 0
    length = 0
    w2v_dim = 0
    with open(origin_path, encoding='utf-8') as origin_f:
        with open(w2v_data_path, 'wb') as write_f:
            for line in origin_f:
                if line[-1] == '\n':
                    line = line[:-1]
                splits = line.split(" ")
                if w2v_dim == 0:
                    w2v_dim = len(splits[1:])
                    length = precision * w2v_dim + 1
                word = splits[0]

                if len(splits[1:]) != w2v_dim:
                    raise ValueError("Length don't match. actual %d, but this line %s. \n%s\n%s" % (
                        w2v_dim, len(splits[1:]), str(splits[1:]), line))
                vector = ''.join([" " * (precision - len(v)) + v for v in splits[1:]])
                max_len = max([len(v) for v in splits[1:]])
                if max_len >= precision:
                    raise ValueError("Precision %d is smaller than real value length: %d.\n %s" % (
                        precision, max_len, str(line)))
                if word in word2idx:
                    print("Word %s occur twice." % word)
                    print(str(line))
                else:
                    word2idx[word] = i
                    write_f.write(vector.encode("utf-8") + b"\n")
                    i += 1

    w2v_dim = w2v_dim
    w2v_type = w2v_type
    vec_len = length
    pickle_dump([w2v_type, w2v_dim, vec_len, word2idx], w2v_index_path)

    # check
    with open(w2v_data_path, 'rb') as read_f:
        for word, idx in word2idx.items():
            read_f.seek(idx * vec_len, 0)
            line = read_f.read(vec_len).decode('utf-8')
            [float(a) for a in line.strip().split()]

    print("No wrong")


def reformat_google(origin_path, data_save_path, index_save_path, precision=10):
    """

    :param origin_path: The original google word2vec file path
    :param data_save_path:
    :param index_save_path:
    :param precision:
    :return:
    """

    import os
    from gensim.models import word2vec

    print("Read original file.")
    temp_path = os.path.join(os.path.split(origin_path)[0], 'temp.txt')
    if not os.path.exists(temp_path):
        print("Convert original file ...")
        model = word2vec.Word2Vec.load_word2vec_format(origin_path, binary=True)
        model.save_word2vec_format(temp_path, binary=False)
        print("Convert work is done.")

    word2idx = {}
    i = 0
    length = 0
    w2v_dim = 0

    print("Reading data ... ")
    # with open(origin_path, mode='rb') as origin_f:
    with open(temp_path, encoding='utf-8', mode='r') as origin_f:
        origin_f.readline()
        with open(data_save_path, 'wb') as write_f:
            for line in origin_f:
                if line[-1] == '\n':
                    line = line[:-1]
                splits = line.split(" ")
                if w2v_dim == 0:
                    w2v_dim = len(splits[1:])
                    length = precision * w2v_dim + 1
                word = splits[0]

                if len(splits[1:]) != w2v_dim:
                    raise ValueError("Length don't match. actual %d, but this line %s. \n%s\n%s" % (
                        w2v_dim, len(splits[1:]), str(splits[1:]), line))
                vector = ''.join([" " * (precision - len(v)) + v for v in splits[1:]])
                max_len = max([len(v) for v in splits[1:]])
                if max_len >= precision:
                    raise ValueError("Precision %d is smaller than real value length: %d.\n %s" % (
                        precision, max_len, str(line)))
                if word in word2idx:
                    print("Word %s occur twice." % word)
                    print(str(line))
                else:
                    word2idx[word] = i
                    write_f.write(vector.encode("utf-8") + b"\n")
                    i += 1

                if (i + 1) % 10000 == 0:
                    print("Write %d words." % (i + 1))
    print("Rewritten work is done.")

    w2v_dim = w2v_dim
    w2v_type = 'google'
    vec_len = length
    pickle_dump([w2v_type, w2v_dim, vec_len, word2idx], index_save_path)
    print("Pickled index into the file.")

    print("Set configuration ...")
    set_w2v('google', w2v_index=index_save_path, w2v_data=data_save_path)
