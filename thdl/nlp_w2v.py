# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2016/11/8

@notes:
    
"""

import platform
import sys

import numpy as np

from .common import pickle_load
from .common import pickle_dump

platform_name = sys.platform.lower()
computer_name = platform.node().lower()


class W2VPath:
    if platform_name == 'win32':
        if computer_name == 'chaoming-iscas-pc':
            pass
        elif computer_name == 'chaoming-pc':
            # google word2vec path in windows
            google_300_idx_path = 'D:/My Data/NLP/google/google_news_word2vec_indexes.pkl'
            google_300_data_path = 'D:/My Data/NLP/google/google_news_word2vec.txt'
        else:
            raise ValueError("Unknown computer")

    elif platform_name == 'linux':
        if computer_name == 'chaoming-pc':
            # google word2vec path in linux
            google_300_idx_path = '/media/chaoming/Files/data/NLP/google/google_news_word2vec_indexes.pkl'
            google_300_data_path = '/media/chaoming/Files/data/NLP/google/google_news_word2vec.txt'
        elif computer_name == 'bdi4-server':
            # google word2vec path in linux
            google_300_idx_path = '/media/data/chaoming/software/Google/google_news_word2vec_indexes.pkl'
            google_300_data_path = '/media/data/chaoming/software/Google/google_news_word2vec.txt'

            # glove word2vec path in linux
            glove_idx_root_path = '/media/data/chaoming/software/glove/%s-%sd-indexes.pkl'
            glove_data_root_path = '/media/data/chaoming/software/glove/%s-%sd.txt'
        else:
            raise ValueError("Unknown Linux computer.")

    else:
        raise Exception("Unknown Operating System.")


class W2VGet:
    """
    Get Word2Vec that has build the index and reformat the f_data
    """

    def __init__(self, w2v_type='Google', w2v_dim=300):

        self.w2v_type = w2v_type
        self.w2v_dim = w2v_dim

        if w2v_type.lower() == 'google':
            pkl_w2v_type, pkl_w2v_dim, self.vec_len, self.w2v_idx = pickle_load(W2VPath.google_300_idx_path)
            self.w2v_data_file = open(W2VPath.google_300_data_path, 'rb')

        elif 'glove' in w2v_type:
            pkl_w2v_type, pkl_w2v_dim, self.vec_len, self.w2v_idx = pickle_load(W2VPath.glove_idx_root_path % (w2v_type, w2v_dim))
            self.w2v_data_file = open(W2VPath.glove_data_root_path % (w2v_type, w2v_dim), 'rb')
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


def get_glove(origin_path, w2v_data_path, w2v_index_path, w2v_type, precision=12):
    # variable

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


def get_google_news():
    precision = 10
    origin_path = '/media/data/chaoming/software/Google/GoogleNews-vectors-negative300.txt'
    w2v_data_path = '/media/data/chaoming/software/Google/google_news_word2vec.txt'
    w2v_index_path = '/media/data/chaoming/software/Google/google_news_word2vec_indexes.pkl'

    word2idx = {}
    i = 0
    length = 0
    w2v_dim = 0
    with open(origin_path, encoding='utf-8') as origin_f:
        origin_f.readline()
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
    w2v_type = 'google'
    vec_len = length
    pickle_dump([w2v_type, w2v_dim, vec_len, word2idx], w2v_index_path)

    # check
    with open(w2v_data_path, 'rb') as read_f:
        for word, idx in word2idx.items():
            read_f.seek(idx * vec_len, 0)
            line = read_f.read(vec_len).decode('utf-8')
            [float(a) for a in line.strip().split()]

    print("No wrong")


if __name__ == "__main__":

    get_google_news()
