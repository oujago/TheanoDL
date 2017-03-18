# -*- coding: utf-8 -*-

"""
@author: ChaoMing (www.oujago.com)

@date: Created on 2017/3/18

@notes:
    
"""


from thdl.data import w2v


def google():
    w2v.reformat_google(origin_path="E:/data/NLP/google/GoogleNews-vectors-negative300.bin.gz",
                        index_save_path="E:/data/NLP/google/google_word2vec_index.pkl",
                        data_save_path='E:/data/NLP/google/google_word2vec_data.txt')


def glove():
    pass


if __name__ == '__main__':
    google()
