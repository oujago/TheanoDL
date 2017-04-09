# -*- coding: utf-8 -*-


from thdl.data.text_classification import movie_review_corpus
from thdl.data.text_classification import stanford_sentiment_treebank_phrase
from thdl.data.text_classification import subjective_corpus
from thdl.data.text_classification import trec_corpus


def mr():
    file_folder = 'files/mr'
    save_path = "files/handled/mr.data"
    movie_review_corpus(file_folder, save_path)


def subjective():
    file_folder = "files/subjective"
    save_path = 'files/handled/subjective.data'
    subjective_corpus(file_folder, save_path)


def sst_phrase():
    file_folder = 'files/sst'
    save_path = 'files/handled/sst_phrase.data'
    stanford_sentiment_treebank_phrase(file_folder, save_path)


def trec():
    file_folder = 'files/trec'
    save_path = 'files/handled/trec.data'
    trec_corpus(file_folder, save_path)


if __name__ == '__main__':
    # subjective()
    sst_phrase()
    trec()
