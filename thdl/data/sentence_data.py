# -*- coding: utf-8 -*-

"""
@author: ChaoMing (https://oujago.github.io/)

@date: Created on 2017/3/18

@notes:
    
"""

import os
from collections import Counter

import numpy as np
from nltk import word_tokenize

from .processing import get_split
from .processing import item_list2index_list
from .processing import yield_item
from .w2v import W2VGet
from ..tool import pickle_dump
from ..tool import pickle_load


class SentenceCorpus:
    def __init__(self, corpus_name, folder_num=None, test_idx=None, valid_idx=None, **kwargs):
        """
        If folder_num and test_idx are provided,
        then this is the [test_idx]-th folder of [folder_num] cross validation.

        'processed.data' File Format ——
            1. Each line is a sentence;
            2. The first split element is the true label;
            3. Next elements are words.
        For example:
            subjective	a remarkable 179-minute meditation on the nature of revolution .
            subjective	the film is small in scope , yet perfectly formed .
            objective	in return for a small plot of land , george agrees to search for princess lunna .
            objective	big-shot executive robert stiles' car is damaged when parked at the lodge .


        If folder_num and test_idx are all None,
        then this is the fixed train-valid-test split.

        'processed.data' File Format ——
            1. Each line is a sentence;
            2. First line is test-[test_start_idx]-[test_end_idx];
            3. Second line is valid-[valid_start_idx]-[valid_end_idx];
            4. Next each line is a sentence.
            5. The first split element is the split label,
            6. The second split element is the true sentence classification label,
            7. All next elements are words.
        For example:
            train	negative	The modern-day royals have nothing on these guys when it comes to scandals .
            train	negative	It 's only in fairy tales that princesses that are married for political reason live happily ever after .
            train	very_positive	A terrific B movie -- in fact , the best in recent memory .
            train	positive	`` Birthday Girl '' is an actor 's movie first and foremost .
            test	positive	It 's rather like a Lifetime special -- pleasant , sweet and forgettable .
            test	positive	A moody horror\/thriller elevated by deft staging and the director 's well-known narrative gamesmanship .
            test	very_positive	As a singular character study , it 's perfect .

        :param corpus_name:
        :param folder_num:
        :param test_idx:
        :param valid_idx: if doesn't exist and CV is True, valid_idx = (test_idx + 1) % folder_num
        :param kwargs:
        """
        self.corpus_name = corpus_name
        self.data_path = os.path.join(os.path.dirname(__file__), 'f_data/%s/processed.data' % corpus_name)
        self.pickle_root_path = os.path.join(os.path.dirname(__file__), 'f_data/pickle')

        if folder_num is None and test_idx is None and valid_idx is None:
            self.CV = False
            self.each_line_tag_idx = 1

            # get train, valid, test split index
            sen_total_num = 0
            with open(self.data_path, encoding='utf-8') as f:
                split, start, end = next(f).strip().split('-')
                assert split == 'test'
                self.test_start, self.test_end = int(start), int(end)
                split, start, end = next(f).strip().split('-')
                assert split == 'valid'
                self.valid_start, self.valid_end = int(start), int(end)
                for _ in f:
                    sen_total_num += 1
        else:
            self.CV = True
            self.each_line_tag_idx = 0
            valid_idx = (test_idx + 1) % folder_num if valid_idx is None else valid_idx

            # check
            assert folder_num is not None, 'folder_num must be provided.'
            assert test_idx is not None, 'test_idx must be provided.'

            # corpus sentence total number
            sen_total_num = 0
            with open(self.data_path, encoding='utf-8') as f:
                for _ in f:
                    sen_total_num += 1

            # get train, valid, test split index
            self.test_start, self.test_end = get_split(sen_total_num, folder_num, test_idx)
            self.valid_start, self.valid_end = get_split(sen_total_num, folder_num, valid_idx)

        self.sen_total_num = sen_total_num

    def _total_words(self, lower_case, remove_punc):
        """
        File format:
            the first word is tag, next words are words, the middle is separator '\t'
        :param lower_case:
        :param remove_punc:
        :return:
        """
        total_words = []
        with open(self.data_path, encoding='utf-8') as f:
            if not self.CV:
                next(f)
                next(f)
            for line in f:
                line = '\t'.join(line.split("\t")[self.each_line_tag_idx + 1:])

                line = line.strip()
                # lower case?
                if lower_case: line = line.lower()
                # remove punctuation?
                if remove_punc: line = remove_punc(line)
                # remove tag

                words = word_tokenize(line)
                total_words.append(words)
        return total_words

    def _total_tags(self):
        """
        File format:
            the first word is tag, next words are words, the middle is separator '\t'
        """
        total_tags = []
        with open(self.data_path, encoding='utf-8') as f:
            if not self.CV:
                next(f)
                next(f)
            for line in f:
                tag = line.split("\t")[self.each_line_tag_idx]
                total_tags.append(tag)
        return total_tags

    def get_words(self, **kwargs):
        """
        :param kwargs:
                lower_case: True or False
                threshold: word frequency threshold
                w2v_type: 'Google', 'Glove'
                w2v_dim: word2vec dimension
                remove_punc: If True, remove punctuation
                test_from: the source test word2vec from.
                    If 'total', the word out of total pre-trained word2vec will regard to be 'UNKNOWN';
                    if 'train', the word out of train words will regard to be "UNKNOWN".
        :return:
        """
        # get the parameters

        lower_case = kwargs.get('lower_case', True)
        threshold = kwargs.get('threshold', 1)
        w2v_type = kwargs.get('w2v_type', 'Google')
        w2v_dim = kwargs.get('w2v_dim', 300)
        remove_punc = kwargs.get('remove_punc', False)
        test_from = kwargs.get('test_from', 'total')

        # pickle path
        pickle_path = os.path.join(
            self.pickle_root_path,
            "%s-%s-%s-all-%d-t-%d-%d-v-%d-%d-lower-%s-thre-%d-remove_punc-%s-from-%s.words" % (
                self.corpus_name, w2v_type, w2v_dim, self.sen_total_num, self.test_start, self.test_end,
                self.valid_start, self.valid_end, lower_case, threshold, remove_punc, test_from))

        # if exists, load and return
        if os.path.exists(pickle_path):
            return pickle_load(pickle_path)

        # else, construct

        # get total_words
        total_words = self._total_words(lower_case, remove_punc)

        # get splits
        test_words = total_words[self.test_start: self.test_end]
        valid_words = total_words[self.valid_start: self.valid_end]
        train_words = total_words[min(self.test_end, self.valid_end):max(self.test_start, self.valid_start)] + \
                      total_words[:min(self.test_start, self.valid_start)] + \
                      total_words[max(self.test_end, self.valid_end):]

        # get total words frequencies
        if test_from == 'total':
            words_freq = Counter(yield_item(total_words))
        elif test_from == 'train':
            words_freq = Counter(yield_item(train_words))
        else:
            raise ValueError("Unknown test_from: %s" % test_from)

        # get index2word and word2index
        index2word = []
        for word, freq in words_freq.items():
            if freq >= threshold:
                index2word.append(word)
        index2word.extend(['UNKNOWN', 'ZERO'])
        word2index = {word: i for i, word in enumerate(index2word)}

        # embeddings
        w2v = W2VGet(w2v_type=w2v_type, w2v_dim=w2v_dim)
        embeddings = []
        for word in index2word:
            if word not in w2v.w2v_idx:
                w2v.add_word(word, low=-0.25, high=0.25)
            embeddings.append(w2v.find_word(word))
        embeddings[-1] = np.zeros((w2v_dim,))
        embeddings = np.asarray(embeddings, dtype='float32')

        # get train, test, valid index
        train_indices = []
        item_list2index_list(train_words, train_indices, word2index)
        valid_indices = []
        item_list2index_list(valid_words, valid_indices, word2index)
        test_indices = []
        item_list2index_list(test_words, test_indices, word2index)

        # results, pickle and return
        res = {
            'embeddings': embeddings,
            'index2word': index2word,
            'word2index': word2index,
            'train': train_indices,
            'valid': valid_indices,
            'test': test_indices
        }
        pickle_dump(res, pickle_path)
        return res

    def get_tags(self, **kwargs):

        total_tags = self._total_tags()
        index2item_name, item2index_name = 'index2tag', 'tag2index'

        # get train or test words
        test_items = total_tags[self.test_start: self.test_end]
        valid_items = total_tags[self.valid_start: self.valid_end]
        train_items = total_tags[min(self.test_end, self.valid_end): max(self.test_start, self.valid_start)] + \
                      total_tags[:min(self.test_start, self.valid_start)] + \
                      total_tags[max(self.valid_end, self.test_end):]

        # get index2pos & pos2index
        index2item = set(yield_item(total_tags))
        index2item = sorted(index2item)
        item2index = {item: i for i, item in enumerate(index2item)}

        # get index train pos and test train pos
        train_index_items = []
        item_list2index_list(train_items, train_index_items, item2index)
        test_index_items = []
        item_list2index_list(test_items, test_index_items, item2index)
        valid_index_items = []
        item_list2index_list(valid_items, valid_index_items, item2index)

        res = {
            index2item_name: index2item,
            item2index_name: item2index,
            'train': train_index_items,
            'test': test_index_items,
            'valid': valid_index_items
        }

        return res
