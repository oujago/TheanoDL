# -*- coding: utf-8 -*-

import os
from collections import Counter

import numpy as np
from nltk import word_tokenize

from thdl.base import ThdlObj
from thdl.utils.data_nlp_processing import get_split
from thdl.utils.data_nlp_processing import item_list2index_list
from thdl.utils.data_nlp_processing import remove_punctuation
from thdl.utils.data_nlp_processing import yield_item
from thdl.utils.data_nlp_processing import one_hot
from ..w2v import W2VGet


class SentenceGetter(ThdlObj):
    def __init__(self, data_path, w2v_type='google', w2v_dim=300, test_from='train',
                 lower_case=True, threshold=1, remove_punc=True):
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

        """
        self.data_path = os.path.join(os.getcwd(), data_path) if data_path else None

        self.w2v_type = w2v_type
        self.w2v_dim = w2v_dim
        self.lower_case = lower_case
        self.threshold = threshold
        self.remove_punc = remove_punc
        self.test_from = test_from

    def _get_split_boundary(self, folder_num=None, valid_idx=None, test_idx=None):
        """

        :param folder_num:
        :param valid_idx: if doesn't exist and CV is True, valid_idx = (test_idx + 1) % folder_num
        :param test_idx:
        """
        if folder_num is None and test_idx is None and valid_idx is None:

            # get train, valid, test split index
            sen_total_num = 0
            with open(self.data_path, encoding='utf-8') as f:
                split, start, end = next(f).strip().split('-')
                assert split == 'test'
                test_start, test_end = int(start), int(end)
                split, start, end = next(f).strip().split('-')
                assert split == 'valid'
                valid_start, valid_end = int(start), int(end)
                for _ in f:
                    sen_total_num += 1
        else:
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
            test_start, test_end = get_split(sen_total_num, folder_num, test_idx)
            valid_start, valid_end = get_split(sen_total_num, folder_num, valid_idx)

        return valid_start, valid_end, test_start, test_end, sen_total_num

    def get_words(self, folder_num=None, valid_idx=None, test_idx=None, ):
        """
            File format:
                the first word is tag, next words are words, the middle is separator '\t'
        """
        ##############################
        # get total corpus
        ##############################

        total_words = []
        with open(self.data_path, encoding='utf-8') as f:
            if folder_num is None:
                next(f)
                next(f)
            for line in f:
                line = '\t'.join(line.split("\t")[1:])

                line = line.strip()
                # lower case?
                if self.lower_case:
                    line = line.lower()
                # remove punctuation?
                if self.remove_punc:
                    line = remove_punctuation(line)
                # remove tag

                words = word_tokenize(line)
                total_words.append(words)

        ##############################
        # get index
        ##############################
        valid_start, valid_end, test_start, test_end, sen_total_num = \
            self._get_split_boundary(folder_num, valid_idx, test_idx)

        ##############################
        # get final corpus
        ##############################

        # get splits
        test_words = total_words[test_start: test_end]
        valid_words = total_words[valid_start: valid_end]
        train_words = total_words[min(test_end, valid_end):max(test_start, valid_start)] + \
                      total_words[:min(test_start, valid_start)] + \
                      total_words[max(test_end, valid_end):]

        # get total words frequencies
        if self.test_from == 'total':
            words_freq = Counter(yield_item(total_words))
        elif self.test_from == 'train':
            words_freq = Counter(yield_item(train_words))
        else:
            raise ValueError("Unknown test_from: %s" % self.test_from)

        # get index2word and word2index
        index2word = []
        for word, freq in words_freq.items():
            if freq >= self.threshold:
                index2word.append(word)
        index2word.extend(['UNKNOWN', 'ZERO'])
        word2index = {word: i for i, word in enumerate(index2word)}

        # embeddings
        w2v = W2VGet(w2v_type=self.w2v_type, w2v_dim=self.w2v_dim)
        embeddings = []
        for word in index2word:
            if word not in w2v.w2v_idx:
                w2v.add_word(word, low=-0.25, high=0.25)
            embeddings.append(w2v.find_word(word))
        embeddings[-1] = np.zeros((self.w2v_dim,))
        embeddings = np.asarray(embeddings, dtype='float32')

        # get train, test, valid index
        train_indices = item_list2index_list(train_words, word2index)
        valid_indices = item_list2index_list(valid_words, word2index)
        test_indices = item_list2index_list(test_words, word2index)

        # results, pickle and return
        res = {
            'embeddings': embeddings,
            'index2word': index2word,
            'word2index': word2index,
            'train': train_indices,
            'valid': valid_indices,
            'test': test_indices
        }
        return res

    def get_tags(self, folder_num=None, test_idx=None, valid_idx=None, ):
        """
        File format:
            the first word is tag, next words are words, the middle is separator '\t'
        """
        ##############################
        # get total corpus
        ##############################
        total_tags = []
        with open(self.data_path, encoding='utf-8') as f:
            if folder_num is None:
                next(f)
                next(f)
            for line in f:
                tag = line.split("\t")[0]
                total_tags.append(tag)

        ##############################
        # get index
        ##############################
        valid_start, valid_end, test_start, test_end, sen_total_num = \
            self._get_split_boundary(folder_num, valid_idx, test_idx)

        ##############################
        # get final corpus
        ##############################

        # get train or test words
        test_items = total_tags[test_start: test_end]
        valid_items = total_tags[valid_start: valid_end]
        train_items = total_tags[min(test_end, valid_end): max(test_start, valid_start)] + \
                      total_tags[:min(test_start, valid_start)] + \
                      total_tags[max(valid_end, test_end):]

        # get index2pos & pos2index
        index2item = set(yield_item(total_tags))
        index2item = sorted(index2item)
        item2index = {item: i for i, item in enumerate(index2item)}

        # get index train pos and test train pos
        train_index_items = item_list2index_list(train_items, item2index)
        test_index_items = item_list2index_list(test_items, item2index)
        valid_index_items = item_list2index_list(valid_items, item2index)

        res = {
            'index2tag': index2item,
            'tag2index': item2index,
            'train': one_hot(train_index_items),
            'test': one_hot(test_index_items),
            'valid': one_hot(valid_index_items)
        }

        return res

    def to_json(self):
        config = {
            "w2v_type": self.w2v_type,
            "w2v_dim": self.w2v_dim,
            "lower_case": self.lower_case,
            "threshold": self.threshold,
            "remove_punctuation": self.remove_punc,
            "test_from": self.test_from,
        }
        return config
