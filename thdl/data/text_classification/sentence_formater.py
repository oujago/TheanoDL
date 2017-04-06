# -*- coding: utf-8 -*-


import os
import csv


def movie_review_corpus(file_folder, save_path):
    with open(os.path.join(os.getcwd(), save_path), 'w', encoding='utf-8') as f:
        with open(os.path.join(os.getcwd(), file_folder, 'rt-polarity.pos'), encoding='latin-1') as pos_f:
            with open(os.path.join(os.getcwd(), file_folder,'rt-polarity.neg'), encoding='latin-1') as neg_f:
                for pos_line, neg_line in zip(pos_f, neg_f):
                    pos_line = pos_line.strip()
                    f.write("%s\t%s\n" % ("positive", pos_line))
                    neg_line = neg_line.strip()
                    f.write("%s\t%s\n" % ('negative', neg_line))


def subjective_corpus(file_folder, save_path):
    with open(os.path.join(os.getcwd(), save_path), 'w', encoding='utf-8') as f:
        with open(os.path.join(os.getcwd(), file_folder, 'quote.tok.gt9.5000'), encoding='latin-1') as subj_f:
            with open(os.path.join(os.getcwd(), file_folder, 'plot.tok.gt9.5000'), encoding='latin-1') as obj_f:
                for subj_line, obj_line in zip(subj_f, obj_f):
                    f.write('%s\t%s\n' % ("subjective", subj_line.strip()))
                    f.write('%s\t%s\n' % ("objective", obj_line.strip()))


def _stanford_sentiment_treebank(file_folder):

    sentences = {}
    with open(os.path.join(os.getcwd(), file_folder, "datasetSentences.txt"), "r") as f:
        rd = csv.reader(f, delimiter='\t')
        count = 0
        for line in rd:
            if count == 0:
                count = 1
                continue
            line[1] = line[1].replace('-LRB-', '(')
            line[1] = line[1].replace('-RRB-', ')')
            line[1] = line[1].replace('Â', '')
            line[1] = line[1].replace('Ã©', 'e')
            line[1] = line[1].replace('Ã¨', 'e')
            line[1] = line[1].replace('Ã¯', 'i')
            line[1] = line[1].replace('Ã³', 'o')
            line[1] = line[1].replace('Ã´', 'o')
            line[1] = line[1].replace('Ã¶', 'o')
            line[1] = line[1].replace('Ã±', 'n')
            line[1] = line[1].replace('Ã¡', 'a')
            line[1] = line[1].replace('Ã¢', 'a')
            line[1] = line[1].replace('Ã£', 'a')
            line[1] = line[1].replace('\xc3\x83\xc2\xa0', 'a')
            line[1] = line[1].replace('Ã¼', 'u')
            line[1] = line[1].replace('Ã»', 'u')
            line[1] = line[1].replace('Ã§', 'c')
            line[1] = line[1].replace('Ã¦', 'ae')
            line[1] = line[1].replace('Ã­', 'i')
            line[1] = line[1].replace('\xa0', ' ')
            line[1] = line[1].replace('\xc2', '')
            sentences[line[0]] = line[1]

    train = {}
    test = {}
    dev = {}
    sents = []
    with open(os.path.join(os.getcwd(), file_folder, "datasetSplit.txt"), "r") as f:
        rd = csv.reader(f, delimiter=',')
        count = 0
        for line in rd:
            if count == 0:
                count = 1
                continue
            if line[1] == '1':
                train[sentences[line[0]]] = 0
                sents.append(sentences[line[0]])
            elif line[1] == '2':
                test[sentences[line[0]]] = 0
            elif line[1] == '3':
                dev[sentences[line[0]]] = 0

    train_sent = train.copy()
    string = " ".join(sents)
    with open(os.path.join(os.getcwd(), file_folder, "dictionary.txt"), "r") as f:
        rd = csv.reader(f, delimiter='|')
        for line in rd:
            line[0] = line[0].replace('é', 'e')
            line[0] = line[0].replace('è', 'e')
            line[0] = line[0].replace('ï', 'i')
            line[0] = line[0].replace('í', 'i')
            line[0] = line[0].replace('ó', 'o')
            line[0] = line[0].replace('ô', 'o')
            line[0] = line[0].replace('ö', 'o')
            line[0] = line[0].replace('á', 'a')
            line[0] = line[0].replace('â', 'a')
            line[0] = line[0].replace('ã', 'a')
            line[0] = line[0].replace('à', 'a')
            line[0] = line[0].replace('ü', 'u')
            line[0] = line[0].replace('û', 'u')
            line[0] = line[0].replace('ñ', 'n')
            line[0] = line[0].replace('ç', 'c')
            line[0] = line[0].replace('æ', 'ae')
            line[0] = line[0].replace('\xa0', ' ')
            line[0] = line[0].replace('\xc2', '')
            if line[0] in string:
                train[line[0]] = line[1]
            if line[0] in test:
                test[line[0]] = line[1]
            if line[0] in train_sent:
                train_sent[line[0]] = line[1]
            if line[0] in dev:
                dev[line[0]] = line[1]

    labels = {}
    with open(os.path.join(os.getcwd(), file_folder, "sentiment_labels.txt"), "r") as f:
        rd = csv.reader(f, delimiter='|')
        count = 0
        for line in rd:
            if count == 0:
                count = 1
                continue
            labels[line[0]] = float(line[1])

    for key in train:
        train[key] = labels[train[key]]
    for key in train_sent:
        train_sent[key] = labels[train_sent[key]]
    for key in test:
        test[key] = labels[test[key]]
    for key in dev:
        dev[key] = labels[dev[key]]

    return train, train_sent, test, dev


def stanford_sentiment_treebank_phrase(file_folder, save_path):
    train, _, test, valid = _stanford_sentiment_treebank(file_folder)

    train_len, valid_len, test_len = len(train), len(valid), len(test)
    runout = ('valid-%s-%s\n' % (train_len, train_len + valid_len))
    runout += ('test-%s-%s\n' % (train_len, train_len + test_len + valid_len))

    with open(os.path.join(os.getcwd(), save_path), 'w', encoding='utf-8') as f:
        f.write(runout)
        for data_set, split_name in [(train, 'train'), (valid, 'valid'), (test, 'test')]:
            for score, sen in data_set.items():
                score = float(score)

                # sen = sen.replace('\xa0', ' ')
                if 0 <= score <= 0.2:
                    score_ = 'very_negative'
                elif 0.2 < score <= 0.4:
                    score_ = 'negative'
                elif 0.4 < score <= 0.6:
                    score_ = 'neutral'
                elif 0.6 < score <= 0.8:
                    score_ = 'positive'
                elif 0.8 < score <= 1.0:
                    score_ = 'very_positive'
                else:
                    raise ValueError("Invalid score.")
                f.write("%s\t%s\t%s\n" % (split_name, score_, sen))


def trec_corpus(file_folder, save_path):
    res = []

    # train data
    test_idx = 0
    for split, path in [('test', 'TREC_10.label'), ('train', 'train_5500.label'), ]:
        with open(os.path.join(os.getcwd(), file_folder, path), encoding='latin-1') as f:
            for line in f:
                splits = line.strip().split(" ")
                label = splits[0].split(":")[0]
                res.append((split, label, ' '.join(splits[1:])))
                if split == 'test':
                    test_idx += 1

    # write into file
    with open(os.path.join(os.getcwd(), save_path), 'w', encoding='utf-8') as f:
        f.write("test-%d-%d\nvalid-%d-%d\n" % (0, test_idx, test_idx, test_idx))
        for split, label, sentence in res:
            f.write("%s\t%s\t%s\n" % (split, label, sentence))


