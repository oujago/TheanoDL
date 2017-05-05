# -*- coding:utf-8 -*-


with open('./f_data/zh_query_pairs.xs.data', encoding='utf-8') as fin:
    i = 0
    true = True
    for line in fin:
        i += 1
        print()
        for sent in line.strip().split("\t"):
            print(true, sent)
        print()

        if i % 2 == 0:
            true = True
        else:
            true = False


print(i)

