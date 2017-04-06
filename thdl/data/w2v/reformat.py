# -*- coding: utf-8 -*-


from thdl.utils.config import set_w2v
from thdl.utils.file import pickle_dump


def reformat_glove(origin_path, data_save_path, index_save_path, w2v_type, precision=12):
    # write
    word2idx = {}
    i = 0
    length = 0
    w2v_dim = 0
    print("Reading data from %s ... " % origin_path)
    with open(origin_path, encoding='utf-8') as origin_f:
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
    w2v_type = w2v_type
    vec_len = length
    pickle_dump([w2v_type, w2v_dim, vec_len, word2idx], index_save_path)
    print("Pickled index into the file.")

    print("Set configuration ...")
    set_w2v(w2v_type, w2v_index=index_save_path, w2v_data=data_save_path)


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

    reformat_glove(temp_path, data_save_path, index_save_path, 'google', precision)
