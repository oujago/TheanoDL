# -*- coding: utf-8 -*-


from thdl.data import w2v


def google():
    w2v.reformat_google(origin_path="E:/data/NLP/google/GoogleNews-vectors-negative300.bin.gz",
                        index_save_path="E:/data/NLP/google/google_word2vec_index.pkl",
                        data_save_path='E:/data/NLP/google/google_word2vec_data.txt')


def glove_6B():
    for dim in ['50d', '100d', '200d', "300d"]:
        file_name = "glove.6B.%s.txt" % dim
        print("Handling corpus: %s" % file_name)
        w2v.reformat_glove(origin_path="E:/data/NLP/glove/%s" % file_name,
                           index_save_path="E:/data/NLP/glove/handled/glove.6B.%s.index.pkl" % dim,
                           data_save_path="E:/data/NLP/glove/handled/glove.6B.%s.data.txt" % dim,
                           w2v_type="glove.6B.%s" % dim)
        print("\n" * 4)


def glove_27B():
    for dim in ['25d', '50d', '100d', '200d']:
        file_name = "glove.twitter.27B.%s.txt" % dim
        print("Handling corpus: %s" % file_name)
        w2v.reformat_glove(origin_path="E:/data/NLP/glove/%s" % file_name,
                           index_save_path="E:/data/NLP/glove/handled/glove.27B.%s.index.pkl" % dim,
                           data_save_path="E:/data/NLP/glove/handled/glove.27B.%s.data.txt" % dim,
                           w2v_type="glove.27B.%s" % dim)
        print("\n" * 4)


def glove_42B():
    w2v.reformat_glove(origin_path="E:/data/NLP/glove/glove.42B.300d.txt",
                       index_save_path="E:/data/NLP/glove/glove.42B.300d.index.pkl",
                       data_save_path="E:/data/NLP/glove/glove.42B.300d.data.pkl",
                       w2v_type="glove.42B.300d")


def glove_840B():
    w2v.reformat_glove(origin_path="E:/data/NLP/glove/glove.840B.300d.txt",
                       index_save_path="E:/data/NLP/glove/glove.840B.300d.index.pkl",
                       data_save_path="E:/data/NLP/glove/glove.840B.300d.data.txt",
                       w2v_type="glove.840B.300d")


if __name__ == '__main__':
    # google()
    glove_6B()
    glove_27B()
    glove_42B()
    glove_840B()

