# -*- coding:utf-8 -*-

try:
    from data import AnswerSelectionData
    from exeval import AnswerSelectionExe
    from layer import ASLayer
except ImportError:
    from examples.answer_selection.data import AnswerSelectionData
    from examples.answer_selection.exeval import AnswerSelectionExe
    from examples.answer_selection.layer import ASLayer

from thdl import model
from thdl.task import ClassificationTask


def net0(net, data, w2v_dim, batch_size, maxlen):
    """Bi-CNN
    
    :param net: 
    :param data: 
    :param w2v_dim: 
    :param batch_size: 
    :param maxlen: 
    :return: 
    """

    net.add_layer(ASLayer(
        embedding_layer=model.layers.Embedding(input_size=len(data.vocab_to_idx),
                                               n_out=w2v_dim,
                                               static=True,
                                               zero_idxs=(-1, -2)),
        q1_conv_layer=model.layers.NLPConvPooling((batch_size, 1, maxlen, w2v_dim), 50, (1, 2, 3)),
        q2_conv_layer=model.layers.NLPConvPooling((batch_size, 1, maxlen, w2v_dim), 50, (1, 2, 3))
    ))

    return 300


def net1(net, data, w2v_dim, batch_size, maxlen):
    """CNN structure
    
    :param net: 
    :param data: 
    :param w2v_dim: 
    :param batch_size: 
    :param maxlen: 
    :return: 
    """
    net.add_layer(model.layers.MultiInput(model.layers.Embedding(input_size=len(data.vocab_to_idx),
                                                                 n_out=w2v_dim,
                                                                 static=True,
                                                                 zero_idxs=(-1, -2))))
    net.add_layer(model.layers.MultiInput(model.layers.Dimshuffle((0, 'x', 1, 2))))
    net.add_layer(model.layers.MultiInput(model.layers.NLPConvPooling((batch_size, 1, maxlen, w2v_dim), 50, (1, 2, 3))))
    net.add_layer(model.layers.Concatenate(axis=1))

    return 300


def net2(net, data, w2v_dim, batch_size, maxlen):
    """SimpleRNN structure
    
    :param net: 
    :param data: 
    :param w2v_dim: 
    :param batch_size: 
    :param maxlen: 
    :return: 
    """
    n_out = 300

    net.add_layer(model.layers.MultiInput(model.layers.Embedding(input_size=len(data.vocab_to_idx),
                                                                 n_out=w2v_dim,
                                                                 static=True,
                                                                 zero_idxs=(-1, -2))))
    net.add_layer(model.layers.MultiInput(model.layers.SimpleRNN(n_in=w2v_dim, n_out=n_out, activation=model.nonlinearity.ReLU())))
    net.add_layer(model.layers.MultiInput(model.layers.MaxPooling(pool_size=(maxlen, 1))))
    net.add_layer(model.layers.MultiInput(model.layers.Flatten(2)))
    net.add_layer(model.layers.Concatenate(axis=1))

    return n_out * 2


def net3(net, data, w2v_dim, batch_size, maxlen):
    """SimpleRNN structure + Attention

    :param net: 
    :param data: 
    :param w2v_dim: 
    :param batch_size: 
    :param maxlen: 
    :return: 
    """
    n_out = 200

    net.add_layer(model.layers.MultiInput(model.layers.Embedding(input_size=len(data.vocab_to_idx),
                                                                 n_out=w2v_dim,
                                                                 static=True,
                                                                 zero_idxs=(-1, -2))))
    net.add_layer(model.layers.MultiInput(model.layers.SimpleRNN(n_in=w2v_dim, n_out=n_out,
                                                                 activation=model.nonlinearity.ReLU())))
    net.add_layer(model.layers.MultiInput(model.layers.Attention(n_in=n_out)))
    # net.add_layer(model.layers.MultiInput(model.layers.MaxPooling(pool_size=(maxlen, 1))))
    net.add_layer(model.layers.MultiInput(model.layers.Flatten(2)))
    net.add_layer(model.layers.Concatenate(axis=1))

    return n_out * 2


def net4(net, data, w2v_dim, batch_size, maxlen):
    """GRU structure

    :param net: 
    :param data: 
    :param w2v_dim: 
    :param batch_size: 
    :param maxlen: 
    :return: 
    """
    n_out = 200

    net.add_layer(model.layers.MultiInput(model.layers.Embedding(input_size=len(data.vocab_to_idx),
                                                                 n_out=w2v_dim,
                                                                 static=True,
                                                                 zero_idxs=(-1, -2))))
    net.add_layer(model.layers.MultiInput(model.layers.GRU(n_in=w2v_dim,
                                                           n_out=n_out,
                                                           activation=model.nonlinearity.ReLU())))
    net.add_layer(model.layers.MultiInput(model.layers.MaxPooling(pool_size=(maxlen, 1))))
    net.add_layer(model.layers.MultiInput(model.layers.Flatten(2)))
    net.add_layer(model.layers.Concatenate(axis=1))

    return n_out * 2




def main():

    batch_size = 40
    w2v_dim = 200

    # data
    maxlen = 30
    data = AnswerSelectionData(xs_path='./f_data/zh_query_pairs.xs.data',
                               ys_path='./f_data/zh_query_pairs.ys.data',
                               vocab_path="./f_data/zh_vocabs_freqs.data",
                               maxlen=maxlen,
                               threshold=3,
                               prefix='chinese_only',
                               batch_size=batch_size,
                               total_len=100000)
    data.build()

    # net preparation
    net = model.Network()
    net.set_input_tensor([model.tensors.imatrix(), model.tensors.imatrix()])



    # n_in = net0(net, data, w2v_dim, batch_size, maxlen)
    # n_in = net1(net, data, w2v_dim, batch_size, maxlen)
    # n_in = net2(net, data, w2v_dim, batch_size, maxlen)
    # n_in = net3(net, data, w2v_dim, batch_size, maxlen)
    n_in = net4(net, data, w2v_dim, batch_size, maxlen)



    # net.add_layer(model.layers.Dropout(0.5))
    net.add_layer(model.layers.Softmax(n_in=n_in, n_out=len(data.get_index_to_tag())))
    net.set_output_tensor(model.tensors.fmatrix())
    net.set_objective(model.objective.CategoricalCrossEntropy())
    net.set_optimizer(model.optimizer.Adam(learning_rate=0.001))
    net.set_metrics(model.metrics.Loss())
    net.build()

    # execution and evaluation
    exeval = AnswerSelectionExe(batch_size=batch_size, epochs=20,
                                convert_func=data.batch_pairs_to_idxs)
    exeval.set_aspects('training', 'valid', 'test')
    exeval.set_cpu_metrics("micro", 'macro_f1')
    exeval.set_model_chosen_metrics('micro')

    # task
    task = ClassificationTask()
    task.set_model(net)
    task.set_data(data)
    task.set_exeval(exeval)
    task.set_logfile()
    task.hold_out_validation()


main()


