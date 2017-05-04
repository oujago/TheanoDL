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


batch_size = 40
w2v_dim = 200

# data
maxlen = 30
data = AnswerSelectionData('./f_data/query_pairs.xs.data',
                           './f_data/query_pairs.ys.data',
                           maxlen=maxlen,
                           threshold=3)
data.build()

# model
net = model.Network()
net.set_input_tensor([model.tensors.imatrix(), model.tensors.imatrix()])
net.add_layer(ASLayer(
    embedding_layer=model.layers.Embedding(input_size=len(data.vocab_to_idx),
                                           n_out=w2v_dim,
                                           static=False,
                                           zero_idxs=(-1, -2)),
    q1_conv_layer=model.layers.NLPConvPooling((batch_size, 1, maxlen, w2v_dim), 100, (1, 2, 3)),
    q2_conv_layer=model.layers.NLPConvPooling((batch_size, 1, maxlen, w2v_dim), 100, (1, 2, 3))
))
net.add_layer(model.layers.Dropout(0.5))
net.add_layer(model.layers.Softmax(n_in=600, n_out=len(data.get_index_to_tag())))
net.set_output_tensor(model.tensors.fmatrix())
net.set_objective(model.objective.CategoricalCrossEntropy())
net.set_optimizer(model.optimizer.Adam(learning_rate=0.001))
net.set_metrics(model.metrics.Loss())
net.build()

# execution and evaluation
exeval = AnswerSelectionExe(batch_size=batch_size, epochs=50,
                            convert_func=data.batch_pairs_to_idxs)
exeval.set_aspects('training', 'valid', 'test')
exeval.set_cpu_metrics("micro", 'macro_f1', 'macro_acc')

# task
task = ClassificationTask()
task.set_model(net)
task.set_data(data)
task.set_exeval(exeval)
task.set_logfile()
task.hold_out_validation()
