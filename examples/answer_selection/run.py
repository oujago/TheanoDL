# -*- coding:utf-8 -*-

from examples.answer_selection.data import AnswerSelectionData
from examples.answer_selection.exeval import AnswerSelectionExe

from thdl.task import ClassificationTask
from thdl import model
from examples.answer_selection.layer import ASLayer

# data
maxlen = 50
data = AnswerSelectionData('./f_data/query_pairs.xs.data',
                           './f_data/query_pairs.ys.data',
                           maxlen=50,
                           threshold=3,)
data.build()


# model
net = model.Network()
net.set_input_tensor(model.tensors.imatrix(), model.tensors.imatrix())
net.add_layer(ASLayer(
    embedding_layer=model.layers.Embedding(input_size=len(data.vocab_to_idx), n_out=200, seq_len=maxlen, static=False),
    q1_conv_layer=model.layers.NLPConvPooling(100, (1,2,3))
))
net.add_layer(model.layers.Dropout(0.5))
net.add_layer(model.layers.Softmax(n_out=len(data.index_to_tag)))
net.set_output_tensor(model.tensors.fmatrix())
net.set_objective(model.objective.CategoricalCrossEntropy())
net.set_optimizer(model.optimizer.Adam(learning_rate=0.001))
net.set_metrics(model.metrics.Loss())
net.build()

# execution and evaluation
exeval = AnswerSelectionExe(batch_size=50, epochs=50)
exeval.set_aspects('training', 'valid', 'test')
exeval.set_cpu_metrics("micro", 'macro_f1', 'macro_acc')

# task
task = ClassificationTask()
task.set_model(net)
task.set_data(data)
task.set_exeval(exeval)
task.set_logfile()
task.hold_out_validation()

