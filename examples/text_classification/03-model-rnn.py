# -*- coding: utf-8 -*-

from thdl import model
from thdl.data import text_classification as corpus
from thdl.exeval import ClassifyExeEval
from thdl.task import ClassificationTask

# data
data_getter = corpus.SentenceGetter('./files/handled/trec.data')
data_processor = corpus.SentenceProcessor()
data = corpus.SentenceProvider(shuffle=True)
data.set_getter(data_getter)
data.set_processor(data_processor)
data.build()

# model
net = model.Network()
net.set_input_tensor(model.tensors.imatrix())
net.add_layer(model.layers.Embedding(data.get_embedding()))
net.add_layer(model.layers.Dropout(0.5))
net.add_layer(model.layers.LSTM(n_out=200))
net.add_layer(model.layers.LSTM(n_out=300))
net.add_layer(model.layers.Mean(axis=1))
net.add_layer(model.layers.Softmax(n_out=len(data.index_to_tag)))
net.set_output_tensor(model.tensors.fmatrix())
net.set_objective(model.objective.CategoricalCrossEntropy())
net.set_optimizer(model.optimizer.Adam(learning_rate=0.001))
net.set_metrics(train_metrics=[model.metrics.Loss(), model.metrics.Regularizer()],
                predict_metrics=model.metrics.Loss())
net.build()

# execution and evaluation
exeval = ClassifyExeEval(batch_size=50)
exeval.set_aspects('training', 'valid', 'test')
exeval.set_cpu_metrics("micro", 'macro_f1', 'macro_acc')

# task
task = ClassificationTask()
task.set_model(net)
task.set_data(data)
task.set_exeval(exeval)
task.set_logfile()

# run
task.hold_out_validation()
