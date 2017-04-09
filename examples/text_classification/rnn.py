# -*- coding: utf-8 -*-


from thdl.data.text_classification import SentenceGetter
from thdl.data.text_classification import SentenceProcessor
from thdl.data.text_classification import SentenceProvider
from thdl import model
from thdl.exeval import ClassifyExeEval
from thdl.task import ClassificationTask


# data
data_getter = SentenceGetter('sst')
data_processor = SentenceProcessor()
data_module = SentenceProvider(shuffle=True)
data_module.set_getter(data_getter)
data_module.set_processor(data_processor)
data_module.build()


# model
model_module = model.Model()
model_module.set_input_tensor(model.tensors.ftensor3())
model_module.add_layer(model.layers.LSTM(n_out=500, n_in=300))
model_module.add_layer(model.layers.Dropout(0.5))
model_module.add_layer(model.layers.LSTM(n_out=400, return_sequence=False))
model_module.add_layer(model.layers.Dropout(0.5))
model_module.add_layer(model.layers.Softmax(n_out=10))
model_module.set_output_tensor(model.tensors.fmatrix())
model_module.set_objective(model.objective.CategoricalCrossEntropy())
model_module.set_optimizer(model.optimizer.Adam())
model_module.set_metrics([model.metrics.precision, model.metrics.recall])
model_module.build()


# execution and evaluation
exeval_module = ClassifyExeEval(batch_size=40)
exeval_module.set_aspects('training', 'valid', 'test')


# task
task = ClassificationTask()
task.set_model(model_module)
task.set_data(data_module)
task.set_exeval(exeval_module)
task.set_logfile(open("test.log", 'w'))

# run
task.hold_out_validation()

