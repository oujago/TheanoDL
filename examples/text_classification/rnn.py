# -*- coding: utf-8 -*-


from thdl.data.text_classification import SentenceProvider
from thdl import model
from thdl.execution import Execution
from thdl.evaluation import ClassificationEvaluation
from thdl.task import ClassificationTask


# data
data_module = SentenceProvider(
    corpus_name='sst',
    w2v_type='glove.840B.300d',
    w2v_dim=300,
    lower_case=True,
    maxlen=30
)
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
model_module.set_metrics()
model_module.build()

# execution
exe_module = Execution(batch_size=40)


# evaluation
eval_module = ClassificationEvaluation(aspects=('training', 'valid'))


# task
task = ClassificationTask()
task.set_model(model_module)
task.set_data(data_module)
task.set_execution(exe_module)
task.set_evaluation(eval_module)
task.set_logfile(open("test.log", 'w'))

# run
task.hold_out_validation()

