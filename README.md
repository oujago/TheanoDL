# Deep Learning Framework based on Theano

## Requirements

- Python3
- Theano >= 0.9
- scipy
- Numpy
- Matplotlib
- xlwt
- nltk

## Design Philosophy

Totally speaking, in every task, **_the operation flow/process is unchanged_** and **_the only
changing thing is just the specific operation in each operation_**. (业务流程是不变的，变化的只是
具体的业务。)

Every artificial intelligence(AI) task involves four components: **Model**, **Data**, 
**Execution** and **Evaluation**.

![four classes](doc/pics/p2.PNG)

The process of AI task is as follows:

![process](doc/pics/p3.PNG)

So, by extracting the unchanged process, we got our deep learning framework.

Thdl framework contains four modules:
 
- **data**
- **model**
- **execution**
- **evaluation**

## Features

- Support commonly used models, including convnets, RNNs and LSTMS.
- Support visualization of evaluation results.


