# Framework Architecture

Every artificial intelligence(AI) task involves four components: **Model**, **Data**, 
**Execution** and **Evaluation**. At the same time, there are four kinds of parameters respectively:  
- **Model Parameters**: What kinds of layers? 
How many layers? What hyperparameters of each layer?
- **Data Parameters**: How to process the data? 
- **Execution Parameters**: How many epoches? Should we shuffle data?
- **Evaluation Parameters**: What metric should be chosen to evaluate our model? 

So, there should be _eight classes_ to manage the execution of every task, as shown in the following figure.

![eight classes](doc/pics/p1.PNG)

But, if we manage the parameters within each component by using same interface, we just need _four classes_ 
to perform an AI task.

![four classes](doc/pics/p2.PNG)

More importantly, in every task, **_the operation flow is unchanged_** and **_the only changing 
thing is just the specific content in each operation_**. Thus, our deep learning framework 
is born under the surpervision of this idea. The operation flow is shown in
[thdl/classify_framework.py](thdl/classify_framework.py)

# Advantages

My implemented deep learning framework is on the top of [Theano](https://github.com/Theano/Theano) 
library. I encapsulate commonly used models such as SimpleRNN, GRU, LSTM, CNN, CNN-RNN and several 
variants of LSTM. Our framework is flexible and convenient. The most important thing is that my framework 
**supports the visualization of evaluation results**, such as accuracy and loss.

# Usage




# Example

