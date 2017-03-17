# Framework Architecture

Every artificial intelligence(AI) task involves four components: **Model**, **Data**, 
**Execution** and **Evaluation**. At the same time, there are four kinds of parameters respectively:  
- **Model Parameters**: What kinds of layers? 
How many layers? What hyperparameters of each layer?
- **Data Parameters**: How to process the data? 
- **Execution Parameters**: How many epoches? Should we shuffle data?
- **Evaluation Parameters**: What metric should be chosen to evaluate our model? 

So, there should be _eight classes_ to manage the execution of every task, as shown in the following figure.

[figure1]: /doc/pics/p1.png "eight_classes"
![eight classes][figure1]

But, if we manage the parameters within each component by using same interface, we just need _four classes_ 
to perform an AI task.

[figure2]: /doc/pics/p2.png "four_classes"

![four classes][figure2]

More importantly, in every task, **_the operation flow is unchanged_** and **_the only changing 
things are just the specific content in each operation_**. Thus, our deep learning framework 
is born under the surpervision of this idea.

# Advantages

My implemented deep learning framework is on the top of [Theano](https://github.com/Theano/Theano) 
library. We encapsulate commonly used models such as SimpleRNN, GRU, LSTM, CNN, CNN-RNN and several 
variants of LSTM. Our framework is flexible and convenient. The most important thing is that my framework 
**supports the visualization of evaluation results**, such as accuracy and loss.



