# DNNs (Deep Neural Networks)🧠
## Introduction
**Q:** Why Deep Learning becomes popular today ? <br>

**A:** 
- It significant improved performance in NLP, ASR, Computer Vision, Robotics, Machine Translation, surpassed human performance in many tasks
- Big Data: DNN can take advantage of large amounts of data
- GPU: enable training bigger models possible
- Deep: Easier to avoid bad local minima when the model is large

## Double Descent & Inductive bias

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/Bias-Variance-Tradeoff.png)
[picture reference](https://www.cs.cornell.edu/courses/cs4780/2017sp/lectures/lecturenote11.html)<br>


As we have already known from BIAS-VARIANCE TRADEOFF that the larger model, the larger error (overfitting occurs) according to above picture shown, so, how about deep learning ? their large model will obtain much error or not ?

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/DoubleDescentProblem.png)

[OpenAI](https://openai.com/blog/deep-double-descent/) is first group where talk about DOUBLE DESCENT,as shown on the picture above, at significant large of model, the error will go down, not follow the classical statistical theory, so the way that the significant large model doesn't become overfit since it has [INDUCTIVE BIAS](http://www.cs.cmu.edu/~tom/pubs/NeedForBias_1980.pdf), a set of assumption that the algorithm used to generalize to new inputs into deep learning model 

## Fully connected networks(FNN)

### Neuron
### Non-linearity
### Softmax layer
### Cross Entropy
### MSE
### Regularization
### BackPropagation
### Non-Linearity and Gradients
### Initialization
### Learning Rate and Scheduling
### Beyond SGD
### Monitoring overfitting 
   - Dropout
   - Batchnorm
## CNN, RNN, LSTM, GRU





