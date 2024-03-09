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

As we have already known from bias-variance tradeoff that the larger model, the larger error (overfitting occurs) according to above picture shown, so, how about deep learning ? their large model will obtain much error or not ?

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/DoubleDescentProblem.png)

OpenAI is first group where talk about DOUBLE DESCENT,as shown on the picture above, at significant large of model, the error will go down, not follow the classical statistical theory
