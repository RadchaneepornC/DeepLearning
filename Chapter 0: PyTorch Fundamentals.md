# Chapter 0: PyTorch Fundamentals
## What is Deep Learning

**Deep Learning** is a subset of machine learning, and machine learning is turning things (data: text, image, video) into numbers and finding patterns(by using code & math) in those numbers

- **Traditional programming:** rule based: put rules into the inputs to get the outputs
- **Machine learning algorithm:** findout rules from inputs and outputs

## Why use Machine Learning or Deep Learning ?

- For a complex problem, we can not think of all the rules, so ML and DL can help
  
**NOTE** If you can build a simple rule-based system that doesn't require machine learning to do that

**What deep learnign is good for**

- Problems with long lists of rules:  when the traditional approach fails, machine learning/deep learning may help
- Continually changing environments: deep learning can adapt ('learn') to new scenarios
- Discovering insighr with large collections of data: it is may not easy if we trying to hand-craft rules for diffenrentiating what 101 different kinds of food
  
**What deep learnign is not good for**
- When we need to explain the patterns since the patterns learned by a deep learnign model are typically uninterpretable by a human, it has a lot of numbers called weight and bias, has million, billion ofparameters(numbers or patterns in the data)
- When the traditional approach is a better option such as a simple rule-based system
- when errors are unacceptable since the outputs of deep learning model aren't always predictable
- when you don't have much data: deep learning models usually require a fairly large amount of data to produce great result

## Machine Learning VS Deep Learning
**Machine learning** is suit for structured data (table of data) while **Deep Learning** is suit for unstructured data (images, text, voice)
for deep learning you need to turn unstructure data to have some sort of structure through a beauty of a tensor

**Algorithm for Machine Learning (structure data)**
- Random forest
- Gradient boosted models
- Naives Bayes
- Nearest neighbour
- Support vector machine, and more

**Algorithm for Deep Learning (unstructure data)**
- Neural Network (focus on building with Pytorch)
- Fully conneted neural network (focus on building with Pytorch)
- Convolutional neural network
- Recurrent neural network
- Transformer

But many algorithms can be used for both depending how you represent your problem

## Neural Network and its Anatomy
1. before unstructured data gets used with a neural network, it needs to be turned into numbers (numerical encoding or representation)
2. then pass things through neural network (should a type that suit you problem, CNN for image, transformer for NLP, Speech)
3. then Neural Network learn representation (pattern, features, weights) from the input data
4. got the representation outputs that human can understand

![alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/Neural%20Network.png)

**Architecture of Neural Network**

![alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/Achitecture%20of%20neural%20network.jpg)

## Learning Paradigms of Neural Network

**1. Supervised Learning** <br>
have data and label <br>
**2. Unsupervised & Self-supervised Learning**
- **Unsupervised Learning:**<br>
**Definition**:<br>
 In unsupervised learning, the algorithm is provided with input data that has no corresponding output labels. The algorithm tries to find patterns, structures, or relationships within the data without explicit guidance.<br>
**Goal:**<br>
The primary goal of unsupervised learning is to discover the inherent structure of the data. Common tasks include clustering, dimensionality reduction, and density estimation.<br>
**Examples:**<br>
  - Clustering: Grouping similar data points together.<br>
  - Dimensionality Reduction: Reducing the number of features while preserving relevant information.<br>

- **Self-supervised:** <br>
**Definition**:<br>
Self-supervised learning is a specific type of unsupervised learning where the algorithm generates its own supervision signal from the input data. It creates a pretext task by defining relationships within the data, and the model learns to solve this task without external labels.<br>
**Goal:** <br> 
The primary goal of self-supervised learning is to learn useful representations of the data. The pretext task is designed to capture meaningful features that can be transferred to other downstream tasks.<br>
**Examples:** <br>
  - Word Embeddings: Predicting missing words in a sentence.
  - Image Inpainting: Reconstructing missing parts of an image.
  - Temporal Order Prediction: Determining the correct temporal order of shuffled input sequences.<br>
  
**3. Transfer Learning**
The algorithm taking pattern that learn in one model over the dataset and **transfering** it to another model

## What is deep learning actually used for ?

**Some deep learning use case**
- Recommendation system
- Translation (sequence to sequence: seq2seq)
- Speech Recognition (sequence to sequence: seq2seq)
- Computer Vision (Classification/Regression)
- Natural Language Processing (NLP) e.g. Spam/Not Spam classifier (Classification/Regression)

## What is Pytorch ?
- Most popular python deep learning framework allow you to write fast deep learning python code accelerate by GPU
- Able to access many pre-built deep learning models (Torch Hub/torvision.models)
- Whole stack of Machine Learning: prepocess data, model data, deploy model in your application/cloud
- Originally designed and used in-house by Facebook/Meta (now open-source and used by companies such as Tesla, Microsoft, OpenAI

## What is a GPU/TPU
- **GPU (Graphics Processing Unit)**: orginally design for video games
  - Pytorch enables you to leverage cuda to enable us to run our machine learning code on nvidia gpus
    - CUDA is stand for **C**ompute **U**nified **D**evice **A**rchitecture. It's a technology developed by NVIDIA that allows your computer's graphics processing unit (GPU) to be used for more than just creating cool graphics in video games  
    - CUDA lets programmers use the power of the GPU for other tasks too, not just graphics. It's like unlocking the super abilities of the GPU to help with all kinds of computations, making certain tasks run much faster and more efficiently. This is particularly useful for things like scientific simulations, artificial intelligence, and other complex calculations
  
- **TPU (Tensor Processing Unit)**

## What is tensor ?

![alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/Tensors.jpg)












