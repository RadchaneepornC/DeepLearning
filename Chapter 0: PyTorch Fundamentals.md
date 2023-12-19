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

![image] ()
