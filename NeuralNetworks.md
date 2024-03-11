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
fragment of function
FNN: networks created by neuron conneced to each others
### Neuron

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/Neuron.png)

[picture reference](https://cs231n.github.io/neural-networks-1/)

Neuron: this concept mimics neuron in the human brainhaving the process of receiving signals from one place and sent out to another place,for analogy, we receive inputs(x) and multiply by weights then plus with bias, now become linear regression, then push into activation function(Non-linear)since it has study found that response of neuron signal is not in the linear relationship
- **if activation function is logistic function, so, one neuron network is linear regression + logistic regression ---> logistic regression**

- **the more number of neuron, the more complex of function**
![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/CombineNeuron.png)

- **Terminology**
![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/Terminology.png)
  - **Input layer:** scaling inputs to have to be scaled to have zero mean & unit variance(z-score normalization) or scaling them to a specific range, such as [0, 1] or [-1, 1], normalization helps in improving the stability and convergence of optimization algorithms, making the learning process more efficient, below are normalization from scratch

    ```python
    import numpy as np
    class CustomStandardScaler:
       def fit(self, X):
           self.mean_ = np.mean(X, axis=0)
           self.scale_ = np.std(X, axis=0)
       def transform(self, X):
           return (X - self.mean_) / self.scale_
       def fit_transform(self, X):
           self.fit(X)
           return self.transform(X)

    #for use
    # Sample input data
      X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
      # Create a CustomStandardScaler object
      scaler = CustomStandardScaler()
      # Fit the scaler to the data and transform the data
      X_scaled = scaler.fit_transform(X)
      print(X_scaled)
      ```

         [[-1.22474487 -1.22474487 -1.22474487]
          [ 0.          0.          0.        ]
          [ 1.22474487  1.22474487  1.22474487]]



  - **Hidden layer**
    one vertically pallarel is one layers, for each layer can compose of any
    number of neural network, and hidden layers compose of several layers
  - **Output layer**
    the last layer give us output, we called outputs from this layers as ```logits``` for classification problem
    
- **Projections and Neural Network weight**

$W^T \cdot X$, in neuron is the dot product or it is like projection in PCA, so 1 neuron is one eigen vector, if 2 neurons: project of matrix with 2 eigen vectors at the same time and add matrix with 2 data points

- **Neural network layer acts as nonlinear feature transform**

 $f(V^Tf(W^TX))$ , inner $f$ is **non-linear function**, otherwise the matrix can swap sequence of linear operations and the layer remain one layer

- **computational graph of neural network**
  
  this is why GPU is requirement because GPUs contain thousands of cores that can perform computations simultaneously, allowing them to process many operations in parallel, compute operations in parallel for each neuron of that layer for this case.
![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/ComputationalGraph.png)

### Non-linearity(Activation function)
Below are types of non-linearity function for stacking on layers of neural network architecture
  
![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/non-linear.png)

[picture reference](https://www.v7labs.com/blog/neural-networks-activation-functions)
   - **Sigmoid** or **Logistic function**: [0,1]

      $\sigma(x) = \frac{1}{1 + e^{-x}}$

   - **tanh**: [-1,1]

      $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
     
   - **Rectified Linear Unit (ReLU)** (most popular, default choice in most libary)<br>
     negative values --> 0, positive values --> its values

     $\text{ReLU}(x) = \max(0, x)$
     
     - LeakyReLu, ELU, PreLU
   - **Sigmoid Linear Units (SiLU)**

     $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$

     
     - Swish, Mish, GELU
       
### Softmax layer

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/SoftMax.png)


## DNN training
optimize loss
### Objective function(loss function)
This function can be any function that summarizes the performance into a single number
#### Cross Entropy 
- used for softmax outputs(probabilities), or classification problems

$$ L = - \sum_{i=1}^{n} y_i \log q_n(x, \theta) $$

Where:
- $L$ : the cross-entropy loss
- $n$ : the number of samples or data points
- $y_i$ : the true label or true probability associated with the the $i$ th sample (1 if data x comes from class n, 0 otherwise)
- $q_n(x, \theta)$ : the predicted probability distribution or output of the model $q$ parameterized by $\theta$ for the input $x$(Probability just go out the softmax function)

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/CrossEntropyLoss.jpg)

Log loss is the other names used for calling Cross Entropy loss, as you can see in the below picture, if we take log to the likelihood eq. of logistic regression(lowest eq.), we will obtain Cross Entropy loss for Binary class eq.
(Actually, these three term are the same, Entropy, Cross Entropy, KL Divergence, further studying [here](https://www.youtube.com/watch?v=ErfnhcEV1O8))
![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/CrossEntropy%26LogarithmLoss.png)

**Cons of Cross Entropy Loss**
It assumes every mistakes have equally mistake, this indicate that this value does not suit for kinds of problems having different cost to pay for each error<br>

**Ex** If the probabilities of Class0: Perfect, Class1: Good, Class 2: Average, Class3:Bad, 

- $y_i$ is [1, 0, 0, 0]
- **model A** got the $q_n(x, \theta)$: [0.4, 0.2, 0.1, 0.3]
- **model B** got the $q_n(x, \theta)$: [0.4, 0.3, 0.0, 0.3]

both model A and B got the same Cross Entropy Loss value, which is -log(0.4) though model B a bit better than model A since the score of Good (almost Perfect)from B is higher than A, so, solution for this CONs can be solved with [Squared EMD loss](https://arxiv.org/abs/1611.05916)

**Ex** Different direction of accuracy and loss(the better loss but worse accuracy)

- Groundtruth [1,0,0]
  - **Model A** [0.4, 0.3, 0.3], Model B [0.45, 0.5, 0.05]
  - **Model B** got the lower loss since the Probalility for correct class is higher, but model B incorrectly predict (it predict class 2 because of highest accuracy), this can be solved by monitoring accuracy as well as loss

**Ex** Leads to overconfidence
Cross Entropy usually lead model to answer 1,this can be solved by [label smoothing](https://paperswithcode.com/method/label-smoothing), [calibration](https://paperswithcode.com/method/label-smoothing)

#### MSE
used for regression problems

- **L1 vs L2 loss**
  - L1 robust to outliers
  - L2 easier to optimize (smooth gradient) since they can diff
  - Smoothed L1 combination of L1 and L2
    for part of less than 1 will work like L2, and part of more than 1 will work like L1

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/SmoothedL1.png)
[picture reference](https://www.researchgate.net/publication/321180616_Wing_Loss_for_Robust_Facial_Landmark_Localisation_with_Convolutional_Neural_Networks)




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
other fragments





