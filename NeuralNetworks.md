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
other fragments





