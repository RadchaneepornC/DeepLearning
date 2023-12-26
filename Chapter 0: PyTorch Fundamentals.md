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
- Discovering insight with large collections of data: it is may not easy if we trying to hand-craft rules for diffenrentiating what 101 different kinds of food
  
**What deep learnign is not good for**
- When we need to explain the patterns since the patterns learned by a deep learnign model are typically uninterpretable by a human, it has a lot of numbers called weight and bias, has million, billion ofparameters(numbers or patterns in the data)
- When the traditional approach is a better option such as a simple rule-based system
- When errors are unacceptable since the outputs of deep learning model aren't always predictable
- When you don't have much data: deep learning models usually require a fairly large amount of data to produce great result

## Machine Learning VS Deep Learning
**Machine learning** is suit for structured data (table of data) while **Deep Learning** is suit for unstructured data (images, text, voice)
for deep learning we need to turn unstructure data to have some sort of structure through a beauty of a tensor

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

But many algorithms can be used for both depending how we represent our problem

## Neural Network and its Anatomy
1. before unstructured data gets used with a neural network, it needs to be turned into numbers (numerical encoding or representation)
2. then pass things through neural network (should a type that suit you problem, CNN for image, transformer for NLP & Speech for example)
3. then Neural Network learn representations/patterns/features/weights from the input data
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
  - Word Embeddings: Predicting missing words in a sentence, this is used in the step of Large Language Models pretaining which the model is pretrained from unlebel-text but self-supervised techniques can be of help, MLM for encoder only, CLM for decoder only, and sequence to sequence for encoder-decoder  
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

**Tensor** is a multi-dimensional matrix containing elements of a single data type

Pytorch tensors are created using 

```
torch.tensor()
```
torch.Tensor is an alias for the default tensor type torch.FloatTensor (32-bit floating point for CPU tensor)

Firstly, Let's explore about Scalar, Vector, Matrix, and followed by Tensor

| Type  |               Meaning                                                            | 
| ------| --------------------------------------------------------------------------------                            
|scalar | a singular number (just magnitude) |    
|vector | a number with direction            |   
|MATRIX | a 2-dimensional array of numbers   |     
|TENSOR | an n-dimensional array of numbers  | 

![Tensors](https://github.com/RadchaneepornC/DeepLearning/blob/d7d903a15f26b211071bb8bbd32dbff1dc86312f/images/Tensors.png)

![Tensors](https://github.com/RadchaneepornC/DeepLearning/blob/aee09dcc064c5c6b6bfe227dabfa5c8385f51395/images/Tensors%202.png)

| Type      |        Code                                                                                      |Dimensions   |  Size       |
| ----------| -------------------------------------------------------------|-----|-------|
| |  |```type.ndim``` (number of bracket for easy noticing)|```type.shape```(sets of matrix, rows of matrix, columns of matrix)|
| scalar    |```scalar = torch.tensor(7)```                                |   0       | -  |
| vector    |```vector = torch.tensor([7,7])```                            |1       | torch.Size([2]) |
| MATRIX    |```MATRIX = torch.tensor([[7, 8], [9, 10]])```                |   2       | torch.Size([2, 2])  |              
| TENSOR    |```TENSOR = torch.tensor([[[1, 2, 3],[3, 6, 9],[2, 4, 4]]])```|can be any number(if 0-dimension, tensor is a scalar,if 1-dimension, tensor is a vecor) | torch.Size([1, 3, 3])  |

**To avoid confusion**
- in the context of PyTorch, the dimension of a tensor and the number of channels in a tensor are related but distinct concepts. **The number of channels is typically the size of _the third dimension_ in a tensor**, especially in the context of image data in CNNs.
However, a tensor can have more than three dimensions, and each dimension has its own size.<br>

**(number of channels = size of the third dimension of tensor = number of outmosted bracket)**

- .shape is an attribute of the tensor whereas size() is a function. They both return the same value.


See about tensor dimensions belows

![Tensor Dimensions](https://github.com/RadchaneepornC/DeepLearning/blob/d7d903a15f26b211071bb8bbd32dbff1dc86312f/images/tensor%20dimension%20example.png)


### **Random tensors**
Many neural networks learn things by starting with these following steps:
1. random number in the tensors with  [torch.rand](https://pytorch.org/docs/stable/generated/torch.rand.html)
2. look at data
3. update random numbers
4. look at data
5. update random numbers and repeat 2-3 again and again for adjusting those random numbers to better represent the data

**Example**

- create a random tensor of size (3,3,4)

```random_tensor = torch.rand(3,3,4)```

tensor([[[0.8392, 0.2026, 0.3443, 0.3889],
         [0.4184, 0.2061, 0.5102, 0.7971],
         [0.9066, 0.4758, 0.8494, 0.1696]],

        [[0.2988, 0.3745, 0.1139, 0.7772],
         [0.9042, 0.8952, 0.4544, 0.4371],
         [0.5692, 0.0865, 0.8937, 0.6553]],

        [[0.2144, 0.8869, 0.4258, 0.6195],
         [0.6845, 0.8133, 0.3455, 0.7764],
         [0.2148, 0.8847, 0.7957, 0.7891]]])


### Represent an image as a tensor with shape
by identify size of tensor as [height, width, color channels(R,G,B)] 

![Image as tensors](https://github.com/RadchaneepornC/DeepLearning/blob/af6b52ba4c0e22f300ee8a2eabd5962a1b34e046/images/Image%20as%20tensors.png)

**Example**

```random_image_size_tensor = torch.rand(size=(224, 224, 3))```

### **Create a tensor with all zero and one**
- **create a tensor of all zeros**
```zeros = torch.zeros(size=(3,4)) ```

- **create a tensor of all ones**
```ones = torch,ones(size = (3,4))```

### **For check the type of tensor**
```tensor_name.dtype```

A [torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html) is an object that represents the data type of a tensor, Pytorch has 12 different data type 

### **Create a range of tensor**
```torch.arange(start, end+step, step =1)```

```torch.arange(start = 1, end = 11, step =1)```

```one_to_ten = torch.arange(1 , 11)```

```zero_to_ninety = torch.arange(0,100,10)``` --> tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

### **Create a range of tensor-like**
tensor_like is use for creating a tensor with the size same as other specufy tensor

```ten_zeros = torch.zero_like(input = one_to_ten)```

the ten_zeros tensor will have the same size as one_to_ten tensor

### **Dealing with Tensor datatypes, Device, Shape**

**big errors you'll run into with Pytorch & Deep Learning:**
      - Tensors not right datatype
      - Tensors not right shape
      - Tensors not on the right device

**1. Datatype**
1. Float 32 tensor is the default tensor datatypes, even though we specify dtype = None, the dtype still be torch.float32

```float_32_tensor = torch.tensor([3.0, 6.0, 9.0], **dtype = None** , device = None, requires_grad = False)```
or we can specify **dtype** we want [see list of tensor dtype](https://pytorch.org/docs/stable/tensors.html)


- the attribute used for checking tensor datatypes
```tensor_name.dtype```

- the method used for changing type of tensor datatypes, ex. converting to dtype = torch.float32
  ```existing_tensor.type(torch.float32)```


 **2. Device we specify when we create Tensor**
```float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype = None, **device = None** , requires_grad = False)```

we can set device for specifying what device is our tensor on
- device = None
- device = 'cpu'
- device = 'cuda'
  

**NOTE** operation between two or more tensors should be in the same device
- the attribute used for checking tensor device
```tensor_name.device```

3. Tensor shape
  - the attribute used for checking tensor shape
```tensor_name.shape``` 
   
(addition) requires_grad = False
   requires_grad (Default: True)
   to specify whether or not to track gradients with this tensors operation, we set True if autograd should record operations on this tensor

For more reading: [Precision in Computing](https://en.wikipedia.org/wiki/Precision_(computer_science))

 **Manipulating Tensor (Tensor Operations)**

For neural network, it has a lot of mathematic operations that Pytorch run behide the scence for us
**Tensor operations include:**
- **Addition**

```created_tensor + 10```
or 
```torch.add(created_tensor, 10)```

- **Substraction**
  ```created_tensor - 10```

- **Division** <br>

**This is two main ways of performimg multification in neural networks and deep learning:**

 import numpy as np

**Define the matrices**
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
  
  - **Multiplication (element-wise)**


```created_tensor * 10```
or 
```torch.mul(created_tensor, 10)```

element_wise_product = A * B

array([[ 5, 12],
        [21, 32]])


- **Matrix (Dot-product) multiplication**
dot_product =       
 array([[19, 22],
        [43, 50]]))

  ```torch.matmul(tensor,tensor)``` or  ```torch.mm(tensor,tensor)`` or ```tensor @ tensor```

**NOTE**
The inner dimensions must match so that the matrix multiplication can perform and the resulting matrix has the shape of the outer dimensions

e.g  (2,2) @ (3,2) --> the inner dimension doesn't match, so we can't perform matrix multification

e.g  (2,3) @ (3,2)
- (2,**3**) @ (**3**,2) --> the inner dimension match, so we can perform matrix multification
- (**2**,3) @ (3,**2**) --> as per the outer dimentions, the result of Matrix multification will have the shape of (2,2)

to fix our tensor **shape issues**, we can mulnipulate the shape of our tensors using a **transpose**.
A transpose switches the axes or dimentions of a given tensor: (2,3) to (3,2) for example

```Tensor_name.T```

### **Tensor Aggregation: min, max, mean, sum, etc**

- ```find_min = torch.min(x)``` or ```x.min()```
- ```find_max = torch.max(x)``` or ```x.max()```
- ```find_mean = torch.mean(x)``` or ```x.mean()``` <br>
  **NOTE** The torch.mean() function requires a tensor of float32 datatype to work <br>

- ```find_sum = torch.sum(x)``` or ```x.sum()```

### **To find the positional min and max**
- ```tensor.argmin()``` will find the position in tensor that has the minimum value with ```argmin()```, this return
the index position of the target tensor where the minimum value occurs
- ```tensor.argmax()``` will find the position in tensor that has the maximum value with ```argmax()```, this return
the index position of the target tensor where the maximum value occurs <br>
**NOTE** this is useful when we use the softmax activation function

### **Reshaping, stacking, squeezeing and unsqueezing Tensor (Shape and Dimention Tensor Munipulating)**
- **Reshaping** : reshapes an input tensor to a defined shape
- **View** : Return a view of an input tensor of certain shape but keep the same memory as the original tensor
- **Stacking** : combine multiple tensors on top of each other(vstack) or side by side (hstack)
- **Squeeze**: removes all 1 dimentions from a tensor
- **Unsqueeze**: add a 1 dimention to a target tensor
- **Permute**: Return a view of the input with dimensions permuted (swapped) in a certain way




  



