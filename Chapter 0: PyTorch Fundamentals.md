# Chapter 0: PyTorch Fundamentals
## What is Deep Learning

**Deep Learning** is a subset of machine learning, and machine learning is turning things (data: text, image, video) into numbers and finding patterns(by using code & math) in those numbers

- **Traditional programming:** rule based: put rules into the inputs to get the outputs
- **Machine learning algorithm:** findout rules from inputs and outputs

## Why use Machine Learning or Deep Learning ?

- For a complex problem, we can not think of all the rules, so ML and DL can help
  
**NOTE** If we can build a simple rule-based system, it means that the system doesn't require any machine learning 

**What deep learning is good for**

- Problems with long lists of rules:  when the traditional approach fails, machine learning/deep learning may help
- Continually changing environments: deep learning can adapt ('learn') to new scenarios
- Discovering insight with large collections of data: it is may not easy if we trying to hand-craft rules for diffenrentiating what 101 different kinds of foods
  
**What deep learning is not good for**
- When we need to explain the patterns since the patterns learned by a deep learnimg model are typically uninterpretable by a human, it has a lot of numbers called weight and bias, has million, billion of parameters (numbers or patterns in the data)
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
- Fully connected neural network (focus on building with Pytorch)
- Convolutional neural network
- Recurrent neural network
- Transformer

But many algorithms can be used for both depending how we represent our problem

## Neural Network and its Anatomy
1. before unstructured data gets used with a neural network, it needs to be turned into numbers (numerical encoding or representation)
2. then pass those numbers through neural network (should a type that suit you problem, CNN for image, transformer for NLP & Speech for example)
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
- torch.Tensor is an alias for the default tensor type torch.FloatTensor (32-bit floating point for CPU tensor)

- A tensor of specific data type can be constructed by passing a ```torch.dtype``` and/or a ```torch.device``` to a constructor or tensor creation op

          torch.ones([2, 4], dtype=torch.float64, device=cuda0)



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
by identifying size of tensor as [height, width, color channels(R,G,B)] 

![Image as tensors](https://github.com/RadchaneepornC/DeepLearning/blob/af6b52ba4c0e22f300ee8a2eabd5962a1b34e046/images/Image%20as%20tensors.png)

**Example**

```random_image_size_tensor = torch.rand(size=(224, 224, 3))```

-------------------------------------------------------------------------------
🤔If you, like me, are doubtful about **order of dimensions in the tensor**🤔<br>
💡The answers below can make it crystal clear for you❗️<br>

There are two common format to order dimensions of tensor 
- **NCHW** (Batch, Channel, Height, Width)
  N: Number of images in a batch.
  C: Number of channels in the image (e.g., 3 for RGB, 1 for grayscale).
  H: Height of the image.
  W: Width of the image.

**Ex** Suppose you have a batch of 10 color images (RGB), each of which is 28 pixels high and 28 pixels wide. 

> In PyTorch, this batch would be represented as a 4-dimensional tensor with the shape (10, 3, 28, 28):




  
- **NHWC** (Batch, Height, Width, Channel)


-------------------------------------------------------------------------------

### **Create a tensor with all zero and one**
- **create a tensor of all zeros**
```zeros = torch.zeros(size=(3,4)) ```

- **create a tensor of all ones**
```ones = torch,ones(size = (3,4))```

### **For check the type of tensor**
```tensor_name.dtype```

A [torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html) is an object that represents the data type of a tensor, Pytorch has 12 different data type 
- **torch.float32** (or torch.float): This is the default data type for floating-point numbers. It's used commonly for most computations <br>
> to know more about change number to [float32 type representative](https://www.youtube.com/watch?v=8afbTaA-gOQ) 
- **torch.float64** (or torch.double): This data type provides double precision floating-point numbers
- **torch.float16** (or torch.half): It provides half precision floating-point numbers, which are useful for reducing memory usage and improving performance on some GPUs
- **torch.int8**: This type represents signed 8-bit integers
- **torch.uint8**: It represents unsigned 8-bit integers
- **torch.int16** (or torch.short): This type is for signed 16-bit integers
- **torch.int32** (or torch.int): This data type represents signed 32-bit integers
- **torch.int64** (or torch.long): This type is used for signed 64-bit integers. It's often used for indexing tensors
- **torch.bool**: This type represents Boolean values (True or False)
- **torch.complex64**: This type is used for complex numbers with float32 real and imaginary parts
- **torch.complex128** (or torch.cdouble): This type is for complex numbers with float64 real and imaginary parts
- **torch.bfloat16:** This is a 16-bit data type used for floating-point numbers, offering higher precision than float16, especially in the range critical for deep learning
- **other four quantized data types** 


### **Create a range of tensor**
```torch.arange(start, end+step, step =1)```

```torch.arange(start = 1, end = 11, step =1)```

```one_to_ten = torch.arange(1 , 11)```

```zero_to_ninety = torch.arange(0,100,10)``` --> tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

### **Create a range of tensor-like**
tensor_like is use for creating a tensor with the size same as other specufy tensor

```ten_zeros = torch.zero_like(input = one_to_ten)```

the ten_zeros tensor will have the same size as one_to_ten tensor

          input = torch.empty(2, 3)
          torch.zeros_like(input)
  >tensor([[ 0.,  0.,  0.],<br>
               [ 0.,  0.,  0.]])

### **Dealing with Tensor datatypes, Device, Shape**

**big errors you'll run into with Pytorch & Deep Learning:** <br>

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

**3. Tensor shape**
  - the attribute used for checking tensor shape
```tensor_name.shape``` 
   
(addition) requires_grad = False <br>
   requires_grad (Default: True) <br>
   to specify whether or not to track gradients with this tensors operation, we set True if autograd should record operations on this tensor

For more reading: [Precision in Computing](https://en.wikipedia.org/wiki/Precision_(computer_science))

 ### **Manipulating Tensor (Tensor Operations)**

For neural network, it has a lot of mathematic operations that Pytorch run behide the scence for us
**Tensor operations include:**
- **Addition**

```created_tensor + 10```
or 
```torch.add(created_tensor, 10)```

- **Substraction**
  ```created_tensor - 10```

- **Multification** <br>

**These are two main ways of performimg multification in neural networks and deep learning:**

 

**Define the matrices**

      import numpy as np
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

  ```torch.matmul(tensor,tensor)``` or  ```torch.mm(tensor,tensor)``` or 
  ```tensor @ tensor```

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
#### **Reshaping** : reshapes an input tensor to a defined shape

```existing_tensor.reshape(1,9)```

**NOTE 1** 
The Pytorch tensor can be reshaped if it has a consistency of total elements: The total number of elements in the tensor must remain the same before and after the reshape operation, this means the product of the tensor's dimensions must be the same before and after reshaping. 

**For example,** a tensor of shape (4, 5) with 20 elements can be reshaped to (2, 10), (10, 2), (20,), etc., but not to (3, 7) as that would require 21 elements

**NOTE 2** <br>
**Examples of -1 Usage in reshape** <br>
- Creating a 1-Dimensional Tensor: <br>


       tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]) 
       reshaped_tensor = tensor.reshape(-1)
       reshaped_tensor will be: [1, 2, 3, 4, 5, 6]

      For this, reshaped_tensor is 1-dimensional because -1 is the only value in the reshape argument

 -  Reshaping to a 2-Dimensional Tensor with One Inferred Dimension:


        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])  
        reshaped_tensor = tensor.reshape(3, -1)
        reshaped_tensor will be: [[1, 2], [3, 4], [5, 6]] #3x2 since 6 = 3X then x =2 and dim = 2 because there are two value as a size

- Reshaping to a Higher-Dimensional Tensor:


      tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
      reshaped_tensor = tensor.reshape(2, 2, -1)
      reshaped_tensor will be: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
      reshaped_tensor = 2x2x2 since 2x2xX= 8 then x=2 and dimension is 3

  
  
#### **View** : Return a view of an input tensor of certain shape but keep the same memory as the original tensor
this attribute is quite similar to shape, but it shares memory with the original tensor before viewing, this means if the viewed tensor change any element, the element in the original tensor will change follows the viewed tensors

```existing_tensor.view(1,9)```

**NOTE**
- view is used when you want to change the shape of a tensor and you are sure that the tensor is contiguous **in memory**. If the tensor is not contiguous, using view will result in an error.
- contiguous refers to how the data of a tensor is stored in memory. Specifically, a tensor is considered contiguous when its elements are stored in a continuous, unbroken block of memory in the order that they are indexed
  
   - These examples will demonstrate how certain operations can lead to non-contiguous tensors and how to check and handle this situation.
     
      **Example 1**: Creating a Non-Contiguous Tensor with Transpose <br>
 Transposing a tensor often results in a non-contiguous tensor because the operation  changes the memory layout without actually moving the data in memory

           import torch
          #Create a simple 2x3 tensor
          tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

          #Transpose the tensor
          transposed = tensor.t()

          #Check contiguity
          print("Original tensor contiguous:", tensor.is_contiguous()) --> True
          print("Transposed tensor contiguous:", transposed.is_contiguous()) --> False

       **Example 2**: Slicing a Tensor, Slicing a tensor can also lead to a non-contiguous tensor. This is because slicing changes the           way elements are accessed in memory

          # Slice the tensor
          sliced = tensor[:, 1]
          # Check contiguity
          print("Original tensor contiguous:", tensor.is_contiguous()) --> True
          print("Sliced tensor contiguous:", sliced.is_contiguous()) --> False

        **NOTE** Using ```Tensor.contiguous()``` to Make a Tensor Contiguous




#### **Stacking** : combine multiple tensors on top of each other (vstack) or side by side (hstack)

- ```torch.vstack``` stacks tensors vertically, which is equivalent to concatenation along the row, it's useful when you want to stack tensors on top of each other
- ```torch.hstack``` stacks tensors horizontally, which means concatenation along the columns. It's useful when you want to place tensors side by side
- These functions are particularly useful for manipulating and combining data of the **same dimensionality** in machine learning and data preprocessing tasks
- **Example**

          import torch

          # Create two 2D tensors
          tensor1 = torch.tensor([[1, 2], [3, 4]])
          tensor2 = torch.tensor([[5, 6], [7, 8]])

          # Vertically stack the tensors
          vstacked_tensor = torch.vstack([tensor1, tensor2])

         # Horizontally stack the tensors
          hstacked_tensor = torch.hstack([tensor1, tensor2])


 ![Tensor stack](https://github.com/RadchaneepornC/DeepLearning/blob/b719f36332b3aa3e9cbd9d8a11b9a5eff016b4da/images/stack%20example.png)

#### **Squeeze**: removes all 1 dimentions from a tensor

            torch.squeeze(input, dim=None)
            input = (A×1×B)
            squeeze(input, 0) #tensor unchanged
            squeeze(input, 1) #tensor shape becomes (A×B)

When dim is given, a squeeze operation is done only in the given dimension(s).

             x = torch.zeros(2, 1, 2, 1, 2)
             x.size()  
             
  > torch.Size([2, 1, 2, 1, 2])

             y = torch.squeeze(x)
             y.size()  

  > torch.Size([2, 2, 2])

             y = torch.squeeze(x, 0)
             y.size()

  > torch.Size([2, 1, 2, 1, 2])

             y = torch.squeeze(x, 1)
             y.size()
             
  >torch.Size([2, 2, 1, 2])

             y = torch.squeeze(x, (1, 2, 3))
             
  >torch.Size([2, 2, 2])


#### **Unsqueeze**: add a 1 dimention to a target tensor

          x = torch.tensor([1, 2, 3, 4])
          x, x.size(), x.ndim

> (tensor([1, 2, 3, 4]), torch.Size([4])), 1)

          x1 = torch.unsqueeze(x, 0) #same as torch.unsqueeze(x, -2)
          x1, x1.size(), x.size(), x1.ndim
          
> (tensor([[1, 2, 3, 4]]), torch.Size([1, 4]), torch.Size([4]), 2)

          x2 = torch.unsqueeze(x, 1) #same as torch.unsqueeze(x, -1)
          x2, x2.size(), x2.ndim

> (tensor([[1],
         [2],
         [3],
         [4]]),
 torch.Size([4, 1]),2)

The **squeeze and unsqueeze** operations in PyTorch are quite useful in deep learning for manipulating tensor shapes, which is a common requirement in various stages of building and training neural networks 

 **Squeeze**:
 - **Removes Single-Dimension Entries**: squeeze removes dimensions of size 1 from the tensor's shape. This is often needed after certain operations that might introduce singleton dimensions.
- **Useful in Convolutional Neural Networks (CNNs)**: After convolutional layers and pooling layers, you might end up with tensors that have singleton dimensions. For example, a tensor might have a shape like (batch_size, num_channels, height, width, 1). The squeeze operation can be used to remove the last dimension, simplifying the tensor to (batch_size, num_channels, height, width).
- **Batch Processing**: In scenarios where the batch size might be 1 (especially during inference), squeeze can be used to remove the batch dimension, transforming a tensor from (1, features...) to (features...).

**Unsqueeze**
- **Adds a Singleton Dimension**: unsqueeze adds a dimension of size 1 at a specified index. This is useful for shaping tensors to meet the requirements of certain layers or operations.
- **Preparing Tensors for Batch Processing**: Deep learning models often expect inputs in a batch format, even when you're processing a single item. unsqueeze can be used to add a batch dimension to a tensor, transforming a tensor from (features...) to (1, features...).
- **Expanding Tensor for Broadcasting**: In operations involving multiple tensors, PyTorch can automatically broadcast tensors of different shapes. Using unsqueeze to add dimensions can facilitate such broadcasting by aligning the dimensions of the tensors.
- **Compatibility with Certain Layers**: Some layers, like certain kinds of RNNs, expect inputs with specific dimensionalities. unsqueeze can help reshape tensors to match these expectations.

#### **Permute**: Rearrange the dimensions of a target tensor in a specified order <br>
It is essential for preparing data for models, making tensors compatible with different neural network architectures, and aligning data to the expectations of various deep learning frameworks.

          x = torch.randn(2, 3, 5)
          x.size()
          
> torch.Size([2, 3, 5])

        torch.permute(x, (2, 0, 1)).size()  #shifts axis 0-->1, 1-->2, 2-->0

> torch.Size([5, 2, 3]) 


**Scenario of using torch.permute**

- **Rearranging Tensor Dimensions**:
if you have a tensor with shape (batch_size, height, width, channels) and you need it in the format (batch_size, channels, height, width), you can use permute to rearrange the dimensions

- **Data Preprocessing**
In case of an image data might be in the format (height, width, channels), but convolutional neural networks (CNNs) in PyTorch usually expect the format (channels, height, width). permute is used to reorder these dimensions

- **Compatibility with Different Deep Learning Frameworks**
Different deep learning frameworks might expect data in different formats. When moving data between frameworks, or when using models trained in different frameworks, permute is used to align the tensor formats

- **Time Series and Sequence Data**: In the context of recurrent neural networks (RNNs) and other sequence models, sometimes it's necessary to rearrange the dimensions of the input tensor, for example, from (batch_size, seq_len, features) to (seq_len, batch_size, features)






  



