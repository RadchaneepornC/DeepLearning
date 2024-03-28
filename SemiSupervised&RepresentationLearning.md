# Flavors of supervision

As there are several types of supervision as listed below

- **Supervised** : All labels known 
- **Semi-supervised** : Some labels known
- **Unsupervised** : No labels
- **Representation learning**
  - Transfer learning
  - Self-supervised learning: surrogate task from pseudo labels
- **Reinforcement learning** : Agent gathers data, reward function known

So, this note will focus on Semi-supervised and Representation learning, and self-supervised types
    
## Semi-supervised and Representation learning

<ul><details>
<summary> Semi-supervised </summary>
Some labels known

### Idea 1: Bootstrapping with self-training

- Use a trained classifier on new data to get fake labels
- Filter data with high scores (confidence estimate)
- Train/ adapt with filtered data

**Caveats:**

 - Scores cannot be trusted (ranking can be trusted)
 - Learning from same errorful data
 - If new data is quite different from the data used to train the classifier, classifier performance is bad 

### Idea 2: Tri-training

- Train 3 models with different subset of source data, if two models agree on a label on target data, add this data to train the third
- If all three models agree, it might be an easy data point, not so useful, only use data when two agree and the third disagree, by putting the
  data to the third people who disagree to learn them **(Tri-training with disagreement)**


### Note on label
- Psuedo labal (further reading on [Google 's Pseudo Label Is Better Than Human Label](https://arxiv.org/pdf/2203.12668.pdf))
- Soft label: A:80% B:20% this is the new way of labels that modern techniques try to use instead of hard label (k-mean vs GMM)


</ul></details>



<ul><details>
<summary> Representation Learning </summary>
        
 ## **Finding a magical function f()**  

###  use a supervised model and extract hidden values from the network <br>
    
**Drawbacks**: need labelled data

### use unsupervised (Autoencoder)

the concept is encoding itself then decoding itself, after that train network with L2 loss calculated from input and output as picture shown below

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/Autoencoder.png)

**There are many ways to help with a supervised task e.g. recognition**

- Append the input with the code from the encoder
- Stick a classifier on top of the encode (can even be a linear classifier-liner probe)
- Used for pretaining a network <br>
  



<ul><details>
    <summary> Transfer Learning </summary>
  
- The concept of transfer learning is utilizing the trained network captured good representation to initialize a new network for a different tasks
- The way we take the networks trained on a different domain for a different source task to adapt it to our domain for our target tasks called **fine-tune**
        
</details></ul>
</ul></details>

##  Self-supervised learning
<ul><details>
<summary> Self-supervised learning </summary>
Some labels known

Surrogate task from pseudo labels<br>

the concept is 
- adding noise and want the model to answer the same answer
- unsupervised data use consistency concept, learn by supervised loss

**GPT** is one of self-supervised learning that train to predict next word

 ### Contrastive Learning 

Disclaimer: multiple communities working on similar concepts but different names

- **Consistency training**

Use data to predict something obtained loss, and bring the same data pass the augmentation process to get the loss, we want these two loss having the same(consistency) because it comes from the same data, in the other words: "things same in the input, should same in the output" 

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/ConsistencyTraining.png)


- **Contrastive training**
Get rid of different things 

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/ContrastiveTraining.png)


### Deep face verification

#### 1. use **PCA**
#### 2. [Recently] use **Contrastive Learning** to make a function(neural network) that change face image to vector

- **Triplet loss(2015)**:
Want eucidian distance between Positive and Anchor less than distance between Negative and Anchor, because we want to minimize the triplet loss that calculate from this formula and the larger negative diff term, the smaller the triple loss

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/TripletLoss.png)

- **NCE (Noise constastive estimation) loss(2015)**
  <br>

  Random negative sample to calculate
> Max LogP(data) - Log P(noise or negative samples)

**Ex** If the data is "This is a pen"

we want **P(pen) > P(pencil)**, this is the same as concept used to train word embedding such as Word2Vec, too many classes in the softmax output

- **InfoNCE (2018)** 
    - conduct a softmax with an amount of minibatch, it is quite similar to N-pair loss (2016)
    - use a softmax-like function to keep positive together and push negative away

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/InfoNCE.png)

- **Soft Nearest Neighbor Loss(2019)**
    - can have more than positive sample in minibatch
![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/SoftNearestNeighborLoss.png)


- **Angular Margin/ Consine distance** Aspect


- **Centor loss(2016)**

![ALt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/CenterLoss.png)
 
**Key details to make Contrastive Learning work**

1. Large batch size make this technique works since the more negative sample, the more precise of direction
2. Temperature need to be tuned, normally value more than 1
3. Hard/semi-hard negative mining is matters (Hard negative means the sample that really close to positive), the harder negative, the better model is trained
4. Augmentation on the anchor and positive (consistency training)
5. Other improvement includes - adding classification loss (CE/softmax loss)


</ul></details>



