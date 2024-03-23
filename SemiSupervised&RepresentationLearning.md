# Flavors of supervision

As there are several types of supervision as listed below

- **Supervised** : All labels known 
- **Semi-supervised** : Some labels known
- **Unsupervised** : No labels
- **Representation learning**
  - Transfer learning
  - Self-supervised learning: surrogate task from pseudo labels
- **Reinforcement learning** : Agent gathers data, reward function known

So, this note will focus on Semi-supervised and Representation learning types
    
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
        <ul><details>
        <summary> Transfer Learning </summary>
        </details></ul>
        <ul><details>
        <summary> Self-supervised learning </summary>
        Surrogate task from pseudo labels
        </details></ul>





</ul></details>
