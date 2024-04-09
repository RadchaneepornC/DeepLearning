# Deep Generative Model

- Generative modelling learn P(X,Y) or P(X|Y)
  - This generates X from P(X|Y) or a parametric distribution, while Y is the controlling parameter
- This type of model is different from Discrimative models in which they learn from (P(Y|X))

Deep generative models that can learn via the principle of maximim likeli- hood differ with respect to how they represent or approximate the likelihood
Let's deep dive through some types of Deep Generative Model as below

## Variational Autoencoder (VAE)

This is the use of method of maximum likelihood  using approximate density (ELBO) and use variational to approximate ELBO






Ref: https://arxiv.org/pdf/1701.00160.pdf

**Original and by-product motivations**

0. Learning a non-deterministic mapping

Instead of predicting Encoder output as the deterministic vector or standard AE, we predict means and variance(Std.) of Gaussian distribution (Probabilistic), after that random(sample) the number 
and bring that number to decode, as the picture below

![Alt text](https://github.com/RadchaneepornC/DeepLearning/blob/main/images/DeterministicAE%26NonDeterministicAE.png)
   
2. Allows approximation of intractable posterior

   
4. Generate photos from the decoder


## GAN




## Diffusion model

## Flow-based model
