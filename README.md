# Improved uncertainty quantification for neural networks with Bayesian last layer

## Introduction

Dear visitor, 

thank you for your interest in our research. The content of this repository supplements our paper "Improved uncertainty quantification for neural
networks with Bayesian last layer".
We provide all the necessary materials to recreate the presented results in our paper and to use our implementations in your own work. 
For questions about these materials we kindly ask you to sent an e-mail at [felix.fiedler@tu-dortmund.de](mailto:felix.fiedler@tu-dortmund.de) or to use the [discussions](https://github.com/4flixt/2022_Paper_BLL_LML/discussions) in this repository for more public questions.

## Abstract of the work 

Uncertainty quantification is an essential task in machine learning - a task in which neural networks (NNs) have traditionally not excelled. This can be a limitation for safety-critical applications, where uncertainty-aware methods like Gaussian processes or Bayesian linear regression are often preferred. Bayesian neural networks are an approach to address this limitation. They assume probability distributions for all parameters and yield distributed predictions. However, training and inference are typically intractable and approximations must be employed. A promising approximation is NNs with Bayesian last layer (BLL). They assume distributed weights only in the last linear layer and yield a normally distributed prediction. NNs with BLL can be seen as a Bayesian linear regression model with learned nonlinear features. To approximate the intractable Bayesian neural network, point estimates of the distributed weights in all but the last layer should be obtained by maximizing the marginal likelihood. This has previously been challenging, as the marginal likelihood is expensive to evaluate in this setting and prohibits direct training through backpropagation. 

We present a reformulation of the log-marginal likelihood of a NN with BLL which allows for efficient training using backpropagation. Furthermore, we address the challenge of quantifying uncertainty for extrapolation points.  We provide a metric to quantify the degree of extrapolation and derive a method to improve the uncertainty quantification for these points. Our methods are derived for the multivariate case and demonstrated in a simulation study, where we compare Bayesian linear regression applied to a previously trained neural network with our proposed algorithm.


## Structure of this repository

1. Our reusible code can be found [here](https://github.com/4flixt/2023_Paper_BLL_LML/tree/main/bll) and contains:
  - An implementation based on Tensorflow/Keras for Bayesian last layer neural networks
  - An implementation based on CasADi for Bayesian linear regression with engineered features
2. The results to investigate the enhanced extrapolation are created in the Jupyter notebook [Fundamentals/01_BLL_Inter_vs_Extrapolation.ipynb](https://github.com/4flixt/2022_Paper_BLL_LML/blob/main/Fundamentals/01_BLL_Inter_vs_Extrapolation.ipynb) and shown in the paper in
  - Figure 3
  - Figure 4
3. The BLL results for the simulation example are created in [Fundamentals/02_Multvariate_BLL_Toy_example.ipynb](https://github.com/4flixt/2022_Paper_BLL_LML/blob/main/Fundamentals/02_Multvariate_BLL_Toy_example.ipynb) and shown in the paper in 
  - Figure 1
  - Table 1
  
