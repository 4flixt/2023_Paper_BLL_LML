# Improved uncertainty quantification for neural networks with Bayesian last layer

## Introduction

Dear visitor, 

thank you for your interest in our research. The content of this repository supplements our paper "Improved uncertainty quantification for neural
networks with Bayesian last layer".
We provide all the necessary materials to recreate the presented results in our paper and to use our implementations in your own work. 
For questions about these materials we kindly ask you to sent an e-mail at [felix.fiedler@tu-dortmund.de](mailto:felix.fiedler@tu-dortmund.de) or to use the [discussions](https://github.com/4flixt/2022_Paper_BLL_LML/discussions) in this repository for more public questions.

## Abstract of the work 

Uncertainty quantification is an essential task in machine learning - a task in which neural networks (NNs) have traditionally not excelled. Bayesian neural networks (BNNs), in which parameters and predictions are probability distributions, can be a remedy for some applications, but often require expensive sampling for training and inference. NNs with Bayesian last layer (BLL) are simplified BNNs where only the weights in the last layer and the predictions follow a normal distribution. They are conceptually related to Bayesian linear regression (BLR) which has recently gained popularity in learning based-control under uncertainty. Both consider a non-linear feature space which is linearly mapped to the output, and hyperparameters, for example the noise variance, are typically learned by maximizing the marginal likelihood. For NNs with BLL, these hyperparameters should include the deterministic weights of all other layers, as these impact the feature space and thus the predictive performance. Unfortunately, the marginal likelihood is expensive to evaluate in this setting and prohibits direct training through back-propagation.
In this work, we present a reformulation of the BLL log-marginal likelihood, which considers weights in previous layers as hyperparameters and allows for efficient training through back- propagation. Furthermore, we derive a simple method to improve the extrapolation uncertainty of NNs with BLL. In a multivariate toy example and in the case of a dynamic system identification task, we show that NNs with BLL, trained with our proposed algorithm, outperform standard BLR with NN features.

## Structure of this repository

1. Our reusible code can be found [here](https://github.com/4flixt/2022_Paper_BLL_LML/tree/main/bll) and contains:
  - An implementation based on Tensorflow/Keras for Bayesian last layer neural networks
  - An implementation based on CasADi for Bayesian linear regression with engineered features
2. The results to investigate the enhanced extrapolation are created in the Jupyter notebook [Fundamentals/01_BLL_Inter_vs_Extrapolation.ipynb](https://github.com/4flixt/2022_Paper_BLL_LML/blob/main/Fundamentals/01_BLL_Inter_vs_Extrapolation.ipynb) and shown in the paper in
  - Figure 3
  - Figure 4
3. The BLL results for the toy example are created in [Fundamentals/02_Multvariate_BLL_Toy_example.ipynb](https://github.com/4flixt/2022_Paper_BLL_LML/blob/main/Fundamentals/02_Multvariate_BLL_Toy_example.ipynb) and shown in the paper in 
  - Figure 1
  - Table 1
4. The results for the system identification with BLL and BLR are created in [SysID/01_BLL_Sys_ID.ipynb](https://github.com/4flixt/2022_Paper_BLL_LML/blob/main/SysID/01_BLL_Sys_ID.ipynb). In this Jupyter Notebook we also import the investigated linear system (in terms of the quadrupel A,B,C,D). The results of this investigation are shown in 
  - Figure 5
  - Table 2
  
