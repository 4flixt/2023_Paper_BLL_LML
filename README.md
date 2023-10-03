# Improved uncertainty quantification for neural networks with Bayesian last layer

## Introduction

Dear visitor, 

thank you for your interest in our research. The content of this repository supplements our paper "Improved uncertainty quantification for neural
networks with Bayesian last layer".
We provide all the necessary materials to recreate the presented results in our paper and to use our implementations in your own work. 
For questions about these materials we kindly ask you to sent an e-mail at [felix.fiedler@tu-dortmund.de](mailto:felix.fiedler@tu-dortmund.de) or to use the [discussions](https://github.com/4flixt/2022_Paper_BLL_LML/discussions) in this repository for more public questions.

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
4. The implementation of Bayes by Backprop to train a Bayesian neural network with Variational inference can be found [Fundamentals/03_vi_toy_example.py](https://github.com/4flixt/2023_Paper_BLL_LML/blob/main/Fundamentals/03_vi_toy_example.py). An exported version of this code (containing a snapshot of our results) is included in [Fundamentals/03_vi_toy_example.ipynb](https://github.com/4flixt/2023_Paper_BLL_LML/blob/main/Fundamentals/03_vi_toy_example.ipynb). The results are shown in:
  - Figure 5
  - Table 1

  
