#  Data-driven Transportability

This repository contains the code for our work: [Transportability without Graphs: A Bayesian Approach to Identifying s-Admissible Backdoor Sets](https://arxiv.org/pdf/2505.12801) 

## Overview
Our method combines experimental data (De) from the source distribution with observational data (Do*) from the target distribution to compute the probability that a causal effect is both identifiable from observational data and transportable. When this holds,
we leverage both observational data from the target domain and experimental data from the source domain to obtain an unbiased estimate of the causal effect in the target environment. 

FindsABS uses Bayesian regression models and approximate inference for combining De and Do* to estimate the probability of a set to be a backdoor and an s-admissible set and provide conditional post-interventional prediction for the outcome 
of the target population when possible. 

It can be applied to binary outcomes and binary or mixed covariates (allowing the presence of measured and unmeasured confounders).

## Getting Started
### Packages
This project uses several Python packages to perform probabilistic programming, Bayesian inference, and efficient array computations:

* JAX and JAX.numpy
* NumPyro
* itertools
* pandas

You can also visualize the posterior distributions of regression parameters using the ArviZ and Matplotlib packages by uncommenting the corresponding lines in the function "sample_posterior".
  
## Contact
If you have any questions, email me:  [konlelova@gmail.com](mailto:konlelova@gmail.com) 
