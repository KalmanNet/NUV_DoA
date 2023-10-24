# DOA Estimation Code

This repository contains the code for DOA (Direction of Arrival) estimation using the [NUV-DoA](https://arxiv.org/abs/2309.03114) (Non-Uniform Variational Expectation-Maximization) algorithm. Details of the code are introduced below.


## data

In data/Data_generation.py, generating data for this experiment consists of the following steps:
1. Generate the true DoAs, and generate its corresponding steering matrix.
2. Generate noncoherent source signals.
3. Define noise level, and generate the observations

The generated datasets are saved in this "data" folder.

## simulations

All the parameters are summarized in simulations/config.py.
Useful functions including peak finding, permuted MSE, etc are in simulations/utils.py.

## NUV.py

Implementations of NUV-SSR, its batch version and NUV-DoA.


## Main files

Setup the parameters and run simulations.

