# Parallel MCMC Algorithms

This repository contains implementations of three Markov Chain Monte Carlo (MCMC) algorithms:
- **Metropolis**
- **Metropolis-Hastings (MH)**
- **Hamiltonian Monte Carlo (HMC)**

Each algorithm is parallelized to run multiple chains simultaneously, and uses the **Rhat** statistic to check for convergence. 

The HMC method uses the leapfrog integration method to simulate Hamiltonian dynamics.

The **1D Gaussian** is used as the example case for each algorithm.
