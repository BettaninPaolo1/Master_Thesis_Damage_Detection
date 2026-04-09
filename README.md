This repository contains the official code implementation for my Master's Thesis: "Approximate Bayesian Inference via Normalizing Flows and Variational Autoencoders for Parameter Estimation".

The accurate estimation of physical parameters from observational data (the inverse problem) is a fundamental challenge in engineering. While Bayesian inference provides a rigorous framework for characterizing parameter uncertainty, it is often limited by the prohibitive computational cost of evaluating high-fidelity physical simulators (e.g., Finite Element Models), rendering the exact likelihood function intractable.

This project decouples the expensive simulator from the inference loop using Deep Generative Models:

Conditional Variational Autoencoder (CVAE): Performs robust dimensionality reduction, compressing high-dimensional observational time series (sensor data) into a structured latent space.

RealNVP Normalizing Flow: Acts as an exact, differentiable surrogate for the intractable likelihood function. The architecture has been enhanced with Feature-wise Linear Modulation (FiLM) layers and an expanded 8D latent space to dynamically adapt to physical conditioning.

Capitalizing on the analytical gradients provided by the RealNVP surrogate, this repository implements and compares three advanced Bayesian inference strategies:

Hamiltonian Monte Carlo (HMC) via the No-U-Turn Sampler (NUTS) - The asymptotically exact baseline.

Automatic Differentiation Variational Inference (ADVI) - The fast, optimization-based parametric approach.

Stein Variational Gradient Descent (SVGD) - The optimal particle-based hybrid, balancing exactness and computational speed.
