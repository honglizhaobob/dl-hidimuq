# Learning for High-Dimensional Uncertainty Quantification

This repository contains driver codes for the data-driven reduced-order probability density function (PDF) method. Our goal is to numerically solve for the PDF of a quantity of interest (QoI) that depends on an underlying stochastic system. The modeling approach is as follows:

1. **High-dimensional stochastic system**:
   
   \[
   dX_t = \mu(X_t)dt + \sigma(X_t)dW_t, \quad X_0 \in \mathbb{R}^d \sim f_{X_0}
   \]

2. **Define the quantity of interest for a function \( u:\mathbb{R}^d \rightarrow \mathbb{R} \)**:

   \[
   U_t := u(X_t)
   \]

3. **Derive the reduced-order governing equation from the full-state Fokker-Planck equation**:

   \[
   \partial_t f_U + \nabla_u (\mathbb{E}[\mu_u(X_t) \mid U_t = u] f_U) + \nabla_u \cdot (\nabla_u \mathbb{E}[D_u(X_t) \mid U_t = u] f_U) = 0
   \]

   where \( \mu_u \) and \( D_u \) are modified drift and diffusion coefficients for the process of \( U_t \).

4. **Solve the reduced-order equation numerically using the Lax-Wendroff method (1D) and the corner transport upwind method (2D)**.

5. **Post-process the low-dimensional PDF by computing statistics and tail probabilities**.

# References

- Hongli Zhao, Tyler E. Maltba, D. Adrian Maldonado, Emil Constantinescu, Mihai Anitescu. "Data-Driven Estimation of Failure Probabilities in Correlated Structure-Preserving Stochastic Power System Models" [arXiv:2401.02555](https://arxiv.org/abs/2401.02555), 2024.

- Tyler E. Maltba, Hongli Zhao, D. Adrian Maldonado. "Data-driven Closures & Assimilation for Stiff Multiscale Random Dynamics" [arXiv:2312.10243](https://arxiv.org/abs/2312.10243), 2023.

