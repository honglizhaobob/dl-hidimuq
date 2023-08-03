Materials
---
Click on this [link to paper](https://arxiv.org/abs/2201.05624). This is a survey paper that links to various other papers in the field.

Intro
---
Central design questions: what neural net architectures should be used + how to incorporate physics.

Good to know
> The concept of incorporating prior knowledge into a machine learning algorithm is not entirely novel. In fact [Dissanayake and Phan-Thien (1994)](https://onlinelibrary.wiley.com/doi/abs/10.1002/cnm.1640100303) can be considered one of the first PINNs.

The PDE takes general form:
$$\mathcal{F}(u(z); \gamma) = f(z)$$
$$\mathcal{B}(u(z)) = g(z)$$ where the first equation is the dynamics, the second being the set of boundary conditions. 

The following optimization problem is solved:
```math

\theta^* = \text{argmin}_{\theta}(w_F\mathcal{L}_F(\theta) + w_B\mathcal{L}_B(\theta) + w_d\mathcal{L}_d(\theta)

``` 
where $F,B,d$ denotes interior PDE loss, boundary loss, and data loss.

* Architectures: single feedforward neural net (shallow/DNN), multiple DNNs, recurrent network, convolutional, encoder-decoder, LSTM, Bayesian PINN, generative adversarial NN, multiple PINNs.

* Regularization by additional loss functions is a soft constraint and may not be satisfied exactly, and sensitive to weights of the losses. Hard constraint on boundary or initial conditions can also be considered. 

* Optimization: Normally uses Adam or L-BFGS, or a two-step approach (first use Adam, then switch to BFGS). Non-dimensionlization of the PDE and normalization of data are also things to try.

* Finding a global minimum is **NP-hard**:
	- [Gradient descent gets trapped in local minima](https://arxiv.org/abs/1602.04915)
	- [Finding global minimum is NP-hard](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800103)
	
* This is a good reminder for numerical analysts:

> The Lax-Richtmyer Equivalence Theorem is often referred to as a fundamental result of numerical analysis where roughly the convergence is ensured when there is consistency and stability.

* Learning theory:
	- Define error:
```math
	\mathcal{R}[\hat{u}_{\theta}] := \int_{\overline{\Omega}}(\hat{u}_{\theta} - u)^2dz
``` 

Let $\hat{\mathcal{R}}$ denote the discrete approximation to $\mathcal{R}$, or MSE. 

The trained neural net risk can be bounded by:
```math
	\mathcal{R}[\hat{u}_{\theta}^*] \le \mathcal{E}_O + 2\mathcal{E}_G + \mathcal{E}_A
```

where 
```math
	\mathcal{E}_O := \mathcal{\hat{R}}[\hat{u}_{\theta}] - \inf_{\theta\in\Theta}\hat{\mathcal{R}}[u_{\theta}]
``` 
is the optimization error. 

```math
\mathcal{E}_G := \text{sup}_{\theta\in\Theta}|\mathcal{R}[u_{\theta} - \mathcal{\hat{R}}[u_{\theta}]]|
``` 
is the generalization error. And: 
```math
\mathcal{E}_A := \text{inf}_{\theta\in\Theta}\mathcal{R}[u_{\theta}]
```
is the approximation error. Most of the statistical learning results referenced in section 2.4.3 are quite recent, from 2020 to 2022.

* We would like to solve essentially this question:
> In general, PINN can fail to approximate a solution, not due to the lack of expressivity in the NN architecture but due to soft PDE constraint optimization problems.

* Paper recommends that the weights of various loss terms be on the same magnitude. 

* The following statements correspond to empirical experiments I have done, interesting ... would need to read the referenced papers:

> In Krishnapriyan et al (2021) the authors analyze two basicPDE problems of diffusion and convection and show that PINN can fail to learn the ph problem physics when convection or viscosity coeffients are high. They found that the PINN loss landscape becomes increasingly complex for large coefficients.

And:
> PINNs have trouble propagating information from the initial condition or boundary condition to unseen areas of the interior or to future times as an iterative solver.

And:
> PINNs also fail to solve PDEs with high-frequency or multi-scale structure