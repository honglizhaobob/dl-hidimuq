Materials
---
Find the paper [here](https://arxiv.org/abs/1912.08081).

Assumptions
---
* small Gaussian noise, lossless system, low temperature, 

Model description
---
* 3 types of buses in connectivity graph: generator, load, reference buses.
* Swing equations with stochastic fluctuations around a synchronous point. Port-Hamiltonian dynamics, which has general form:
$$\dot{x_t} = -K\nabla\mathcal{H}^y(x_t), x_0 = \overline{x}$$, where $\mathcal{H}^y(\cdot)$ is the system's energy function with parameters $y$.

* Additive stochastic perturbations to active and reactive power, yielding the stochastic port-Hamiltonian (pH) model:
$$dx_t^{\tau} = (J-S)\nabla\mathcal{H}(x_t^{\tau}) + \sqrt{2\tau}\cdot S^{1/2}dW_t$$.








