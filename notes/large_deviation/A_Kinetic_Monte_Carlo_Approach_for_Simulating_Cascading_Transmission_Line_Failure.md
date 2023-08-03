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
$$\dot{x_t} = -K\nabla\mathcal{H}^y(x_t), x_0 = \overline{x}$$.