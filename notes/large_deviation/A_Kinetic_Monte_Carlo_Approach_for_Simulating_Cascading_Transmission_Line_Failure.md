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
	- We are interested in the event when the line energy $\Theta_l(x)$, exceeds a safety threshold (a constant): $\Theta_l^{\text{max}}$. Paper simulates this event as the first exit time, i.e., simulate the process $x_t^{\tau}$, kill the process once its energy has hit the boundary of the event. 
	- In small noise limit, $\tau\rightarrow 0$, the log mean of first exit time converges to the Freidlin-Wentzell quasipotential. In particular, the first exit time follows an exponential distribution with parameter $\lambda^{\tau}$ that can be approximated. 
	- equation (2.13) shows the mean first exit time is of order $\exp(V(\overline{x},x^*)/\tau)$ where $x^*$ can be solved by the following nonlinear program:
$$x^* = \text{argmin}_{x\in \partial D_l}\mathcal{H}(x)$$.





