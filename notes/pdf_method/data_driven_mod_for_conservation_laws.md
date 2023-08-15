Materials
---

Find paper [here](https://epubs.siam.org/doi/10.1137/19M1260773).

Main ideas
---
* Hyperbolic conservation laws with randomness in initial condition or parameters. In particular, the CDF method is used to derive a PDE (governing PDF) for the solution of a hyperbolic conservation law $u(\mathbf{x}, t)$. 

* In the presence of discontinuity in solutions (i.e. shock), a weak formulation can be considered. A "kinetic defect" source term is introduced to account for presence of shocks.

* Error distributions may be highly non-Gaussian, such that Kalman filters would not work well. 
	- Technique 1: data assimilation. The kinetic defect is learned from data through Newtonian relaxation / "nudging". 
	- Technique 2: neural-network ("instantaneous relaxation"). Learn the kinetic defect as a DNN. 

* Once the kinetic defect is learned (interpolation), one may predict solution at future times (extrapolation)






