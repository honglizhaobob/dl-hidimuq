Materials
---
This is a survey paper, find at [this link](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125).


Thoughts and Comments
---

* Overstatement, at best not rigorous.
> Such neural networks are constrained to respect any symmetries, invariances, or conservation principles originating from the physical laws that govern the observed data, as modeled by general time-dependent and non-linear
partial differential equations. 

* Examples used:
1. Forward problem
	- Schrodinger equation
	- Allen-Cahn equation
	- 1d Burgers
2. Inverse problem
	- Navier-Stokes in 2d: $\lambda_2$ identification has an error of ~5\%, not super impressive. 
	- Kortewegde Vries Equation: errors of identifying parameters $\lambda_1,\lambda_2$ are on the order $O(10^{-2})$ (percentage), sounds impressive. 
	
* Ending the paper on a positive note,
> Admittedly, more work is needed collectively to set the foundations in this field.

* Tables A1 and A2 show that (at least for Burgers equation), increasing model complexity (by adding layers) does not seem to strictly improve accuracy for the forward problem.

* Table B6 and B7, same issue, it's unclear whether a more complex architecture is going to improve or impair accuracy for inverse problem

* Paper introduced a discrete formulation of the spatio-temporal learning by discretizing time (but keep space continuous) and learn a vector output NN as a function of $x$. This formulation is compared with the Runge-Kutta scheme
	
I notice a frequent use of periodic boundary conditions (Allan-Cahn, Schrodinger, and KdV), not sure why. Overall, PINN seems like a promising method for textbook PDE problems (if you get tired of using other numerical schemes :)	
	
	
	