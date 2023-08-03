Materials
---
* Paper link [here](https://www.sciencedirect.com/science/article/pii/S2590005621000515)


Motivation
---
* Most PDEs do not have analytical solutions --> need to use numerical methods. However, special PDEs on complex geometry / with conserved quantities are not easy to preserve --> deep learning approach. PINN = data + DNN + physics

* PINN can be considered as a form of unsupervised learning if you don't use data in the interior of the domain (i.e. only provide initial and boundary conditions, and physics). 


Comments
---
* This paper is very engineering focused (focuses on using PyTorch), and considers simple 1d PDEs. But there are some statements that I find a bit dubious, such as:

> PINNs also have the advantage of allowing one to use the PDEs-free parameters in the solution. 
> Therefore, a solution trained for various values of these parameters will generalize to different conditions rather than a single scenario, eliminating the need for new calculations any time a parameter is modified.

When the PDE parameters change, my understanding is that typically you would need to re-train (will come back to this if I'm wrong). The above statement about "eliminating the need ..." perhaps needs some clarification. 


* Equation (4) has a very interesting notation choice, $\emptyset(t,x)$.

* I think this probably means "differentiable".

> Our trained PINNs, unlike the FVM solution, are continuous and derivable over the entire domain.

* Paper argues that PINN learns a continuous model and has a computational advantage over traditional numerical methods. Sure, but don't PINNs take a long time to train to begin with? 


Thoughts and Takeaways
---
* This paper talks about the process of solving a 1d advection equation with PINN, compared accuracy with exact solution (i.e. method of characteristics) and numerical methods (FTBS, Lax-Wendroff, Lax-Friedrich). And concludes that PINN has the highest prediction accuracy. I must stress that this is only empirically corroborated, in this paper, for **1d advection equation** (i.e. it likely will not work for other PDEs, I'm not sure why the authors don't make it clear). 

* I fail to see why this paper adds novelty to the literature. This paper is strangely interesting because it is published in 2022, and PINNs have been around for some while. 
