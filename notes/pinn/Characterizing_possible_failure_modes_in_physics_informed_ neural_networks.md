Materials
---
[Paper github](https://github.com/a1k12/characterizing-pinns-failure-modes)

Main takeaways
---
* This paper explored using PINN to solve (the forward problem):
	- 1d constant advection
	- 1d reaction diffusion equation
where the speeds of advection or reaction are large. The paper observes that the loss landscape becomes ill-conditioned and hard to optimize. The problem not being able to solve these equations comes not from limited expressiveness of the DNN itself, but the difficulty of optimization. The paper then concludes with two recommendations and observes 1-2 orders of magnitude of improvement (in prediction relative error):
	1. curriculum training: train NN with smaller PDE coefficients, then transfer the weights to initialize NNs with gradually larger coefficient problems.
	2. sequence-to-sequence: I felt seq2seq means something else in NLP (the previous sequence would be involved in the prediction of the next sequence). But the idea here is to break time domain into small chunks, and train the PINN on each chunk, with the initial condition being the terminal prediction at the previous chunk. 



Thoughts and comments
---
* Figure 3 is very interesting even though constant advection is a very simple equation ((d) and (e) show quite a bit of "jaggedness"). Also notice the magnitude grows in powers of 10 while the coefficient is increased from 0 - 50. 
* They used TPUs?





