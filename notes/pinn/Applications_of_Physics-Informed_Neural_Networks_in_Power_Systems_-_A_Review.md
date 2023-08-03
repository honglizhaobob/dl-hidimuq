Materials
---
* Find the paper [here]


Motivation
---
* This is a survey paper that describes aspects on PINNs particularly for power system problems. The core design of PINNs include:
	- physics informed loss
	- physics informed initialization
	- physics informed architecture
	- hybrid physics-DL models

* The goals of such design considerations include:
	- improved prediction accuracy of **certain** applications (not a panacea for all problems)
	- more interpretable
	- physically consistent results (e.g. conservation law)
	- more generalizable
	- better training efficiency (e.g. convergence)
	- reduced reliance on large amounts of training data


Physics-informed loss functions
----
* Physics regularized loss function:
$$\mathcal{L} = L(y,\hat{y}) + \lambda R(\mathbf{W}, \mathbf{b}) + \gamma \underbrace{R_{\text{Phy}}(\mathbf{X}, \hat{y})}_{\text{physics loss}}$$

* An example is the single-machine infinite bus system:
$$\xi\frac{\partial^2\delta}{\partial t^2} + \kappa\omega + BV_gV_e\sin(\delta) - P = 0$$


Physics-informed initialization
----
Transfer learning, train the PINN on a related physical problem, then re-train with weights initialized at the pre-trained weights, for a new yet related problem. 

Physics-informed architecture
----
This means modifying neuron network design to match a certain physical process/domain knowledge. Three examples are mentioned:
* Use of RNN for lake-temperature prediction
* Fixing weights in seismic wave propagation
* Graph neural networks

For Fokker-Planck equations, this would mean letting the output layer of the solution neural net be strictly non-negative (e.g. use ReLU activation in the output layer). Though I think this would limit the NNs precision and convergence speed. Conservation (integrates to 1 for all $t$)
$$\int_{-\infty}^{\infty}u_{\theta}(t,x)dx = 1$$ is tricky. I've tried doing this with the physics-informed loss part, but the training would become extremely slow since now you would need to feed in all spatio-temporal points. Maybe we can do it with the architecture part. Come back to this later ...

Hybrid physics-DL
---
* Three points: (1) physics informed feature engineering (2) hybrid learning of black box and white box models (not clear to me at this point how this is done, but [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0021999115007524) they pointed to seems interesting, my understanding is you replace part of the unknown terms in your PDE with a DNN, trained separately, then you solve the PDE). 
> One example can be found in [22], where a DNN is employed to estimate unknown variables in the turbulence closure model to compensate for omissive physical details ...

(3) ensemble learning, concatenate results of both a physical model and a DL model to make a "joint prediction" (I don't understand what you mean by "concatenate", but I should come back to this later).


Comments
---
The rest of this paper discusses about various power system applications that I only skimmed through. It seems to be describing general deep learning approaches (e.g. reinforcement learning, VAE, generative adversarial neural nets) for power system problems, rather than specifically PINNs. In the end, it seems to have become a concoction of terms without explaining the underlying connection.

> Quantum computing (QC), which depends on quantum entanglement and quantum superposition, is expected to achieve overwhelming superiority in computational efficiency in some specific applications. As a cutting-edge research area, quantum ML explores the interplay of ideas from QC and ML and extends the implementation of ML algorithms on a universal or near-term quantum computer. As a form of hybrid physics-DL models, the differentiable quantum circuits can be devised to mimic and replace some DL components, thereby tackling problems in the power and energy industry.

The above paragraph is kind of just there. 

