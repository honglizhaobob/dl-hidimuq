For anyone interested in learning about physics-informed neural networks, or general deep learning methods for solving PDEs, particularly Fokker-Planck type equations, I am sharing my list of readings here. These notes are inspired by [Denny Britz](https://github.com/dennybritz/deeplearning-papernotes), [Daniel Seita](https://github.com/DanielTakeshi/Paper_Notes), and [Jiahao Yao](https://github.com/JiahaoYao/Paper_Notes). You might also see some domain-specific papers here (e.g. power systems), but the central ideas should belong to the same cluster. Also, these notes may include some of my personal opinions, if any of them do not reflect the reality, please let me know.

The timeline for these readings roughly covers my time as a Givens Associate at the MCS division of Argonne Lab, and continuing. April 2023 -- Present.

There exist some very important papers that everyone wanting to study this field should read. I link them below. Also, I will frequently refer to the terms "forward problem" and "inverse problem" in my notes. Forward problem with PINNs generally mean an unsupervised task [^1] of minimizing the PDE loss (on selected collocation points) and the boundary loss (initial and boundary conditions need to be exactly matched, or data loss for the boundaries). Inverse problem on the other hand refers to discovering unknown parameters from already computed (possibly low-fidelity) PDE solutions. In this case, one would minimize the PDE loss at collocation points, and data loss at known locations. 

[^1]: Perhaps unsupervised is also a bit inaccurate. Unsupervised means "provide no labeled data", however, we would need at least initial and boundary conditions. 







