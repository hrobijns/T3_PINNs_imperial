# Geometric analysis with machine learning: summer project with Imperial College, London
Summer project in collaboration with Meg Dearden-Hellawell, supervised by Daniel Platt (Imperial College, London, maths department) and Daattavya Argarwal (Cambridge University, computer science department). This repository concerns the second of two projects, focusing on using PINNs to find a harmonic-one form on the three torus for different metrics.

## Contents:
##### requirements.txt:
- text file containing required software to run files in this repository.

### Pre-project work:

##### simple_PINN.py:
- implementing a simple physics informed neural network, as in [this paper (1)](https://arxiv.org/abs/1711.10561), as an eductional exercise.

### Project: Harmonic 1-forms on T<sup>3</sup>
Does there exist a metric on the 3-dimensional torus T<sup>3</sup> such that every harmonic 1-form has a vanishing zero?

##### T3_PINN_iden_metr.py:
- implemented a PINN on T3 that calculated loss through setting the laplacian of the 1-form learnt to zero.

##### T3_PINN.py:
- generalised T3_PINN_iden_metr.py to be able to take any general metric, through changing the definition of Hodge star.
- polished up the document so it can be followed more clearly.




## References:
(1) [Raissi et al., 2017, *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*](https://arxiv.org/abs/1711.10561)
