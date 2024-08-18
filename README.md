# Geometric analysis with machine learning: summer project with Imperial College, London
Summer project in collaboration with Meg Dearden-Hellawell, supervised by Daniel Platt (Imperial College, London, maths department) and Daattavya Argarwal (Cambridge University, computer science department). This repository concerns the second of two projects, focusing on using PINNs to find a harmonic-one form on the three torus for different metrics.

## Contents:
##### requirements.txt:
- text file containing required software to run files in this repository.

### Pre-project work:

##### simple_PINN.py:
- implementing a simple physics informed neural network, as in [this paper (1)](https://arxiv.org/abs/1711.10561), as an eductional exercise.

### Project: Harmonic 1-forms on T<sup>3</sup>
Does there exist a metric on the 3-dimensional torus T<sup>3</sup> such that every harmonic 1-form has a vanishing zero? The following python files follow on from each other, and are more complicated versions of the previous document.

##### T3_PINN_iden_metr.py:
- implemented a PINN on T3 that calculated loss through setting the laplacian of the 1-form learnt to zero, but only for an indentity metric, which simplifies the problem.

##### T3_PINN_cons_metr.py:
- generalised T3_PINN_iden_metr.py to be able to take any general metric of constants, through implementing a more rigorous definition of Hodge star.
- polished up the document so it can be followed more clearly (not necessarily the most concise, but useful for working with).

##### T3_PINN.py:
- slightly changed the architecture so it calculates loss point by point, which allows the metric to be dependant on the inputs.
- implemented a 3D extension to a metric set out by [Kerofsky (2)](https://www.researchgate.net/publication/34310555_Harmonic_forms_under_metric_and_topological_perturbations).
- added a zero checker which seeks for the smallest norm vector.

## References:
(1) [Raissi et al., 2017, *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*](https://arxiv.org/abs/1711.10561) <br/>
(2) [Kerofsky, 1995, *Harmonic forms under metric and topological perturbations*](https://www.researchgate.net/publication/34310555_Harmonic_forms_under_metric_and_topological_perturbations)
