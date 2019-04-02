# Poisson Equation on a Circle
This problem setup consists of solving the Poisson equation with homogeneous Dirichlet boundary conditions on a fixed circle:

<p align="center">
  <img width="250" src="../figures/Poisson_Eq.png" style="margin: auto;">
</p>

In particular, the domain `\Omega` is taken to be a fixed unit circle while the source term `f` is permitted to vary; the convolutional network is trained to learn the mapping between these input terms and the target solution `u`.

## Domain Format
The mesh file for the unit circle is encoded as an integer-valued array with values of zero outside of the domain, values of one throughout the interior of the domain, and values of two along the boundary:

<p align="center">
  <img width="300" src="../figures/domain.png" style="margin: auto;">
</p>

This file is used to adapt the network's loss function to the domain in consideration; in particular, the loss is instructed to ignore the values of predictions outside of the domain and is capable of handling boundary and interior predictions separately.
