# Nonlinear Poisson Equation on Varying Domains
This problem setup consists of solving the following nonlinear version of the Poisson equation with homogeneous Dirichlet boundary conditions on varying domains:

<p align="center">
  <img width="400" src="../figures/Nonlinear_Eq.png" style="margin: auto;">
</p>

In particular, the domain `\Omega` and source term `f` are permitted to vary, and the convolutional network is trained to learn the mapping between these input terms and the target solution `u`.

