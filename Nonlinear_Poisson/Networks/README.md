# Network Architecture
The convolutional network architectures are defined in the `Networks/network_*.py` files in each problem directory.  The default `Networks/network_1.py` architecture corresponds to an encoder-decoder network as illustrated below:

<p align="center">
  <img width="750" src="../../figures/Architecture.png" style="margin: auto;">
</p>

Alternative network architectures can be defined in `Network/network_*.py` files and used during training by passing the `--network *` flag (where `*` denotes an integer value used to label the network architecture).


## Loss Functions

### MSE Loss
The conventional mean squared error (MSE) loss function is used by default:

<p align="center">
  <img width="750" src="../../figures/MSE_Loss.png" style="margin: auto;">
</p>

The loss weight `\lambda` for the boundary term can be modified using the `--bdry_weight` flag; in particular, specifying `--bdry_weight 0.0` will omit the boundary component on the loss calculation.

### Probabilistic Loss
A probabilistic training procedure can be employed by using the `--use_prob_loss` flag.  This instructs the network to make both mean and standard deviation predictions and defines the loss function to be the negative log marginal likelihood (NLML) of the true solution values with respect the predicted statistics:

<p align="center">
  <img width="750" src="../../figures/PROB_Loss.png" style="margin: auto;">
</p>

