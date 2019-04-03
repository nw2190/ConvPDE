# Poisson Equation on Varying Domains
This problem setup consists of solving the Poisson equation with homogeneous Dirichlet boundary conditions on varying domains:

<p align="center">
  <img width="250" src="../figures/Poisson_Eq.png" style="margin: auto;">
</p>

In particular, the domain `\Omega` and source term `f` are permitted to vary, and the convolutional network is trained to learn the mapping between these input terms and the target solution `u`.


## File Overview

#### `flags.py`
Provides training options and specifies hyperparameter values for the model.  Key flags include:
* `--training_steps` - number of training steps/iterations
* `--learning_rate` - learning rate for Adam optimizer
* `--dropout_rate` - dropout rate used to avoid overfitting
* `--use_prob_loss` - use the probabilistic loss function for training

#### `Train_Model.sh`
Bash script with a collection of predefined configurations available for training.  In particular, the script provides (1) probabilistic loss, (2) MSE loss, (3) MSE loss without boundary term, (4) probabilistic loss using Laplace distributions, and (5) probabilistic loss using Cauchy distributions.

#### `main.py`
Specifies the workflow for training the neural network models.

#### `base_model.py`
Defines all non-architecture components of the model including the loss function as well as the training loop and network evaluation metrics.

#### `Networks/network_1.py`
Defines the neural network architecture for the model.  In particular, this file provides the `self.encoder`, `self.decoder`, and `self.evaluate` methods used in the `base_model.py` file.

#### `convolutional_layers.py`
Provides custom wrappers for various network layers.

#### `utils.py`
Provides the data parsing function and various utilities for training.


## Training Model
Once the dataset has been created following the instructions provided in `Setup/README.md`, the convolutional model can be trained via:


```console
$ python main.py --model_dir Model_1 --network 1 --use_prob_loss --training_steps 100000 
```

Additional training flags can be passed as prescribed in the `flags.py` files in each problem subdirectory; default values for various training modes are provided in the `Train_Example.sh` bash file.  The models are indexed by integers and store checkpoints, logs, and configuration files in the `Model_*/` subdirectories.

Training is automatically resumed from the last checkpoint if any exist in the `Model_*/Checkpoints/` subdirectory; these checkpoints must be deleted before retraining a model with a modified network architecture.  The automatic resume functionality can, however, be used to evaluate the current training/validation losses on the complete dataset at any point during training; to accomplish this, the training procedure is simply interrupted using the `Ctrl` + `C` command, and the `python main.py` command is run with the `--training_steps` flag set to any number less than the current step.


### TensorBoard

The training progress can be monitored using TensorBoard:

```console
$ tensorboard --logdir Model_1/logs/
```
