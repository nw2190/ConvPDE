# Poisson Equation on Varying Domains
This problem setup consists of solving the Poisson equation with homogeneous Dirichlet boundary conditions on varying domains:

<p align="center">
  <img width="250" src="../figures/Poisson_Eq.png" style="margin: auto;">
</p>

In particular, the domain `\Omega` and source term `f` are permitted to vary, and the convolutional network is trained to learn the mapping between these input terms and the target solution `u`.


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
$ tensorboard --logdir Model/logs/
```
