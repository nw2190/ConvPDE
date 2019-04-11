from argparse import ArgumentParser

# Add learning rate / training parameters
def add_learning_args(parser):

    # Default training options
    parser.add_argument("--training_steps", default=100000, type=int, help="Total number of training steps")
    parser.add_argument("--batch_size", default=64, type=int, help="Training batch size")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Adam optimizer beta1 parameter")

    # Specify learning rate
    parser.add_argument("--learning_rate", default=0.00075, type=float, help="Initial learning rate")
    parser.add_argument("--lr_decay_step", default=10000, type=int, help="Learning rate decay step")
    parser.add_argument("--lr_decay_rate", default=0.95, type=float, help="Learning rate decay rate")    
    parser.add_argument("--lr_epoch_decay_step", default=900, type=int, help="Learning rate decay steps between epochs")
    parser.add_argument("--lr_epoch_decay_rate", default=0.5, type=float, help="Learning rate decay rate between epochs")

    # Dropout and regularization options
    parser.add_argument("--dropout_rate", default=0.045, type=float, help="Dropout rate for dense connections")
    parser.add_argument("--regularize", default=False, action="store_true", help="Option to use weight regularization")
    
    # KL divergence options
    parser.add_argument("--use_kl", default=False, action='store_true', help="Option to KL divergence")
    parser.add_argument("--use_kl_decay", default=False, action='store_true', help="Option to apply decay to KL divergence")    
    parser.add_argument("--kl_start_step", default=500, type=int, help="Step count before adjusting kl_wt for underfitting")
    parser.add_argument("--update_kl_wt_step", default=100, type=int, help="Step count for updating kl weight")

    # Early stopping options
    parser.add_argument("--early_stopping_start", default=2000, type=int, help="Starting step for early stopping checks")
    parser.add_argument("--early_stopping_step", default=10000, type=int, help="Steps between early stopping checks")
    parser.add_argument("--early_stopping_tol", default=0.000025, type=float, help="Tolerance for early stopping")
    parser.add_argument("--stopping_size", default=1000, type=int, help="Batch size for early stopping checks")
    parser.add_argument("--no_early_stopping", default=False, action='store_true', help="Omit early stopping checks")

    # Manual validation checks
    parser.add_argument("--evaluation_step", default=1000, type=int, help="Step count for saving evaluation loss")
    parser.add_argument("--no_validation_checks", default=False, action='store_true', help="Omit validation checks")
    
    return parser


# Add optimization parameters
def add_optimizer_args(parser):

    # Specify optimization algorithm [default: Adam optimizer]
    parser.add_argument("--use_AMSGrad", default=False, action='store_true', help="Option to use AMSGrad optimizer")
    parser.add_argument("--use_SGLD", default=False, action='store_true', help="Option to use Langevin Dynamics optimizer")
    parser.add_argument("--use_relative", default=False, action='store_true', help="Option to use relative error for training")
    parser.add_argument("--use_noise_injection", default=False, action='store_true', help="Option to use noise injection")
    parser.add_argument("--noise_level", default=0.01, type=float, help="Noise level when using noise injection")

    # Specify loss function weights
    parser.add_argument("--int_weight", default=1.0, type=float, help="Weight for interior loss")
    parser.add_argument("--bdry_weight", default=0.1, type=float, help="Weight for boundary loss")
    parser.add_argument("--kl_weight", default=1.0, type=float, help="Weight for kl loss")
    parser.add_argument("--reg_weight", default=0.5, type=float, help="Weight for interior loss")
    
    # Uncertainty distribution [default: normal distribution]
    parser.add_argument("--use_prob_loss", default=False, action='store_true', help="Option to negative probability loss")
    parser.add_argument("--use_laplace", default=False, action='store_true', help="Option to Laplace distribution")
    parser.add_argument("--use_cauchy", default=False, action='store_true', help="Option to Cauchy distribution")

    # Use alternative log implementation [default: softplus uncertainty scale]
    parser.add_argument("--use_log_implementation", default=False, action='store_true', help="Use log implementation for prob loss")

    return parser


# Add dataset details
def add_data_args(parser):

    # Specify total count of available data
    parser.add_argument("--data_count", default=100000, type=int, help="Total number of data points available")
    parser.add_argument("--rotate", default=False, action='store_true', help="Option to rotate dataset for data augmentation")
    parser.add_argument("--flip", default=False, action='store_true', help="Option to flip dataset for data augmentation")
    parser.add_argument("--dataset_step", default=500, type=int, help="Step between dataset cycles")
    parser.add_argument("--data_files", default=0, type=int, help="Number of *.tfrecords training data files to use")

    # Training batches per epoch  ( e.g. 0.8*100000/64 )
    parser.add_argument("--batches_per_epoch", default=1250, type=int, help="Number of training batches in a single epoch")
    
    # Specify resolution of dataset
    parser.add_argument("--use_hires", default=False, type=bool, help="Option to use high resolution dataset")
    parser.add_argument("--default_res", default=128, type=int, help="Specify the default resolution of training data")
    parser.add_argument("--alt_res", default=128, type=int, help="Option to adjust resolution of dataset")
    
    return parser


# Add network parameters
def add_network_args(parser):
    
    # Specify network structure
    parser.add_argument("--network", default=1, type=int, help="Specify network structure")
    parser.add_argument("--use_int_count", default=False, action='store_true', help="Use interior count in place of mean")
    parser.add_argument("--coordconv", default=False, action='store_true', help="Specify use of CoordConv")
    parser.add_argument("--use_extra_dropout", default=False, action='store_true', help="Specify use of extra dropout layer")
    parser.add_argument("--symmetric", default=False, action='store_true', help="Option to symmetrize UNET architecture")
    parser.add_argument("--interpolate", default=False, action='store_true', help="Option to use interpolation")
    parser.add_argument("--use_bn", default=False, action='store_true', help="Option to use batch normalization")
    parser.add_argument("--use_inception", default=False, action='store_true', help="Option to use inception layers")
    parser.add_argument("--extend", default=False, action='store_true', help="Option to extend network for higher resolution")
    parser.add_argument("--factor", default=False, action='store_true', help="Option to use factored convolutions")

    # Network hyperparameters
    parser.add_argument("--z_dim", default=175, type=int, help="Dimension of noise vector in latent space")
    parser.add_argument("--d_res", default=4, type=int, help="Resolution of initial reshaped features in decoder")
    parser.add_argument("--e_res", default=8, type=int, help="Resolution of initial reshaped features in decoder")
    parser.add_argument("--d_chans", default=150, type=int, help="Channel count for initial reshaped features in decoder")
    
    return parser    


# Specify default arguments common to all models
def add_default_args(parser):

    # Saving and display options
    parser.add_argument("--data_dir", default="./Setup/DATA/", type=str, help="Directory containing data files")
    parser.add_argument("--display_step", default=1, type=int, help="Step count for displaying progress")
    parser.add_argument("--summary_step", default=50, type=int, help="Step count for saving summaries")
    parser.add_argument("--model_dir", default="./Model/", type=str, help="Directory for saving model config files")
    parser.add_argument("--log_dir", default="logs/", type=str, help="Directory for saving log files")
    parser.add_argument("--checkpoint_step", default=1000, type=int, help="Step count for saving checkpoints")
    parser.add_argument("--checkpoint_dir", default="Checkpoints/", type=str, help="Directory for saving checkpoints")
    parser.add_argument("--plot_step", default=10000, type=int, help="Step count for saving plots of generated images")
    parser.add_argument("--plot_dir", default="predictions/", type=str, help="Directory for saving plots of generated images")
    parser.add_argument("--plot_res", default=64, type=int, help="Resolution to use when saving generated images")
    
    return parser


# Define flags to specify model hyperparameters and training options
def getFlags():

    # Initialize argument parser and add default arguments
    parser = ArgumentParser(description='Argument Parser')
    parser = add_learning_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_data_args(parser)
    parser = add_network_args(parser)
    parser = add_default_args(parser)
    
    # GPU option
    parser.add_argument("--use_gpu", default=True, type=bool, help="Option to use GPU (i.e. prefetch to device)")
    
    # Parse arguments from command line
    args = parser.parse_args()
    return args


