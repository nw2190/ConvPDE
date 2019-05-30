from argparse import ArgumentParser

# Define flags for dataset creation
def getFlags():

    # Initialize argument parser and add default arguments
    parser = ArgumentParser(description='Argument Parser')

    # Option to save solutions to file for comparison (Note: must be False for time comparisons)
    parser.add_argument("--save_solutions", default=False, action="store_true", help="Option to save solutions to file")
    parser.add_argument("--coarse_mesh", default=False, action="store_true", help="Option to use coarse mesh")
        
    parser.add_argument("--time_count", default=1, type=int, help="Number of data files to solve for time tests")
    parser.add_argument("--no_gpu", default=False, action="store_true", help="Specify if GPU is not being used")
    
    # Specify which data resolution to write to TFRecords
    parser.add_argument("--use_default_res", default=False, action="store_true", help="Use standard data resolution for tfrecords")
    
    # Specify CPU count
    parser.add_argument("--cpu_count", default=8, type=int, help="Number of CPUs available for parallelization")
    parser.add_argument("--chol_pools", default=10, type=int, help="Number of pools for cholesky factors to avoid memory accumulation")
    parser.add_argument("--solver_pools", default=10, type=int, help="Number of pools for solver to avoid memory accumulation")

    # Specify existing data available
    parser.add_argument("--tfrecord_training_start", default=0, type=int, help="Start count for TFRecord training files")
    parser.add_argument("--tfrecord_validation_start", default=0, type=int, help="Start count for TFRecord validation files")
    parser.add_argument("--data_start_count", default=0, type=int, help="Number of data samples for each covariance matrix")    
    
    # Specify data directories
    parser.add_argument("--tfrecord_dir", default="./DATA/", type=str, help="Directory for tfrecord files")
    parser.add_argument("--data_dir", default="./Data/", type=str, help="Directory containing data files")
    parser.add_argument("--mesh_dir", default="./Meshes/", type=str, help="Directory containing mesh files")
    parser.add_argument("--soln_dir", default="./Solutions/", type=str, help="Directory containing solution files")

    # Specify parameters for dataset creation
    parser.add_argument("--cov_count", default=20, type=int, help="Number of covariance matrices")
    
    # NOTE: Should be "data_count/10" w.r.t. to definition in "utils/flags.py"
    parser.add_argument("--data_count", default=50, type=int, help="Number of data samples for each covariance matrix")
    #parser.add_argument("--data_count", default=50, type=int, help="Number of data samples for each covariance matrix")
    
    #parser.add_argument("--resolution", default=64, type=int, help="Resolution for data files")
    #parser.add_argument("--resolution", default=100, type=int, help="Resolution for data files")
    parser.add_argument("--resolution", default=128, type=int, help="Resolution for data files")
    #parser.add_argument("--mesh_resolution", default=25, type=int, help="Resolution for FEniCS solver")
    parser.add_argument("--mesh_resolution", default=35, type=int, help="Resolution for FEniCS solver")

    #parser.add_argument("--vertex_min", default=4, type=int, help="Minimum number of vertices for domain")
    #parser.add_argument("--vertex_max", default=10, type=int, help="Maximum number of vertices for domain"
    parser.add_argument("--vertex_min", default=4, type=int, help="Minimum number of vertices for domain")
    parser.add_argument("--vertex_max", default=16, type=int, help="Maximum number of vertices for domain")

    # Parse arguments from command line
    args = parser.parse_args()
    return args

