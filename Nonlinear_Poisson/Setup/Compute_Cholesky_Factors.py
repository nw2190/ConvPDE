#import pycuda.gpuarray as gpuarray
#import skcuda.linalg as linalg
#import pycuda.autoinit

import os
import sys
import numpy as np
import multiprocessing

#import ctypes

# Import flags specifying dataset parameters
from setup_flags import getFlags

# Import function for generating covariance matrices
from sample_gaussian import generate_covariance


if __name__ == '__main__':
    FLAGS = getFlags()

    #x=ctypes.cdll.LoadLibrary('libcublas.so')
    #x.cublasInit()


    def gen_cov(d):
        generate_covariance(-1.0, 1.0, resolution=FLAGS.resolution, filename=d[0], alpha=1.0, length=d[1])

    length_scales = [0.2, 0.2125, 0.225, 0.24, 0.25, 0.2625, 0.275, 0.2875, 0.3, 0.325,
                     0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.533, 0.566, 0.6]

    start_indices = [n*FLAGS.data_count for n in range(0,FLAGS.cov_count)]
    filenames = ['./Cholesky/L' + str(n) + '.npy' for n in range(0,FLAGS.cov_count)]

    # Make folder to store cholesky factors
    if not os.path.exists('./Cholesky/'):
        os.makedirs('./Cholesky/')


    # Create multiprocessing pool    
    #NumProcesses = FLAGS.cpu_count - 1
    NumProcesses = int(np.floor(FLAGS.cpu_count/4))
    #NumProcesses = 1
    
    # Split tasks over multiple pools
    POOLS = FLAGS.chol_pools
    tasks_per_pool = int(np.floor(len(start_indices)/POOLS))

    print('\n [ Generating Covariances ]\n')
    for P in range(0,POOLS):
                
        indices = start_indices[P*tasks_per_pool:(P+1)*tasks_per_pool]
        files = filenames[P*tasks_per_pool:(P+1)*tasks_per_pool]
                                
        pool = multiprocessing.Pool(processes=NumProcesses)

        num_tasks = FLAGS.cov_count


        #print([d for d in zip(filenames, length_scales)])
        #pool.map(gen_cov, [d for d in zip(filenames, length_scales)])
        #num_tasks = FLAGS.cov_count
        #for i, _ in enumerate(pool.imap_unordered(gen_cov, [d for d in zip(filenames, length_scales)]), 1):
        for i, _ in enumerate(pool.imap_unordered(gen_cov, [d for d in zip(files, indices)]), 1):
            sys.stdout.write('\r  Progress:  {0:.1%}'.format((P*tasks_per_pool + i)/num_tasks))
            sys.stdout.flush()
            
        pool.close()
        pool.join()

    print('\n')
