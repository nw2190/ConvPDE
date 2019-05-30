import os
import sys
import numpy as np
import multiprocessing

# Import flags specifying dataset parameters
from setup_flags import getFlags

# Import sampling and conversion functions
from sample_gaussian import fast_convert_samples, sample_gaussian


if __name__ == '__main__':
    FLAGS = getFlags()

    def sample(d):
        sample_gaussian(FLAGS.data_count, d[0], resolution=FLAGS.resolution, filename=d[1])

    # Check data directories
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    if not os.path.exists(FLAGS.mesh_dir):
        os.makedirs(FLAGS.mesh_dir)
        
    if not os.path.exists(FLAGS.soln_dir):
        os.makedirs(FLAGS.soln_dir)

    # Create multiprocessing pool
    #NumProcesses = FLAGS.cpu_count
    NumProcesses = int(np.floor(FLAGS.cpu_count//2))
    pool = multiprocessing.Pool(processes=NumProcesses)

    start_indices = [n*FLAGS.data_count for n in range(0,FLAGS.cov_count)]
    start_indices = [FLAGS.data_start_count + n for n in start_indices]
    filenames = ['./Cholesky/L' + str(n) + '.npy' for n in range(0,FLAGS.cov_count)]

    #print([d for d in zip(start_indices, filenames)])
    #pool.map(sample, [d for d in zip(start_indices, filenames)])
    print('\n [ Sampling Functions ]\n')
    num_tasks = FLAGS.cov_count
    for i, _ in enumerate(pool.imap_unordered(sample, [d for d in zip(start_indices, filenames)]), 1):
        sys.stdout.write('\r  Progress:  {0:.1%}'.format(i/num_tasks))
        sys.stdout.flush()
    print('\n')


