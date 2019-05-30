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

    # Divide tasks into smaller pieces
    subdivision = 2

    def convert(d):
        fast_convert_samples(int(FLAGS.data_count/subdivision), d, resolution=FLAGS.resolution, use_hires=FLAGS.use_hires)
    
    # Create multiprocessing pool
    NumProcesses = 2*FLAGS.cpu_count
    pool = multiprocessing.Pool(processes=NumProcesses)
    
    start_indices = [int(n*FLAGS.data_count/subdivision) for n in range(0,subdivision*FLAGS.cov_count)]
    start_indices = [FLAGS.data_start_count + n for n in start_indices]
    
    #pool.map(convert, [d for d in start_indices])

    print('\n [ Converting Functions ]\n')
    num_tasks = subdivision*FLAGS.cov_count
    for i, _ in enumerate(pool.imap_unordered(convert, [d for d in start_indices]), 1):
        sys.stdout.write('\r  Progress:  {0:.1%}'.format(i/num_tasks))
        sys.stdout.flush()
    print('\n')
